import numpy as np
import torch as pt
import seaborn as sns
import matplotlib.pyplot as plt
from Functional_Fusion.dataset import DataSetMDTB
import PcmPy as pcm
from scipy.stats import pearsonr
from OptimalBattery.global_config import data_dir,save_dir
import OptimalBattery.util as ut
# Constants
base_dir = f'{data_dir}/FunctionalFusion_new'
space = 'fs32k'
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')


# -------------------------------what is done in the paper analyses (Battery selection (emperical) sections)--------------------------
# Load data
MDTB_dataset = DataSetMDTB(f'{base_dir}/MDTB')
data_run,info_run  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondRun')
data_run, info_run = ut.recenter_data(data_run, info_run, center_full_code='rest_task', keep_center=True)
data_run[np.isnan(data_run)] = 0
task_names = info_run['cond_name'][:32]


# prep cond and part vecs
cond_vec = np.tile(np.arange(1, 32 + 1), 16)
part_vec = np.repeat(np.arange(1, 16 + 1), 32)


# get the individual cov matrices and average 
Gs_list = []
for i in range(data_run.shape[0]):
    Gs,_ = pcm.util.est_G_crossval(data_run[i], cond_vec, part_vec)
    Gs_list.append(Gs)
Gs_list = np.stack(Gs_list, 0)
G_individuals_averaged = np.mean(Gs_list, axis=0)



# -------------------------------what is done when making task libraries for future use--------------------------

# Load data
MDTB_dataset = DataSetMDTB(f'{base_dir}/MDTB')
data_all,info_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondAll')
data_all, info_all = ut.recenter_data(data_all, info_all, center_full_code='rest_task', keep_center=True)
data_all[np.isnan(data_all)] = 0
task_names = info_all['cond_name'][:32]


data_averaged = np.mean(data_all,axis=0)
G_group = data_averaged @ data_averaged.T


def upper_tri_flat(mat):
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]

group_flat = upper_tri_flat(G_group)
ind_flat   = upper_tri_flat(G_individuals_averaged)
corr_group_ind, p_group_ind = pearsonr(group_flat, ind_flat)


fig, axes = plt.subplots(1, 2, figsize=(18, 5))

# 1. Individual (CV)
im0 = axes[0].imshow(G_individuals_averaged, cmap='viridis')
axes[0].set_title("Individual Covariances averaged - CV")
# x and y labels
axes[0].set_xticks(np.arange(len(task_names)))
axes[0].set_yticks(np.arange(len(task_names)))
axes[0].set_xticklabels(task_names, rotation=90, fontsize=6)
axes[0].set_yticklabels(task_names, fontsize=6)
plt.colorbar(im0, ax=axes[0])

# 2. Group  (no CV)
im1 = axes[1].imshow(G_group, cmap='viridis')
axes[1].set_title(
    f"Group covariance - no CV\n"
    f"corr = {corr_group_ind:.2f}, p = {p_group_ind:.1e}"
)
# x and y labels
axes[1].set_xticks(np.arange(len(task_names)))
axes[1].set_yticks(np.arange(len(task_names)))
axes[1].set_xticklabels(task_names, rotation=90, fontsize=6)
axes[1].set_yticklabels(task_names, fontsize=6)
plt.colorbar(im1, ax=axes[1])


plt.tight_layout()
plt.savefig(f"{save_dir}/supp/group_vs_individual_cov/group_vs_indi_cov.pdf")
plt.show()
