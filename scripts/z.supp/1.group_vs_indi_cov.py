from IndividualParcellation.global_config import *
import numpy as np
import torch as pt
import seaborn as sns
import matplotlib.pyplot as plt
from Functional_Fusion.dataset import DataSetMDTB
import PcmPy as pcm
from scipy.stats import pearsonr
from OptimalBattery.global_config import data_dir,save_dir

# Constants
base_dir = f'{data_dir}/FunctionalFusion_new'
space = 'SUIT3'
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')



# Load data
MDTB_dataset = DataSetMDTB(f'{base_dir}/MDTB')
data_mdtb_s1,info_mdtb_1  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondRun')
data_mdtb_s1[np.isnan(data_mdtb_s1)] = 0
task_names = info_mdtb_1['cond_name'][:29]


# prep cond and part vecs
cond_vec = np.tile(np.arange(1, 29 + 1), 16)
part_vec = np.repeat(np.arange(1, 16 + 1), 29)


# get averaged data cov matrix - > group cov
avg_group_data = np.mean(data_mdtb_s1, axis=0)
G_group,E_group = pcm.util.est_G_crossval(avg_group_data, cond_vec, part_vec)


# get the individual cov matrices and average --> individual cov
Gs_list = []
E_list = []
for i in range(data_mdtb_s1.shape[0]):
    Gs,E = pcm.util.est_G_crossval(data_mdtb_s1[i], cond_vec, part_vec)
    Gs_list.append(Gs)
    E_list.append(E)
Gs_list = np.stack(Gs_list, 0)
G_individuals_averaged = np.mean(Gs_list, axis=0)


# regress out group pattern from each individual to get residuals
residuals = []
for i in range(data_mdtb_s1.shape[0]):
    individual_pattern = data_mdtb_s1[i]  
    
    # individual_pattern = beta * group_avg_pattern + residual
    individual_flat = individual_pattern.flatten()
    group_flat = avg_group_data.flatten()
    
    # Calculate beta
    beta = np.sum(group_flat * individual_flat) / np.sum(group_flat**2)
    
    # Compute the residual pattern
    individual_residual = individual_pattern - beta * avg_group_data
    
    residuals.append(individual_residual)

residuals = np.stack(residuals, axis=0)

# each of the residuals is uncorrelated with the group pattern


#Calculate Covariance Matrix for residuals - > residual cov
residual_covariances = []
for i in range(residuals.shape[0]):
    individual_residual = residuals[i]  # n x p
    G_s,E_s = pcm.util.est_G_crossval(individual_residual, cond_vec, part_vec)
    residual_covariances.append(G_s) 

# Avg the 2nd moments of individual residuals
residual_cov_mean = np.mean(np.stack(residual_covariances, axis=0), axis=0)



def upper_tri_flat(mat):
    idx = np.triu_indices_from(mat, k=1)
    return mat[idx]

group_flat = upper_tri_flat(G_group)
ind_flat   = upper_tri_flat(G_individuals_averaged)
res_flat   = upper_tri_flat(residual_cov_mean)

# get corr
corr_group_ind, p_group_ind = pearsonr(group_flat, ind_flat)
corr_group_res, p_group_res = pearsonr(group_flat, res_flat)


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Group covariance
im0 = axes[0].imshow(G_group, cmap='viridis')
axes[0].set_title("Group Covariance")
plt.colorbar(im0, ax=axes[0])

# 2. Average Individual covariance
im1 = axes[1].imshow(G_individuals_averaged, cmap='viridis')
axes[1].set_title(
    f"Individual Covariance\n"
    f"corr = {corr_group_ind:.2f}, p = {p_group_ind:.1e}"
)
plt.colorbar(im1, ax=axes[1])

# 3. Residual covariance
im2 = axes[2].imshow(residual_cov_mean, cmap='viridis')
axes[2].set_title(
    f"Residual Covariance\n"
    f"corr = {corr_group_res:.2f}, p = {p_group_res:.1e}"
)
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig(f"{save_dir}/supp/group_vs_indi_residual_cov.pdf")
plt.show()
