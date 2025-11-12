import Functional_Fusion.reliability as rel
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import DataSetMDTB
from OptimalBattery.global_config import data_dir,save_dir
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# define atlas and dirs
space = 'fs32k'
atlas,_= am.get_atlas(atlas_str=space)

  
# Load data
MDTB_dataset = DataSetMDTB(f'{data_dir}/FunctionalFusion_new/MDTB')

subj = None

data_mdtb_s1_run,info_mdtb_1_run  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondRun',subj=subj)
data_mdtb_s1_run[np.isnan(data_mdtb_s1_run)] = 0

n_cond = info_mdtb_1_run['cond_num'].nunique()
n_part = info_mdtb_1_run['run'].nunique()
cond_vec = np.tile(np.arange(1, n_cond + 1), n_part)
part_vec = np.repeat(np.arange(1, n_part + 1), n_cond)

var = rel.decompose_subj_group(data_mdtb_s1_run, cond_vec, part_vec,separate='subject_wise')

combined = (var[:, 0] + var[:, 1]).tolist()
snr_list = combined

sns.histplot(snr_list, bins=10, kde=True)
plt.xlabel("SNR")
plt.ylabel("Frequency")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(f"{save_dir}/single_vs_multi/fSNR_distribution.pdf")
