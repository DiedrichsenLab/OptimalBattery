import Functional_Fusion.reliability as rel
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import DataSetMDTB
from OptimalBattery.global_config import data_dir,save_dir
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gamma


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

snr_list = (var[:, 0] + var[:, 1]).tolist()

shape, loc, scale = gamma.fit(snr_list, floc=0)
sns.histplot(snr_list, bins=10, color="lightblue", edgecolor="black")

# compute x and scaled pdf
x = np.linspace(0, max(snr_list)*1.1, 300)
pdf = gamma.pdf(x, shape, loc=loc, scale=scale)

N = len(snr_list)
bin_width = (max(snr_list) - min(snr_list)) / 10
pdf_scaled = pdf * N * bin_width

plt.plot(x, pdf_scaled, color="blue")
plt.xlabel("fSNR")
plt.ylabel("Frequency")
plt.title(f"alpha: {shape},beta{scale}")
sns.despine()
plt.tight_layout()
plt.savefig(f"{save_dir}/single_vs_multi/fSNR_distribution.pdf")

