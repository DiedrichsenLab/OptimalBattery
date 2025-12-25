import numpy as np
from sklearn import metrics
import OptimalBattery.util as ut
import os
import pandas as pd
import OptimalBattery.evaluate as ev
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import DataSetMDTB
import OptimalBattery.construct as ct
import OptimalBattery.plot as plot
from OptimalBattery.global_config import data_dir,save_dir
import torch as pt
import matplotlib.pyplot as plt

rois = {
'all':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
}


# define atlas and dirs
space = 'SUIT3'
atlas,_= am.get_atlas(atlas_str=space)
func_fus_dir = os.path.join(data_dir, 'FunctionalFusion_new')
cerebellum_dir = os.path.join(data_dir, 'Cerebellum')


# Load group parcellation
atlas_dir = f'{func_fus_dir}/Atlases/tpl-SUIT'
model_name = f'{atlas_dir}/atl-NettekovenSym32_space-SUIT_dseg.nii'
nettekoven_parcellation = atlas.read_data(model_name)

subj = None

# Load data
MDTB_dataset = DataSetMDTB(f'{func_fus_dir}/MDTB')

data_mdtb_s1_run,info_mdtb_1_run  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondRun',subj=subj)
data_mdtb_s1_run[np.isnan(data_mdtb_s1_run)] = 0

cond_names = info_mdtb_1_run[:32].cond_name.values



for roi_name , parcels in rois.items():
    # Make the ROI mask
    ROI_mask = np.isin(nettekoven_parcellation, parcels).astype(int)
    ROI_indices = np.where(ROI_mask)[0]

    # get the G matrix
    G_Lib = ct.get_G(data= data_mdtb_s1_run[:,:,ROI_indices],n_cond=32,n_part=16)

    D = ct.build_combinations(G_Lib, strategy='random',n_batteries=100000,n_tasks=8,seed = None,replacement=False,rest_idx= None)
    D_best = ct.choose_combination(D,'log_det')
    D_best_idx = D_best.combination.iloc[0]
    combo = cond_names[list(D_best_idx)]
    print(f"Cerebellum best combo 8 tasks: {combo}")

    D = ct.build_combinations(G_Lib, strategy='random',n_batteries=100000,n_tasks=6,seed = None,replacement=False,rest_idx= None)
    D_best = ct.choose_combination(D,'log_det')
    D_best_idx = D_best.combination.iloc[0]
    combo = cond_names[list(D_best_idx)]
    print(f"Cerebellum best combo 6 tasks: {combo}")

    D = ct.build_combinations(G_Lib, strategy='random',n_batteries=100000,n_tasks=4,seed = None,replacement=False,rest_idx= None)
    D_best = ct.choose_combination(D,'log_det')
    D_best_idx = D_best.combination.iloc[0]
    combo = cond_names[list(D_best_idx)]
    print(f"Cerebellum best combo 4 tasks: {combo}")
    

# plot and save
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(G_Lib)

ax.set_xticks(np.arange(len(cond_names)))
ax.set_yticks(np.arange(len(cond_names)))

ax.set_xticklabels(cond_names, rotation=90, fontsize=6)
ax.set_yticklabels(cond_names, fontsize=6)

plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
fig.savefig(f"{save_dir}/supp/region_cov/cov_cerebellum.pdf", bbox_inches="tight")
plt.show()
