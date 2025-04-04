import numpy as np
import matplotlib.pyplot as plt
import construct as ut
import os
import pickle
import PcmPy as pcm
import seaborn as sns
import pandas as pd
import OptimalBattery.evaluate as ev
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import DataSetMDTB
from IndividualParcellation.global_config import *
import OptimalBattery.estimate as es
import OptimalBattery.util as ut
import OptimalBattery.construct as ct
import OptimalBattery.plot as plot
import OptimalBattery.estimate as et
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import SUITPy as suit



# define atlas and dirs
space = 'SUIT3'
atlas,_= am.get_atlas(atlas_str=space)
base_dir = 'Y:/data/'
if not os.path.exists(base_dir):
    base_dir = '/cifs/diedrichsen/data/'

func_fus_dir = os.path.join(base_dir, 'FunctionalFusion')
cerebellum_dir = os.path.join(base_dir, 'Cerebellum')
device = 'cuda' if pt.cuda.is_available() else 'cpu'

subj= None


MDTB_dataset = DataSetMDTB(f'{func_fus_dir}/MDTB')

data_mdtb_s1_run,info_mdtb_1_run  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondRun',subj=subj)
data_mdtb_s1_run[np.isnan(data_mdtb_s1_run)] = 0

data_mdtb_s1_all,info_mdtb_1_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondAll',subj=subj)
data_mdtb_s1_all[np.isnan(data_mdtb_s1_all)] = 0


data_mdtb_s2_all,info_mdtb_2_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondAll',subj=subj)
data_mdtb_s2_all[np.isnan(data_mdtb_s2_all)] = 0

task_names_s1  = info_mdtb_1_all['cond_name'].unique()
task_names_s2  = info_mdtb_2_all['cond_name'].unique()

# load condition dataframe
condition_df= ct.get_condition_indices(info_mdtb_1_run)

region_mapping = {
    (1, 2, 3, 4): 1,
    (5, 6, 7): 2,
    (8, 9, 10, 11): 3,
    (12, 13, 14, 15, 16): 4,
    (17, 18, 19, 20): 5,  
    (21, 22, 23): 6,
    (24, 25, 26, 27): 7,
    (28, 29, 30, 31, 32): 8
}

# roi (full cerebellum)
atlas_dir = f'{func_fus_dir}/Atlases/tpl-SUIT'
model_name = f'{atlas_dir}/atl-NettekovenSym32_space-SUIT_dseg.nii'
parcelation_32 = atlas.read_data(model_name)

flat_mapping = {k: v for keys, v in region_mapping.items() for k in keys}

# Vectorized mapping function
parcelation_8 = np.vectorize(lambda x: flat_mapping.get(int(x), x))(parcelation_32)

# params
ROI_to_include = [4,5,7,8] 
# make a mask of the cerebellum
ROI_mask = np.isin(parcelation_8, ROI_to_include).astype(int)
ROI_indices = np.where(ROI_mask == 1)[0]


# make variables torch
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
data_mdtb_s1_all = pt.tensor(data_mdtb_s1_all, dtype=pt.float32, device=device)
data_test = pt.tensor(data_mdtb_s2_all, dtype=pt.float32, device=device)
data_train = pt.tensor(data_mdtb_s1_run, dtype=pt.float32, device=device)
parcelation = pt.tensor(parcelation_8, dtype=pt.float32, device=device)
ROI_mask = pt.tensor(ROI_mask, dtype=pt.float32, device=device)

# Get G Lib
G_Lib = ct.get_G(data=data_mdtb_s1_run[:,:,ROI_indices],n_cond=29,n_part=16)

# estimate Vs for training using s2 full data
full_vs_train = es.estimate_Vs(data_mdtb_s1_all,parcellation=parcelation,ROI_mask= ROI_mask)
full_vs_train = ut.center_matrix(full_vs_train,axis=0)
full_vs_train = ut.normalize_matrix(full_vs_train,axis=0)

# D_multi = ev.real_localization_multi(G_Lib,condition_df,
#                         data_train,full_vs_train,data_test,
#                         evaluation_indices = None,
#                         battery_sizes = [3,4,5,6,7,8,9,10,12,14,16],
#                         metric = 'log_det_mc',
#                         n_batteries = 1000,
#                         n_iter=20,
#                         rest_idx = 28,
#                         localizer_duration=8,
#                         target_parcel_idx= 2)

# save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
# save_path = os.path.join(save_dir, 'real_localization_multi.tsv')
# D_multi.to_csv(save_path, sep='\t', index=False)



regiona_idx = 2
regionb_idx = 3

D_single = ev.real_localization_single(data_test,regiona_idx,regionb_idx,
                             full_vs_train,condition_df,data_train,ROI_indices,
                             thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95,0.99])

save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
save_path = os.path.join(save_dir, 'real_localization_single.tsv')
D_single.to_csv(save_path, sep='\t', index=False)