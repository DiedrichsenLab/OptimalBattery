import numpy as np
import matplotlib.pyplot as plt
import OptimalBattery.util as ut
import os
import PcmPy as pcm
import seaborn as sns
import pandas as pd
import OptimalBattery.evaluate as ev
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import DataSetMDTB
from IndividualParcellation.global_config import *
import nilearn.plotting as plotting
import nitools as nt
import nibabel as nb
import OptimalBattery.estimate as es
import OptimalBattery.construct as ct
import OptimalBattery.plot as plot

battery_sizes = [3,4,5,6,7,8,10,14,16]
rois = {
    'SD': [8,9,10,11,12,13,14,15,16,24,25,26,27,28,29,30,31,32],
    'MA':[1,2,3,4,5,6,7,17,18,19,20,21,22,23],
    'SDR':[24,25,26,27,28,29,30,31,32],
    'SDL':[8,9,10,11,12,13,14,15,16],
    'MA_L':[1,2,3,4,5,6,7],
    'MA_R':[17,18,19,20,21,22,23],
    'AD':[5,6,7,8,9,10,11,21,22,23,24,25,26,27],
    'rand':[1,4,6,7,8,12,14,16,20,23,26,29,30,31]
}


# define atlas and dirs
space = 'SUIT3'
atlas,_= am.get_atlas(atlas_str=space)
base_dir = 'Y:/data/'
if not os.path.exists(base_dir):
    base_dir = '/cifs/diedrichsen/data/'

func_fus_dir = os.path.join(base_dir, 'FunctionalFusion')
cerebellum_dir = os.path.join(base_dir, 'Cerebellum')


# Load group parcellation
atlas_dir = f'{func_fus_dir}/Atlases/tpl-SUIT'
model_name = f'{atlas_dir}/atl-NettekovenSym32_space-SUIT_dseg.nii'
nettekoven_parcellation = atlas.read_data(model_name)

# Load data
MDTB_dataset = DataSetMDTB(f'{func_fus_dir}/MDTB')

data_mdtb_s2_run,info_mdtb_2_run  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondRun')
data_mdtb_s2_run[np.isnan(data_mdtb_s2_run)] = 0

data_mdtb_s2_all,info_mdtb_2_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondAll')
data_mdtb_s2_all[np.isnan(data_mdtb_s2_all)] = 0

data_mdtb_s1_all,info_mdtb_1_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondAll')
data_mdtb_s1_all[np.isnan(data_mdtb_s1_all)] = 0

data_mdtb_s2_run = ut.recenter_fmri_data(data_mdtb_s2_run,info_mdtb_2_run,task_column_name='cond_name',center_condition='rest')
data_mdtb_s2_all = ut.recenter_fmri_data(data_mdtb_s2_all,info_mdtb_2_all,task_column_name='cond_name',center_condition='rest')

# get condition indices
condition_df= ct.get_condition_indices(info_mdtb_2_run)
corr_list =[]
i = 0
all_results = []
for roi_name , parcels in rois.items():
    print(f'Processing {roi_name}')

    # Make the ROI mask
    ROI_mask = np.isin(nettekoven_parcellation, parcels).astype(int)
    ROI_indices = np.where(ROI_mask)[0]

    # get the G matrix
    G_Lib = ct.get_G(data= data_mdtb_s2_run[:,:,ROI_indices],n_cond=29,n_part=16)

    # make variables torch
    device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
    data_mdtb_s2_all = pt.tensor(data_mdtb_s2_all, dtype=pt.float32, device=device)
    data_test = pt.tensor(data_mdtb_s1_all, dtype=pt.float32, device=device)
    data_train = pt.tensor(data_mdtb_s2_run, dtype=pt.float32, device=device)
    parcelation = pt.tensor(nettekoven_parcellation, dtype=pt.float32, device=device)
    ROI_mask = pt.tensor(ROI_mask, dtype=pt.float32, device=device)

    # estimate Vs for training using s2 full data
    full_vs_train = es.estimate_Vs(data_mdtb_s2_all,parcellation=parcelation,ROI_mask= ROI_mask)
    full_vs_train = ut.center_matrix(full_vs_train,axis=0)
    full_vs_train = ut.normalize_matrix(full_vs_train,axis=0)

    # estimate Vs for testing using s1 full data
    full_vs_test = es.estimate_Vs(data_test,parcellation=parcelation,ROI_mask=ROI_mask)
    full_vs_test = ut.center_matrix(full_vs_test,axis=0)
    full_vs_test = ut.normalize_matrix(full_vs_test,axis=0)

    n_parcels = full_vs_train.shape[1]

    D = ev.real_parcellation(G_Lib,condition_df,
                        data_train,data_test,
                        full_vs_train,full_vs_test,
                        evaluation_indices = ROI_indices,
                        battery_sizes = [3,4,5,6,7,8,9,10,14,16],
                        metrics  = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc'],
                        n_batteries = 20000,
                        n_iter=20,
                        rest_idx = 28,
                        localizer_duration=8)
    
    D['roi'] = roi_name
    D['parcels'] = ', '.join(parcels)
    D['n_parcel'] = len(parcels)
        
    all_results.append(D)

# Concatenate all results into a single DataFrame
final_results_df = pd.concat(all_results, ignore_index=True)

# Save the results
save_dir = os.path.abspath(os.path.join(os.getcwd(), 'notebooks','eval_tsvs'))
save_path = os.path.join(save_dir, 'real_parcellation_cerebellum.tsv')
final_results_df.to_csv(save_path, sep='\t', index=False)

