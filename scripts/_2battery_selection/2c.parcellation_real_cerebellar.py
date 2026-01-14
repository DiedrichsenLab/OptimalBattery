import numpy as np
import OptimalBattery.util as ut
import os
import pandas as pd
import OptimalBattery.evaluate as ev
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import DataSetMDTB
import OptimalBattery.construct as ct
import OptimalBattery.plot as plot
from OptimalBattery.global_config import data_dir
import torch as pt


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

subj = ['sub-03','sub-04']

# Load data
MDTB_dataset = DataSetMDTB(f'{func_fus_dir}/MDTB')

data_mdtb_s2_run,info_mdtb_2_run  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondRun',subj=subj)
data_mdtb_s2_run[np.isnan(data_mdtb_s2_run)] = 0

data_mdtb_s2_all,info_mdtb_2_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondAll',subj=subj)
data_mdtb_s2_all[np.isnan(data_mdtb_s2_all)] = 0

data_mdtb_s1_all,info_mdtb_1_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondAll',subj=subj)
data_mdtb_s1_all[np.isnan(data_mdtb_s1_all)] = 0

data_mdtb_s2_run, info_mdtb_2_run = ut.recenter_data(data_mdtb_s2_run,info_mdtb_2_run,center_full_code='rest_task',keep_center= True)
data_mdtb_s2_all, info_mdtb_2_all = ut.recenter_data(data_mdtb_s2_all,info_mdtb_2_all,center_full_code='rest_task',keep_center= True)

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
    G_Lib = ct.get_G(data= data_mdtb_s2_run[:,:,ROI_indices],n_cond=32,n_part=16)

    # make variables torch
    device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
    data_mdtb_s2_all = pt.tensor(data_mdtb_s2_all, dtype=pt.float32, device=device)
    data_test = pt.tensor(data_mdtb_s1_all, dtype=pt.float32, device=device)
    data_train = pt.tensor(data_mdtb_s2_run, dtype=pt.float32, device=device)
    parcelation = pt.tensor(nettekoven_parcellation, dtype=pt.float32, device=device)
    ROI_mask = pt.tensor(ROI_mask, dtype=pt.float32, device=device)

    
    D = ev.real_parcellation(G_Lib,condition_df,
                    data_train,
                    data_mdtb_s2_all, parcelation,
                    data_test,
                    ROI_mask,
                    evaluation_indices = ROI_indices,
                    battery_sizes = [3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                    metrics  = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc'],
                    n_batteries = 20000,
                    n_iter=100,
                    rest_idx = 31,
                    localizer_duration=8)
    

    D['roi'] = roi_name
    D['n_parcel'] = len(parcels)

    averaged_df = plot.average_per_subject(D,'cos_sim') # lists of subjects are averaged across iterations
    long_df = averaged_df.explode('avg_cos_sim_per_subject') # expands the list of subjects into rows for each subject
    long_df['avg_cos_sim_per_subject'] = long_df['avg_cos_sim_per_subject'].astype(float) # turns each correaltion for each subject into a float
        
    all_results.append(long_df)

# Concatenate all results into a single DataFrame
final_results_df = pd.concat(all_results, ignore_index=True)

# Save the results
save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
save_path = os.path.join(save_dir, 'parcellation_real_cerebellum.tsv')
final_results_df.to_csv(save_path, sep='\t', index=False)

