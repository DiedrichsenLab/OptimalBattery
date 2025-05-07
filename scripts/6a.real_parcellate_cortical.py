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

rois = {
'visual_cortex': [
    "V1", "V2", "V3", "V3A", "V3B", "V3CD", "V4", "V4t", "V6", "V6A", "V7", "V8",
    "MT", "MST", "FST", "LO1", "LO2", "LO3", "PIT",
    "STV", "PCV", "DVT", "VMV1", "VMV2", "VMV3",
    "VVC","FFC","RSC","POS1", "POS2","IPS1","ProS" ],       
                        

    "Non-Visual Cortex": 
     [
    "4", "3b", "FEF", "PEF", "55b", "A1", "PSL", "SFL", "7Pm", "7m",
    "23d", "v23ab", "d23ab", "31pv", "5m", "5mv", "23c", "5L", "24dd", "24dv",
    "7AL", "SCEF", "6ma", "7Am", "7PL", "7PC", "LIPv", "VIP", "MIP", "1", "2",
    "3a", "6d", "6mp", "6v", "p24pr", "33pr", "a24pr", "p32pr", "a24", "d32",
    "8BM", "p32", "10r", "47m", "8Av", "8Ad", "9m", "8BL", "9p", "10d", "8C",
    "44", "45", "47l", "a47r", "6r", "IFJa", "IFJp", "IFSp", "IFSa", "p9-46v",
    "46", "a9-46v", "9-46d", "9a", "10v", "a10p", "10pp", "11l", "13l", "OFC",
    "47s", "LIPd", "6a", "i6-8", "s6-8", "43", "OP4", "OP1", "OP2-3", "52", "RI",
    "PFcm", "PoI2", "TA2", "FOP4", "MI", "Pir", "AVI", "AAIC", "FOP1", "FOP3",
    "FOP2", "PFt", "AIP", "EC", "PreS", "H", "PeEc", "STGa", "PBelt", "A5",
    "PHA1", "PHA3", "STSda", "STSdp", "STSvp", "TGd", "TE1a", "TE1p", "TE2a",
    "TF", "TE2p", "PHT", "PH", "TPOJ1", "TPOJ2", "TPOJ3", "PGp", "IP2", "IP1",
    "IP0", "PFop", "PF", "PFm", "PGi", "PGs", "PHA2", "25", "s32", "pOFC",
    "PoI1", "Ig", "FOP5", "p10p", "p47r", "TGv", "MBelt", "LBelt", "A4",
    "STSva", "TE1m", "PI", "a32pr", "p24","31pd","31a"
],

    "All Parcels":
    ["V1", "MST", "V6", "V2", "V3", "V4", "V8", "4", "3b", "FEF", "PEF", "55b", "V3A", "RSC", "POS2", "V7",
    "IPS1", "FFC", "V3B", "LO1", "LO2", "PIT", "MT", "A1", "PSL", "SFL", "PCV", "STV", "7Pm", "7m", "POS1",
    "23d", "v23ab", "d23ab", "31pv", "5m", "5mv", "23c", "5L", "24dd", "24dv", "7AL", "SCEF", "6ma", "7Am",
    "7PL", "7PC", "LIPv", "VIP", "MIP", "1", "2", "3a", "6d", "6mp", "6v", "p24pr", "33pr", "a24pr", "p32pr",
    "a24", "d32", "8BM", "p32", "10r", "47m", "8Av", "8Ad", "9m", "8BL", "9p", "10d", "8C", "44", "45",
    "47l", "a47r", "6r", "IFJa", "IFJp", "IFSp", "IFSa", "p9-46v", "46", "a9-46v", "9-46d", "9a", "10v",
    "a10p", "10pp", "11l", "13l", "OFC", "47s", "LIPd", "6a", "i6-8", "s6-8", "43", "OP4", "OP1", "OP2-3",
    "52", "RI", "PFcm", "PoI2", "TA2", "FOP4", "MI", "Pir", "AVI", "AAIC", "FOP1", "FOP3", "FOP2", "PFt",
    "AIP", "EC", "PreS", "H", "ProS", "PeEc", "STGa", "PBelt", "A5", "PHA1", "PHA3", "STSda", "STSdp",
    "STSvp", "TGd", "TE1a", "TE1p", "TE2a", "TF", "TE2p", "PHT", "PH", "TPOJ1", "TPOJ2", "TPOJ3", "DVT",
    "PGp", "IP2", "IP1", "IP0", "PFop", "PF", "PFm", "PGi", "PGs", "V6A", "VMV1", "VMV3", "PHA2", "V4t",
    "FST", "V3CD", "LO3", "VMV2", "31pd", "31a", "VVC", "25", "s32", "pOFC", "PoI1", "Ig", "FOP5", "p10p",
    "p47r", "TGv", "MBelt", "LBelt", "A4", "STSva", "TE1m", "PI", "a32pr", "p24"

    ],

    "PFC":
    ['OFC', '10pp', '10r', '8C', 's6-8', '25', 'p24', 'p47r', '46', 'a10p', '10d', '9m', '8Av', 'IFJp', 
    '10v', '13l', '45', 'i6-8', '9-46d', 'IFJa', '47s', 'SFL', 'a24', 'IFSp', '47m', '9p', '9a', 'pOFC',
    '8Ad', '11l', 'IFSa', 'a9-46v', '44', 'a47r', '55b', '47l', 's32', 'p9-46v', '8BM', 'p10p', '8BL', 'p32', 'a32pr', 'd32']

}



for key,value in rois.items():
    print(key)
    print(len(value))


# define atlas and dirs
space = 'fs32k'
atlas,_= am.get_atlas(atlas_str=space)
base_dir = 'Y:/data/'
if not os.path.exists(base_dir):
    base_dir = '/cifs/diedrichsen/data/'

func_fus_dir = os.path.join(base_dir, 'FunctionalFusion')
cerebellum_dir = os.path.join(base_dir, 'Cerebellum')


# roi (prefrontal cortex)
atlas_dir = f'{func_fus_dir}/Atlases/tpl-fs32k'
model_name_L = f'{atlas_dir}/glasser.L.label.gii'
model_name_R = f'{atlas_dir}/glasser.R.label.gii'
glasser_atlas = atlas.read_data([model_name_L,model_name_R])
  
# Load data
MDTB_dataset = DataSetMDTB(f'{func_fus_dir}/MDTB')

subj = None

data_mdtb_s2_run,info_mdtb_2_run  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondRun',subj=subj)
data_mdtb_s2_run[np.isnan(data_mdtb_s2_run)] = 0

data_mdtb_s2_all,info_mdtb_2_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondAll',subj=subj)
data_mdtb_s2_all[np.isnan(data_mdtb_s2_all)] = 0

data_mdtb_s1_all,info_mdtb_1_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondAll',subj=subj)
data_mdtb_s1_all[np.isnan(data_mdtb_s1_all)] = 0

data_mdtb_s2_run = ut.recenter_fmri_data(data_mdtb_s2_run,info_mdtb_2_run,task_column_name='cond_name',center_condition='rest')
data_mdtb_s2_all = ut.recenter_fmri_data(data_mdtb_s2_all,info_mdtb_2_all,task_column_name='cond_name',center_condition='rest')

# get condition indices
condition_df= ct.get_condition_indices(info_mdtb_2_run)


all_results = []
for roi_name , parcels in rois.items():
    print(f'Processing {roi_name}')

    # Load the GIFTI file
    gifti_data = nb.load(model_name_L)
    parcel_names = [label.label for label in gifti_data.labeltable.labels]
    parcel_names = [name[len("L_"):] if name.startswith("L_") else name for name in parcel_names]
    parcel_names = [name[:-len("_ROI")] if name.endswith("_ROI") else name for name in parcel_names]

    # Get the indices of the PFC parcels
    ROI_cortex = []
    for name in parcels:
        ROI_cortex.append(parcel_names.index(name))

    ROI_mask = np.isin(glasser_atlas, ROI_cortex).astype(int)
    mask_reshaped = ROI_mask[np.newaxis, :]  # Reshape to (1, 59518)
    ROI_indices = np.where(ROI_mask == 1)[0]

    # get the G matrix
    G_Lib = ct.get_G(data= data_mdtb_s2_run[:,:,ROI_indices],n_cond=32,n_part=16)

    # make variables torch
    device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
    data_mdtb_s2_all = pt.tensor(data_mdtb_s2_all, dtype=pt.float32, device=device)
    data_test = pt.tensor(data_mdtb_s1_all, dtype=pt.float32, device=device)
    data_train = pt.tensor(data_mdtb_s2_run, dtype=pt.float32, device=device)
    parcelation = pt.tensor(glasser_atlas, dtype=pt.float32, device=device)
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
                        battery_sizes = [3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                        metrics  = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc'],
                        n_batteries = 10000,
                        n_iter=20,
                        rest_idx = 31,
                        localizer_duration=8)
    
    D['roi'] = roi_name
    D['parcels'] = ', '.join(parcels)
    D['n_parcel'] = len(parcels)

    averaged_df = plot.average_per_subject(D,'cos_sim') # lists of subjects are averaged across iterations
    long_df = averaged_df.explode('avg_cos_sim_per_subject') # expands the list of subjects into rows for each subject
    long_df['avg_cos_sim_per_subject'] = long_df['avg_cos_sim_per_subject'].astype(float) # turns each correaltion for each subject into a float
        
    all_results.append(long_df)

# Concatenate all results into a single DataFrame
final_results_df = pd.concat(all_results, ignore_index=True)

# Save the results
save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
save_path = os.path.join(save_dir, 'real_parcellation_cortex.tsv')
final_results_df.to_csv(save_path, sep='\t', index=False)

