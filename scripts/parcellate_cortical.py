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

battery_sizes = [3,4,6,8,10,14,16]
rois = {
    'Dorsolateral_PFC': ['9-46d', '46', '9a', 'a9-46v', 'p9-46v', '9p', 'IFJa', 'IFJp', 'IFSp', 'IFSa'],
    'Parietal_Multisensory': ['LIPv', 'LIPd', 'VIP', 'MIP', 'AIP', '7PC', '7AL', '7Am'],
    'Posterior_Cingulate': ['d23ab', 'v23ab', '23c', '23d', '31pv', '31pd', '31a']
}


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

data_mdtb_s2_run,info_mdtb_2_run  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondRun')
data_mdtb_s2_run[np.isnan(data_mdtb_s2_run)] = 0

data_mdtb_s2_all,info_mdtb_2_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondAll')
data_mdtb_s2_all[np.isnan(data_mdtb_s2_all)] = 0

data_mdtb_s1_all,info_mdtb_1_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondAll')
data_mdtb_s1_all[np.isnan(data_mdtb_s1_all)] = 0

data_mdtb_s2_run = ut.recenter_fmri_data(data_mdtb_s2_run,info_mdtb_2_run,task_column_name='cond_name',center_condition='rest')
data_mdtb_s2_all = ut.recenter_fmri_data(data_mdtb_s2_all,info_mdtb_2_all,task_column_name='cond_name',center_condition='rest')

# run loop


# get condition indices
condition_df= ct.get_condition_indices(info_mdtb_2_run)
corr_list =[]
i = 0
all_results = []
for roi_name , parcels in rois.items():
    print(f'Processing {roi_name}')
    # define PFC parcel and get the indices of the PFC parcels

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
    cond_vec = np.tile(np.arange(1, 32 + 1), 16)
    part_vec = np.repeat(np.arange(1, 16 + 1), 32)
    Gs_list = []
    E_list = []
    for i in range(data_mdtb_s2_run.shape[0]):
        Gs,E = pcm.util.est_G_crossval(data_mdtb_s2_run[i][:,ROI_indices] , cond_vec, part_vec)
        Gs_list.append(Gs)
        E_list.append(E)

    Gs_list = np.stack(Gs_list, 0)
    G_Lib = np.mean(Gs_list, axis=0)

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

    VLib = full_vs_train[:,:]
    Vtest = full_vs_test[:,:]

    for n_task in battery_sizes:
        print(f'Processing {n_task} tasks')
        D = ct.build_combinations(G_Lib, strategy='random',n_batteries=10000,n_tasks=n_task,seed=1,replacement=False,rest_idx=31)

        D_ev = ev.evluate_dataframe( D = D, condition_df= condition_df, YLib= data_train,
                                    VLib= VLib, Ytest= data_test, Vtest= Vtest,indices=ROI_indices,method='correlation')
        D_ev['roi'] = roi_name
        D_ev['parcels'] = ', '.join(parcels)
        D_ev['n_parcel'] = len(parcels)
        D_ev['n_task'] = n_task
        
        all_results.append(D_ev)

# Concatenate all results into a single DataFrame
final_results_df = pd.concat(all_results, ignore_index=True)

# Save the results
final_results_df.to_csv(f'{base_dir}/OptimalBattery/Eval_tsvs/Cortical_rois.tsv', sep='\t', index=False)

