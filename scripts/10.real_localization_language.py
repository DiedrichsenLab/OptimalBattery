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
from Functional_Fusion.dataset import DataSetLanguage
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
import OptimalBattery.simulate as sim



# define atlas and dirs
space = 'SUIT3'
atlas,_= am.get_atlas(atlas_str=space)
base_dir = 'Y:/data/'
if not os.path.exists(base_dir):
    base_dir = '/cifs/diedrichsen/data/'

func_fus_dir = os.path.join(base_dir, 'FunctionalFusion')
cerebellum_dir = os.path.join(base_dir, 'Cerebellum')
device = 'cuda' if pt.cuda.is_available() else 'cpu'

subj= ['sub-02','sub-03','sub-04']


MDTB_dataset = DataSetLanguage(f'{func_fus_dir}/Language')

data_mdtb_s1_run,info_mdtb_1_run  =MDTB_dataset.get_data(space=space,ses_id='ses-localizer_cond',type='CondRun',subj=subj)
data_mdtb_s1_run[np.isnan(data_mdtb_s1_run)] = 0

data_mdtb_s1_all,info_mdtb_1_all  =MDTB_dataset.get_data(space=space,ses_id='ses-localizer_cond',type='CondAll',subj=subj)
data_mdtb_s1_all[np.isnan(data_mdtb_s1_all)] = 0

# remove any spaces in the taskName column
info_mdtb_1_run['taskName'] = info_mdtb_1_run['taskName'].str.replace(' ', '')
info_mdtb_1_all['taskName'] = info_mdtb_1_all['taskName'].str.replace(' ', '')

data_mdtb_s1_run = ut.recenter_fmri_data(data_mdtb_s1_run,info_mdtb_1_run,task_column_name='taskName',center_condition='rest')
data_mdtb_s1_all = ut.recenter_fmri_data(data_mdtb_s1_all,info_mdtb_1_all,task_column_name='taskName',center_condition='rest')

task_names_s1  = info_mdtb_1_all['taskName'].unique()

def get_condition_indices(df):
    """
    Get condition indices from a dataframe and record the duration of each condition
    Parameters:
        df(pd.DataFrame): dataframe containing condition indices needs to include:
            - 'cond_name': name of the condition
            - 'run': run number
            - 'task_name': name of the task
    Returns:
        condition_indices(np.ndarray): condition indices
    """
    unique_conditions = df['taskName'].unique()
    new_df = pd.DataFrame(columns=['taskName', 'indices', 'duration'])
    
    # Filter only the first run
    first_run_df = df[df['run'] == df['run'].min()]
    task_run_counts = first_run_df.groupby('taskName')['taskName'].nunique()
    duration_map = {1: 30, 2: 15, 3: 10}
    
    # Populate the new dataframe
    for condition in unique_conditions:
        indices = df[df['taskName'] == condition].index.tolist()
        
        # Identify task_name for the condition from the original dataframe
        task_name = df[df['taskName'] == condition]['taskName'].values[0]
        num_conditions = task_run_counts.get(task_name, 1)
        duration = duration_map.get(num_conditions, 30)
        
        new_row = {'taskName': condition, 'indices': indices, 'duration': duration}
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return new_df

test_data = data_mdtb_s1_all[:,9,:]
# add dim in middke
test_data = np.expand_dims(test_data, axis=1)
test_data = pt.tensor(test_data, dtype=pt.float32, device=device)

# load condition dataframe
condition_df= get_condition_indices(info_mdtb_1_run)

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
ROI_to_include = [17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
# make a mask of the cerebellum
ROI_mask = np.isin(parcelation_32, ROI_to_include).astype(int)
ROI_indices = np.where(ROI_mask == 1)[0]


# make variables torch
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
data_mdtb_s1_all = pt.tensor(data_mdtb_s1_all, dtype=pt.float32, device=device)
data_train = pt.tensor(data_mdtb_s1_run, dtype=pt.float32, device=device)
parcelation = pt.tensor(parcelation_32, dtype=pt.float32, device=device)
ROI_mask = pt.tensor(ROI_mask, dtype=pt.float32, device=device)

# Get G Lib
G_Lib = ct.get_G(data=data_mdtb_s1_run[:,:,ROI_indices],n_cond=18,n_part=8)

# estimate Vs for training using s2 full data
full_vs_train = es.estimate_Vs(data_mdtb_s1_all,parcellation=parcelation,ROI_mask= ROI_mask)
full_vs_train = ut.center_matrix(full_vs_train,axis=0)
full_vs_train = ut.normalize_matrix(full_vs_train,axis=0)

def real_localization_multi(G_library,condition_df,
                        YLib,VLib,Ytest,
                        evaluation_indices = None,
                        battery_sizes = [3,4,5,6,7,8,9,10,12,14,16],
                        metric = 'log_det_mc',
                        n_batteries = 1000,
                        n_iter=5,
                        rest_idx = 28,
                        localizer_duration=8,
                        target_parcels_indices = 0):
    """ Evaluate the localization performance multitask battery

    """
    results_df = pd.DataFrame()
    for n_task in battery_sizes:
        print(f"Evaluating battery size: {n_task}")
        for n in range(n_iter):
            print(f"Iteration: {n}")
            D = ct.build_combinations(G_library, strategy='random',n_batteries=n_batteries,n_tasks=n_task,seed = None,replacement=False,rest_idx= rest_idx)
            D_best = ct.choose_combination(D,metric)
            top_comb = D_best['combination'].values[0]

            # get the regressors for training data
            combination_regressors = ct.build_combination_regressors(top_comb, condition_df=condition_df, localizer_time=localizer_duration)

            # average, center and normalize the data used for the parcellation
            Ysubset = ct.average_regressors(YLib, combination_regressors)
            Ysubset = ut.center_matrix(Ysubset, axis=1)
            Ysubset = ut.normalize_matrix(Ysubset, axis=1)

            Vsubset = VLib[top_comb,:]
            Vsubset = ut.center_matrix(Vsubset, axis=0)
            Vsubset = ut.normalize_matrix(Vsubset, axis=0)

            Uhats =  et.estimate_Us(Ysubset, Vsubset, method='correlation',hard=True)
            U_hats_collpased = sim.collapse_U(Uhats, target_parcels_indices=target_parcels_indices)
            U_binary  = U_hats_collpased[:,0,:]

            test_profile = ev.compute_region_profiles(Ytest, U_binary)
            corr = ev.average_pairwise_correlation(test_profile)
        
            # record
            D_ev = pd.DataFrame()
            D_ev['n_task'] = [n_task]
            D_ev['metric'] = [metric]
            D_ev['iteration'] = [n]
            D_ev['corr'] = [corr]
            results_df = pd.concat([results_df,D_ev],axis=0)
            results_df.reset_index(drop=True, inplace=True)

    return results_df

D_multi = real_localization_multi(G_Lib,condition_df,
                        data_train,full_vs_train,test_data,
                        evaluation_indices = None,
                        battery_sizes = [5],
                        metric = 'inverse_trace_mc',
                        n_batteries = 20000,
                        n_iter=5,
                        rest_idx = 0,
                        localizer_duration=4,
                        target_parcels_indices= [11,12,13,14,15])

save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
save_path = os.path.join(save_dir, 'real_localization_multi.tsv')
D_multi.to_csv(save_path, sep='\t', index=False)



regiona_idx = 2
regionb_idx = 3

# D_single = ev.real_localization_single(data_test,regiona_idx,regionb_idx,
#                              full_vs_train,condition_df,data_train,ROI_indices,
#                              thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95,0.99])

# save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
# save_path = os.path.join(save_dir, 'real_localization_single.tsv')
# D_single.to_csv(save_path, sep='\t', index=False)