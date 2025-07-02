import numpy as np
import OptimalBattery.util as ut
import os
import OptimalBattery.evaluate as ev
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import DataSetMDTB
from IndividualParcellation.global_config import *
import OptimalBattery.construct as ct
import OptimalBattery.plot as plot
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# define dirs
base_dir = 'Y:/data/'
if not os.path.exists(base_dir):
    base_dir = '/cifs/diedrichsen/data/'
func_fus_dir = os.path.join(base_dir, 'FunctionalFusion')
cerebellum_dir = os.path.join(base_dir, 'Cerebellum')

# define atlas and dataset
space = 'SUIT3'
MDTB_dataset = DataSetMDTB(f'{func_fus_dir}/MDTB')
subj = None
suit_atlas,_= am.get_atlas(atlas_str=space)

# load data
data_mdtb_s2_run_suit,info_mdtb_2_run  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondRun',subj=subj)
data_mdtb_s2_run_suit[np.isnan(data_mdtb_s2_run_suit)] = 0

data_mdtb_s2_all_suit,info_mdtb_2_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondAll',subj=subj)
data_mdtb_s2_all_suit[np.isnan(data_mdtb_s2_all_suit)] = 0

data_mdtb_s1_all_suit,info_mdtb_1_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondAll',subj=subj)
data_mdtb_s1_all_suit[np.isnan(data_mdtb_s1_all_suit)] = 0


space ='fs32k'
fs_atlas,_= am.get_atlas(atlas_str=space)
data_mdtb_s2_run_fs,info_mdtb_2_run  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondRun',subj=subj)
data_mdtb_s2_run_fs[np.isnan(data_mdtb_s2_run_fs)] = 0

data_mdtb_s2_all_fs,info_mdtb_2_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondAll',subj=subj)
data_mdtb_s2_all_fs[np.isnan(data_mdtb_s2_all_fs)] = 0

data_mdtb_s1_all_fs,info_mdtb_1_all  =MDTB_dataset.get_data(space=space,ses_id='ses-s2',type='CondAll',subj=subj)
data_mdtb_s1_all_fs[np.isnan(data_mdtb_s1_all_fs)] = 0

# recenter data around rest for eigenmetric evaluation
data_mdtb_s2_run_fs = ut.recenter_fmri_data(data_mdtb_s2_run_fs,info_mdtb_2_run,task_column_name='cond_name',center_condition='rest')
data_mdtb_s2_all_fs = ut.recenter_fmri_data(data_mdtb_s2_all_fs,info_mdtb_2_all,task_column_name='cond_name',center_condition='rest')

# Get crossvalidated second moment matrix (GLib)
G_lib =  ct.get_G(data=data_mdtb_s2_run_fs,n_cond=29,n_part=16)

# contains information about length of conditions and their indices needed when constructing the battery
condition_df= ct.get_condition_indices(info_mdtb_2_run)


D = ev.real_connectivity(G_lib, condition_df,
                      data_mdtb_s2_run_fs,data_mdtb_s2_run_suit, # training data
                      data_mdtb_s1_all_fs,data_mdtb_s1_all_suit, # test data
                      battery_sizes = [3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                      metrics  = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc'],
                      n_batteries = 20000,
                      rest_idx = 28,
                      n_iter=40)

D_averaged_across_iter = averaged_df = plot.average_per_subject(D,'correlation') # lists of subjects are averaged across iterations
long_df = averaged_df.explode('avg_correlation_per_subject') # expands the list of subjects into rows for each subject
long_df['avg_correlation_per_subject'] = long_df['avg_correlation_per_subject'].astype(float) # turns each correaltion for each subject into a float 

# save
save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
save_path = os.path.join(save_dir, 'real_connectivity.tsv')
long_df.to_csv(save_path, sep='\t', index=False)