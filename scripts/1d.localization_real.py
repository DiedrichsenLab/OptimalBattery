# script for the second part of figure 1. Single contrast vs multi-task localization in real data using the language dataset.

import numpy as np
import matplotlib.pyplot as plt
import construct as ut
import OptimalBattery.evaluate as ev
import Functional_Fusion.atlas_map as am
from IndividualParcellation.global_config import *
import OptimalBattery.estimate as es
import OptimalBattery.util as ut
import OptimalBattery.construct as ct
import OptimalBattery.plot as plot
import matplotlib.pyplot as plt
import numpy as np
from Functional_Fusion.dataset import DataSetLanguage
from matplotlib.colors import ListedColormap
from scipy.stats import ttest_rel
import nitools as nt
from scipy.stats import sem  
import pandas as pd 
from OptimalBattery.global_config import save_dir, data_dir, repo_dir


# save figs?
save_plot = False

############## Load data ##############
space = 'SUIT3'
atlas,_= am.get_atlas(atlas_str=space)
subj = ['sub-02','sub-04','sub-06','sub-07','sub-08','sub-09','sub-10','sub-12','sub-13','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19']
# subj= ['sub-02','sub-03']
lang_dataset = DataSetLanguage(f'{data_dir}/FunctionalFusion_new/Language')

data_run,info_run  =lang_dataset.get_data(space=space,ses_id='ses-localizer',type='CondRun',subj=subj)
data_run[np.isnan(data_run)] = 0

data_all,info_all  =lang_dataset.get_data(space=space,ses_id='ses-localizer',type='CondAll',subj=subj)
data_all[np.isnan(data_all)] = 0


data_run = ut.recenter_fmri_data(data_run,info_run,task_column_name='task_name',center_condition='rest')
data_all = ut.recenter_fmri_data(data_all,info_all,task_column_name='task_name',center_condition='rest')

test_data_language_idx = [5,6]
test_data_language = data_all[:,test_data_language_idx[0],:] - data_all[:,test_data_language_idx[1],:]

test_data_nback_idx = [11]
test_data_nback = data_all[:,test_data_nback_idx[0],:]


task_names_s1  = info_all['task_name'].unique()
#####################################


# Parcellation (nettekoven)
atlas_dir = f'{data_dir}/FunctionalFusion_new/Atlases/tpl-SUIT'
model_type = 'atl-NettekovenSym32'
model_name = f'{atlas_dir}/{model_type}_space-SUIT_probseg.nii'
parcelation_32 = atlas.read_data(model_name)
labels = nt.read_lut(f'{atlas_dir}/{model_type}.lut')[2][1:]


# make coarse parcelation
region_mapping = {
    1: ['M1L', 'M2L', 'M3L', 'M4L'],
    2: ['A1L', 'A2L', 'A3L'],
    3: ['D1L', 'D2L', 'D3L', 'D4L'],
    4: ['S1L', 'S2L', 'S3L', 'S4L', 'S5L'],
    5: ['M1R', 'M2R', 'M3R', 'M4R'],
    6: ['A1R', 'A2R', 'A3R'],
    7: ['D1R', 'D2R', 'D3R', 'D4R'],
    8: ['S1R', 'S2R'],
    9: ['S3R', 'S4R', 'S5R']
}

# make hard coarse parcellation
coarse_parcelation = ut.combine_parcellation_regions(parcelation_32, labels, region_mapping)
custom_colors = [
    (0.85, 0.85, 0.85),  # 0 - light gray (background)
    (0.60, 0.60, 0.80),  # 1 - muted lavender
    (0.30, 0.70, 0.85),  # 2 - teal
    (0.75, 0.50, 0.95),  # 3 - violet
    (0.90, 0.60, 0.60),  # 4 - light coral
    (0.00, 0.45, 0.74),  # 5 - blue
    (0.47, 0.67, 0.19),  # 6 - green
    (0.93, 0.69, 0.13),  # 7 - orange
    (0.85, 0.33, 0.10),  # 8 - red-orange
    (0.49, 0.18, 0.56),  # 9 - purple
]
coarse_cmap = ListedColormap(custom_colors, name="custom_coarse")


# define field of view (right hemi of cerebellum)
ROI_to_include = np.arange(5, 10) 
ROI_mask = np.isin(coarse_parcelation, ROI_to_include).astype(int)
ROI_indices = np.where(ROI_mask == 1)[0]



# Get the G_library for task selection (in this case not used but could be if needed)
n_conds = len(task_names_s1)
n_part = data_run.shape[1] // n_conds
G_Lib = ct.get_G(data=data_run[:,:,ROI_indices],n_cond=n_conds,n_part=n_part)


# torchify
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
data_all = pt.tensor(data_all, dtype=pt.float32, device=device)
data_train = pt.tensor(data_run, dtype=pt.float32, device=device)
parcelation = pt.tensor(coarse_parcelation, dtype=pt.float32, device=device)
ROI_mask = pt.tensor(ROI_mask, dtype=pt.float32, device=device)

# estimate vs for training
full_vs_train = es.estimate_Vs(data_all,parcellation=parcelation,ROI_mask= ROI_mask)
full_vs_train = ut.center_matrix(full_vs_train,axis=0)
full_vs_train = ut.normalize_matrix(full_vs_train,axis=0)


# gets what the indices for each task are and the duration of each regressor
condition_df= ct.get_condition_indices(info_run, task_column_name='task_name', cond_column_name='task_name')

multi_combo = ['tongue_movement','theory_of_mind','demand_grid','sentence_reading']
multi_combination = info_all.index[info_all["task_name"].isin(multi_combo)].tolist()



# build multitask localizer
comb_names, Uhats_multi_full, Uhats_multi_collapsed =ev.real_localization_multi(G_Lib,combination=multi_combination,task_names_s1=task_names_s1,
                                                                                condition_df= condition_df, ROI_mask=ROI_mask,
                                                                                data_train=data_train,full_vs_train=full_vs_train,parcel_interest_idx=4)

if save_plot:
    group_percent_map = np.nanmean(Uhats_multi_collapsed, axis=0) * 100 * np.array(ROI_mask.cpu())
    group_percent_map_masked = np.where(np.array(ROI_mask.cpu()) == 1, group_percent_map, np.nan)
    fig = plot.plot_multi_flat(
            group_percent_map_masked.reshape(1, -1),
            overlay_type='func',
            cscale=[0, 50],
            cmap='inferno',
            colorbar=True,
            stats='nanmean',
            showfigure=False, single_fig= True
        )
    fig.savefig(f"{save_dir}/single_vs_multi/group_mutli_flatmap.pdf", bbox_inches='tight')
    plt.close(fig)

# store how many voxels assigned as target for each sub to use later on for single contrast thresholding
n_voxels_assgiend = np.sum(Uhats_multi_collapsed, axis=1)

# make single contrast and show unthresholded
single_contrast_regs = [16,17]
contrast_names= task_names_s1[single_contrast_regs]
combination_regressors = ct.build_combination_regressors(single_contrast_regs, condition_df=condition_df, localizer_time=8) # sentences vs non words
Ysubset = ct.average_regressors(data_train, combination_regressors).cpu().numpy()
contrast = Ysubset[:,0,:] - Ysubset[:,1,:] 

#  create binary localizer based on n_voxels in the multi task localizer
Uhats_single = ev.size_matched_contrast(contrast,Uhats_multi_collapsed,roi_indices=ROI_indices)
if save_plot:
    group_percent_map = np.nanmean(Uhats_single, axis=0) * 100 * np.array(ROI_mask.cpu())
    group_percent_map_masked = np.where(np.array(ROI_mask.cpu()) == 1, group_percent_map, np.nan)
    fig = plot.plot_multi_flat(
            group_percent_map_masked.reshape(1, -1),
            overlay_type='func',
            cscale=[0, 50],
            cmap='inferno',
            colorbar=True,
            stats='nanmean',
            showfigure=False, single_fig= True
        )
    fig.savefig(f"{save_dir}/single_vs_multi/group_singlecon_flatmap.pdf", bbox_inches='tight')
    plt.close(fig)

############### evaluate the localizers ###############
# evaluate selectivity and specificity
interaction_matrix = ev.calculate_interaction_matrix(Uhats_multi_collapsed, Uhats_single, test_data_language, test_data_nback)
scores = ev.compute_interaction_scores(interaction_matrix)
t_stat, p_val = ttest_rel(scores[:, 0], scores[:, 1])

target_inside_single = interaction_matrix[:, 1, 0, 0]  # [localizer=1 (single), contrast=0 (target), inside=0]
target_inside_multi  = interaction_matrix[:, 0, 0, 0]  # [localizer=0 (multi), contrast=0 (target), inside=0]

# Paired t-test
t_stat, p_val = ttest_rel(target_inside_multi, target_inside_single)

print(f"t = {t_stat:.3f}, p = {p_val:.4f}")


# evaluate cross subject correlation of maps
mean_dice_multi = ev.calculate_spatial_overlap(Uhats_multi_collapsed)
print(mean_dice_multi)
mean_dice_single = ev.calculate_spatial_overlap(Uhats_single)
print(mean_dice_single)


#####################################################

# Compute means and SEMs
means = np.nanmean(interaction_matrix, axis=0)  # [loc, contrast, in/out]
sems = sem(interaction_matrix, axis=0, nan_policy='omit')


# Convert to df
rows = []
for loc_i, loc_name in enumerate(['Multitask', 'Single-task']):
    for con_i, con_name in enumerate(['Intact>Degraded', 'n-back>Rest']):
        for reg_i, reg_name in enumerate(['Inside', 'Outside']):
            rows.append({
                'Region': reg_name,
                'Localizer': loc_name,
                'Contrast': con_name,
                'Value': means[loc_i, con_i, reg_i],
                'SEM': sems[loc_i, con_i, reg_i]
            })
df = pd.DataFrame(rows)
df.to_csv(f"{repo_dir}/eval_tsvs/localization_real_contrasts.tsv", sep="\t", index=False)

