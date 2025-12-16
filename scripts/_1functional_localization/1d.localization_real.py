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
import OptimalBattery.simulate as sim
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import numpy as np
from Functional_Fusion.dataset import DataSetLanguage
from matplotlib.colors import ListedColormap
from scipy.stats import ttest_rel,  ttest_1samp
import nitools as nt
from scipy.stats import sem  
import pandas as pd 
from OptimalBattery.global_config import save_dir, data_dir, repo_dir


# save figs?
save_plot = False

############## Load data ##############
space = 'SUIT3'
atlas,_= am.get_atlas(atlas_str=space)
subj = ['sub-02','sub-03','sub-04','sub-06','sub-07','sub-08','sub-09','sub-10','sub-12','sub-13','sub-14','sub-15','sub-16','sub-17','sub-18','sub-19']
# subj= ['sub-02','sub-04','sub-06']
lang_dataset = DataSetLanguage(f'{data_dir}/FunctionalFusion_new/Language')

data_run,info_run  =lang_dataset.get_data(space=space,ses_id='ses-localizer',type='CondRun',subj=subj)
data_run[np.isnan(data_run)] = 0

data_all,info_all  =lang_dataset.get_data(space=space,ses_id='ses-localizer',type='CondAll',subj=subj)
data_all[np.isnan(data_all)] = 0

test_target_tasks =['intact_passage','degraded_passage']
idx_intact = info_run.index[info_run["task_name"] == test_target_tasks[0]].tolist()
idx_degraded = info_run.index[info_run["task_name"] == test_target_tasks[1]].tolist()
test_target_intact  = data_run[:, idx_intact, :]
test_target_degraded = data_run[:, idx_degraded, :]
tvals_target, pvals_target = ttest_rel(test_target_intact, test_target_degraded, axis=1)

test_control_tasks =['n_back','rest']
idx_nback = info_run.index[info_run["task_name"] == test_control_tasks[0]].tolist()
idx_rest = info_run.index[info_run["task_name"] == test_control_tasks[1]].tolist()
test_nback  = data_run[:, idx_nback, :]
test_rest = data_run[:, idx_rest, :]
tvals_control, pvals_control = ttest_rel(test_nback, test_rest, axis=1)

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

# torchify
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
data_all = pt.tensor(data_all, dtype=pt.float32, device=device)
data_run = pt.tensor(data_run, dtype=pt.float32, device=device)
parcelation = pt.tensor(coarse_parcelation, dtype=pt.float32, device=device)
ROI_mask = pt.tensor(ROI_mask, dtype=pt.float32, device=device)

# gets what the indices for each task are and the duration of each regressor
condition_df= ct.get_condition_indices(info_run, task_column_name='task_name', cond_column_name='task_name')

multi_combo = ['tongue_movement','theory_of_mind','demand_grid','sentence_reading','spatial_navigation']
multi_combination = info_all.index[info_all["task_name"].isin(multi_combo)].tolist()



# build multitask localizer
comb_names, Uhats_multi_full, Uhats_multi_collapsed =ev.real_localization_multi(combination=multi_combination,task_names_s1=task_names_s1,
                                                                                condition_df= condition_df, ROI_mask=ROI_mask,
                                                                                data_train=data_run,data_vs=data_all,parcellation_vs= parcelation,parcel_interest_idx=4)

# compute average roi size in multi to find optimal fixed and adaptive thresholds
roi_sizes = Uhats_multi_collapsed.sum(axis=1)
avg_size = roi_sizes.mean()

if save_plot:
    group_percent_map = np.nanmean(Uhats_multi_collapsed, axis=0) * 100 * np.array(ROI_mask.cpu())
    group_percent_map_masked = np.where(np.array(ROI_mask.cpu()) == 1, group_percent_map, np.nan)
    fig = plot.plot_multi_flat(
            group_percent_map_masked.reshape(1, -1),
            overlay_type='func',
            cscale=[0, 40],
            cmap='inferno',
            colorbar=True,
            stats='nanmean',
            showfigure=False, single_fig= True
        )
    fig.savefig(f"{save_dir}/single_vs_multi/group_mutli_flatmap.pdf", bbox_inches='tight')
    plt.close(fig)


# make single contrast and show unthresholded
single_contrast_names = ['sentence_reading','nonword_reading']
single_combo_indices = info_all.index[info_all["task_name"].isin(single_contrast_names)].tolist()
combination_regressors = ct.build_combination_regressors(single_combo_indices, condition_df=condition_df, localizer_time=8) # sentences vs non words
combination_regressors_sorted = [sorted(sublist) for sublist in combination_regressors]
sentence_data = data_run[:, combination_regressors_sorted[0], :]
nonword_data = data_run[:, combination_regressors_sorted[1], :]
sentence_data = sentence_data.cpu().numpy()
nonword_data = nonword_data.cpu().numpy()

def f(th):
    pred_sizes = [ev.thresholded_t_contrast(sentence_data[i],nonword_data[i],threshold=th,mode='absolute')[0,:].sum().item()
                    for i in range(sentence_data.shape[0])]
    return np.mean(pred_sizes) - avg_size
best_th_fixed = brentq(f, 0.01, 50.0)
print(f"Best fixed threshold (matched to actual data): {best_th_fixed:.3f}")

def f(th):
    pred_sizes = [ev.thresholded_t_contrast(sentence_data[i],nonword_data[i],threshold=th,mode='percentile')[0,:].sum().item()
                    for i in range(sentence_data.shape[0])]
    return np.mean(pred_sizes) - avg_size
best_th_adaptive = brentq(f, 1, 99)
print(f"Best adaptive threshold (matched to actual data): {best_th_adaptive:.3f}")

contrasts_fixed = [ev.thresholded_t_contrast(sentence_data[i],nonword_data[i],threshold=best_th_fixed,mode='absolute')[0] for i in range(sentence_data.shape[0])]
contrasts_fixed = pt.stack(contrasts_fixed,axis=0)
contrasts_fixed = contrasts_fixed * ROI_mask
contrasts_adaptive = [ev.thresholded_t_contrast(sentence_data[i],nonword_data[i],threshold=best_th_adaptive,mode='percentile')[0] for i in range(sentence_data.shape[0])]
contrasts_adaptive = pt.stack(contrasts_adaptive,axis=0)
contrasts_adaptive = contrasts_adaptive * ROI_mask

if save_plot:
    group_percent_map = np.nanmean(contrasts_fixed.cpu(), axis=0) * 100 * np.array(ROI_mask.cpu())
    group_percent_map_masked = np.where(np.array(ROI_mask.cpu()) == 1, group_percent_map, np.nan)
    fig = plot.plot_multi_flat(
            group_percent_map_masked.reshape(1, -1),
            overlay_type='func',
            cscale=[0, 40],
            cmap='inferno',
            colorbar=True,
            stats='nanmean',
            showfigure=False, single_fig= True
        )
    fig.savefig(f"{save_dir}/single_vs_multi/group_contrat_fixed.pdf", bbox_inches='tight')
    plt.close(fig)

    # save adaptive
    group_percent_map = np.nanmean(contrasts_adaptive.cpu(), axis=0) * 100 * np.array(ROI_mask.cpu())
    group_percent_map_masked = np.where(np.array(ROI_mask.cpu()) == 1, group_percent_map, np.nan)
    fig = plot.plot_multi_flat(
            group_percent_map_masked.reshape(1, -1),
            overlay_type='func',
            cscale=[0, 40],
            cmap='inferno',
            colorbar=True,
            stats='nanmean',
            showfigure=False, single_fig= True
        )
    fig.savefig(f"{save_dir}/single_vs_multi/group_contrat_adaptive.pdf", bbox_inches='tight')
    plt.close(fig)

############### evaluate the localizers ###############
# evaluate cross subject correlation of maps (dice for each localization method)
print("Dice scores:")
print ('multi')
dice_multi_list = ev.calculate_crosssub_overlap(Uhats_multi_collapsed)
print(np.mean(dice_multi_list))
print( np.std(dice_multi_list)/np.sqrt(len(dice_multi_list)))
t,p = ttest_1samp(dice_multi_list, 0.0)
print(f"t={t:.3f}, p={p:.3f}")

print ('fixed')
dice_fixed_list = ev.calculate_crosssub_overlap(contrasts_fixed)
print(np.mean(dice_fixed_list))
print(np.std(dice_fixed_list)/np.sqrt(len(dice_fixed_list)))
t,p = ttest_1samp(dice_fixed_list, 0.0)
print(f"t={t:.3f}, p={p:.3f}")

print ('adaptive')
dice_adaptive_list = ev.calculate_crosssub_overlap(contrasts_adaptive)
print(np.mean(dice_adaptive_list))
print(np.std(dice_adaptive_list)/np.sqrt(len(dice_adaptive_list)))
t,p = ttest_1samp(dice_adaptive_list, 0.0)
print(f"t={t:.3f}, p={p:.3f}")

#####################################################

# Compute mean value of each test contrast in and out of roi defined by each localizer
multi_mask = Uhats_multi_collapsed
fixed_mask = contrasts_fixed.cpu().numpy()
adaptive_mask = contrasts_adaptive.cpu().numpy()

contrast_lang = tvals_target
contrast_nback = tvals_control


localizers = {
    "multitask": multi_mask,
    "contrast_fixed": fixed_mask,
    "contrast_adaptive": adaptive_mask
}

contrasts = {
    "intact>degraded": contrast_lang,
    "nback>rest": contrast_nback
}

def compute_in_out(mask, contrast):
    in_vals = contrast[mask == 1]
    out_vals = contrast[mask == 0]
    return np.nanmean(in_vals), np.nanmean(out_vals)


rows = []
n_subs = multi_mask.shape[0]
for loc_name, loc_mask in localizers.items():
    for con_name, con_data in contrasts.items():
        for s in range(n_subs):
            inside, outside = compute_in_out(loc_mask[s], con_data[s])
            rows.append({
                "subject": subj[s],
                "localizer": loc_name,
                "contrast": con_name,
                "inside": inside,
                "outside": outside
            })

# Make dataframe
df_eval = pd.DataFrame(rows)
out_path = f"{repo_dir}/eval_tsvs/localization_real_contrasts.tsv"
df_eval.to_csv(out_path, sep="\t", index=False)
