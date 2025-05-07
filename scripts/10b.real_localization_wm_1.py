import numpy as np
import matplotlib.pyplot as plt
import construct as ut
import os
import OptimalBattery.evaluate as ev
import Functional_Fusion.atlas_map as am
from IndividualParcellation.global_config import *
import OptimalBattery.estimate as es
import OptimalBattery.util as ut
import OptimalBattery.construct as ct
import OptimalBattery.plot as plot
import matplotlib.pyplot as plt
import numpy as np
from Functional_Fusion.dataset import DataSetMDTB
from matplotlib.colors import ListedColormap
from scipy.stats import ttest_rel
import nitools as nt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from scipy.stats import sem
from matplotlib.patches import Patch

# start a pdf to save the figures and numbers
pdf_name = 'localization_cerebellum_wm_1.pdf'
pdf_path = os.path.join('eval_tsvs', pdf_name)
pdf = PdfPages(pdf_path)
get_pdf = True


# define atlas and dirs
space = 'SUIT3'
atlas,_= am.get_atlas(atlas_str=space)
base_dir = 'Y:/data/'
if not os.path.exists(base_dir):
    base_dir = '/cifs/diedrichsen/data/'

func_fus_dir = os.path.join(base_dir, 'FunctionalFusion')
cerebellum_dir = os.path.join(base_dir, 'Cerebellum')

# define constants for this specific analysis (language localizer)

############## Load data ##############
dataset = DataSetMDTB(f'{func_fus_dir}/MDTB')

subj = ['sub-02','sub-03','sub-04']
subj = None

data_run,info_run  =dataset.get_data(space=space,ses_id='ses-s1',type='CondRun',subj=subj)
data_run[np.isnan(data_run)] = 0

data_all,info_all  =dataset.get_data(space=space,ses_id='ses-s1',type='CondAll',subj=subj)
data_all[np.isnan(data_all)] = 0

data_all_test , info_all_test = dataset.get_data(space=space,ses_id='ses-s2',type='CondAll',subj=subj)
data_all_test[np.isnan(data_all_test)] = 0

data_run = ut.recenter_fmri_data(data_run,info_run,task_column_name='cond_name',center_condition='rest')
data_all = ut.recenter_fmri_data(data_all,info_all,task_column_name='cond_name',center_condition='rest')
data_all_test = ut.recenter_fmri_data(data_all_test,info_all_test,task_column_name='cond_name',center_condition='rest')


test_data_nback_idx = [19]
test_data_nback = data_all_test[:,test_data_nback_idx[0],:]

test_data_language_idx = [4,5] # verbgen - wordreading
test_data_language = data_all_test[:,test_data_language_idx[0],:] - data_all[:,test_data_language_idx[1],:]


task_names_s1  = info_all['cond_name'].unique()
#####################################


# Parcellation (nettekoven)
atlas_dir = f'{func_fus_dir}/Atlases/tpl-SUIT'
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
    7: ['D1R', 'D2R', 'D3R', 'D4R'], # target region
    8: ['S1R', 'S2R','S3R', 'S4R', 'S5R']}

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
if get_pdf:
    course_fig = plot.plot_multi_flat(coarse_parcelation.reshape(1, -1),overlay_type='label',cmap=coarse_cmap,showfigure=False)
    plot.save_flatmap_to_pdf(course_fig, "Coarse Parcellation", pdf)

# define field of view
ROI_to_include = np.arange(5, 10) 
ROI_mask = np.isin(coarse_parcelation, ROI_to_include).astype(int)
ROI_indices = np.where(ROI_mask == 1)[0]
if get_pdf:
    field_of_view = plot.plot_multi_flat(ROI_mask.reshape(1, -1),overlay_type='label',cmap='tab10',showfigure=False)
    plot.save_flatmap_to_pdf(field_of_view, "Field of View", pdf)


# Get the G_library for task selection
n_conds = len(task_names_s1)
n_part = data_run.shape[1] // n_conds
G_Lib = ct.get_G(data=data_run[:,:,ROI_indices],n_cond=n_conds,n_part=n_part)
if get_pdf:
    plt.imshow(G_Lib)
    plt.xticks(np.arange(len(task_names_s1)),task_names_s1,rotation=90)
    plt.yticks(np.arange(len(task_names_s1)),task_names_s1)
    plt.title("G_Library")
    pdf.savefig()
    plt.close()

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
data_all = pt.tensor(data_all, dtype=pt.float32, device=device)
data_train = pt.tensor(data_run, dtype=pt.float32, device=device)
parcelation = pt.tensor(coarse_parcelation, dtype=pt.float32, device=device)
ROI_mask = pt.tensor(ROI_mask, dtype=pt.float32, device=device)

full_vs_train = es.estimate_Vs(data_all,parcellation=parcelation,ROI_mask= ROI_mask)
full_vs_train = ut.center_matrix(full_vs_train,axis=0)
full_vs_train = ut.normalize_matrix(full_vs_train,axis=0)
if get_pdf:
    plt.imshow(full_vs_train.cpu().numpy())
    plt.yticks(np.arange(len(task_names_s1)),task_names_s1)
    plt.title("Vs training")
    pdf.savefig()
    plt.close()


# gets what the indices for each task are and the duration of each regressor
condition_df= ct.get_condition_indices(info_run, task_column_name='task_name', cond_column_name='cond_name')


# build multitask localizer
comb_names, Uhats_multi_full, Uhats_multi_collapsed =ev.real_localization_multi(G_Lib,combination=[2,3,15,19,28],task_names_s1=task_names_s1,
                                                                                condition_df= condition_df, ROI_mask=ROI_mask,
                                                                                data_train=data_train,full_vs_train=full_vs_train,parcel_interest_idx=3)
print(comb_names)
# store how many voxels assigned as target for each sub
n_voxels_assgiend = np.sum(Uhats_multi_collapsed, axis=1)

# plot multi full
custom_colors = [
    (0.85, 0.85, 0.85),  # 0 - light gray (background or zero)
    (0.00, 0.45, 0.74),  # 1 - blue
    (0.47, 0.67, 0.19),  # 2 - green
    (0.93, 0.69, 0.13),  # 3 - orange
    (0.85, 0.33, 0.10),  # 4 - orange
]
custom_cmap = ListedColormap(custom_colors)
if get_pdf:
    fig = plot.plot_multi_flat(Uhats_multi_full, overlay_type='label', cmap=custom_cmap, colorbar=True, stats='mode', showfigure=False)
    plot.save_flatmap_to_pdf(fig, "Multi-task localizer full \n tasks: " + str(comb_names), pdf)


# plot multi collapsed
custom_colors = [
    (0.85, 0.85, 0.85),  # 0 - light gray (background or zero)
    (0.85, 0.33, 0.10),  # 1 - orange
]
binary_cmap = ListedColormap(custom_colors)
if get_pdf:
    fig = plot.plot_multi_flat(Uhats_multi_collapsed,overlay_type='label',cmap=binary_cmap,colorbar=True,stats='mode',showfigure=False)
    plot.save_flatmap_to_pdf(fig, "Multi-task localizer collapsed \n n_voxels assigned in each subject: " + str(n_voxels_assgiend), pdf)


# make single contrast and show unthresholded
single_contrast_regs = ev.find_single_contrast(full_vs_train,2,3)
contrast_names= task_names_s1[single_contrast_regs]
print(contrast_names)
combination_regressors = ct.build_combination_regressors(single_contrast_regs, condition_df=condition_df, localizer_time=8) # sentences vs non words
Ysubset = ct.average_regressors(data_train, combination_regressors).cpu().numpy()
contrast = Ysubset[:,0,:] - Ysubset[:,1,:] 

# plot unthresholded
if get_pdf:
    fig = plot.plot_multi_flat(contrast,overlay_type='func',cscale=[-0.2,0.2],cmap='inferno',colorbar=True,stats='nanmean',showfigure=False)
    plot.save_flatmap_to_pdf(fig, "Unthresholded contrast, tasks: " + str(contrast_names), pdf)

#  create binary localizer based on n_voxels in the multi task localizer
Uhats_single = ev.size_matched_contrast(contrast,Uhats_multi_collapsed,roi_indices=ROI_indices)
if get_pdf:
    fig = plot.plot_multi_flat(Uhats_single,overlay_type='label',cmap=binary_cmap,colorbar=True,stats='mode',showfigure=False)
    plot.save_flatmap_to_pdf(fig, "Single contrast localizer", pdf)


############### evaluate the localizers ###############
# evaluate selectivity and specificity
interaction_matrix = ev.calculate_interaction_matrix(Uhats_multi_collapsed, Uhats_single, test_data_nback,test_data_language)
scores = ev.compute_interaction_scores(interaction_matrix)
t_stat, p_val = ttest_rel(scores[:, 0], scores[:, 1])

print(f"Interaction Score Comparison:\n"
      f"  t = {t_stat:.3f}, p = {p_val:.4f}\n")

# evaluate cross subject correlation of maps
mean_dice_multi = ev.calculate_spatial_overlap(Uhats_multi_collapsed)
mean_dice_single = ev.calculate_spatial_overlap(Uhats_single)
print(f"Mean Dice Coefficient:\n"
      f"  Multitask Localizer: {mean_dice_multi:.3f}\n"
      f"  Single Localizer:    {mean_dice_single:.3f}\n")

# add to pdf
if get_pdf:
    fig, ax = plt.subplots(figsize=(8, 11))
    ax.axis('off')
    ax.text(0.05, 0.95,
            f"Evaluation Summary\n\n"
            f"Interaction Score Comparison:\n"
            f"  t = {t_stat:.3f}, p = {p_val:.4f}\n\n"
            f"Mean Dice Coefficient:\n"
            f"  Multitask Localizer: {mean_dice_multi:.3f}\n"
            f"  Single Localizer:    {mean_dice_single:.3f}\n",
            fontsize=12, va='top')
    pdf.savefig(fig)
    plt.close(fig)
######################################################

# Compute means and SEMs
means = np.nanmean(interaction_matrix, axis=0)  # [loc, contrast, in/out]
sems = sem(interaction_matrix, axis=0, nan_policy='omit')


# Convert to df
rows = []
for loc_i, loc_name in enumerate(['Multitask', 'Single-task']):
    for con_i, con_name in enumerate(['nback_pic>rest', 'verbgen>wordread']):
        for reg_i, reg_name in enumerate(['Inside', 'Outside']):
            rows.append({
                'Region': reg_name,
                'Localizer': loc_name,
                'Contrast': con_name,
                'Value': means[loc_i, con_i, reg_i],
                'SEM': sems[loc_i, con_i, reg_i]
            })
df = pd.DataFrame(rows)

# Define unique positions
region_pos = {'Inside': 0, 'Outside': 1}
bar_offsets = {
    ('Multitask', 'nback_pic>rest'): -0.3,
    ('Multitask', 'verbgen>wordread'): -0.1,
    ('Single-task', 'nback_pic>rest'): 0.1,
    ('Single-task', 'verbgen>wordread'): 0.3
}
colors = {'Multitask': 'darkorange', 'Single-task': 'steelblue'}
hatches = {'nback_pic>rest': '//', 'verbgen>wordread': ''}

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
for _, row in df.iterrows():
    base_x = region_pos[row['Region']]
    offset = bar_offsets[(row['Localizer'], row['Contrast'])]
    x = base_x + offset
    ax.bar(x, row['Value'], yerr=row['SEM'], width=0.18,
           color=colors[row['Localizer']],
           hatch=hatches[row['Contrast']],
           edgecolor='black', capsize=4)

# format
ax.set_xticks([0, 1])
ax.set_xticklabels(['Inside', 'Outside'], fontsize=12)
ax.set_ylabel('Mean Contrast Value')
ax.set_title('Contrast by Region, Localizer, and Type')

# Legend
legend = [
    Patch(facecolor='darkorange', edgecolor='black', label='Multitask'),
    Patch(facecolor='steelblue', edgecolor='black', label='Single-task'),
    Patch(facecolor='white', edgecolor='black', hatch='', label='Target'),
    Patch(facecolor='white', edgecolor='black', hatch='//', label='Control')
]
ax.legend(handles=legend)
plt.tight_layout()
if get_pdf:
    pdf.savefig(fig)
    plt.close(fig)



if get_pdf:
    pdf.close()
    plot.compress_pdf(pdf_path)
