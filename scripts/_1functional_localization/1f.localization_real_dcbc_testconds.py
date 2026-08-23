import numpy as np
import matplotlib.pyplot as plt
import OptimalBattery.evaluate as ev
import Functional_Fusion.atlas_map as am
import OptimalBattery.util as ut
import OptimalBattery.construct as ct
from scipy.optimize import brentq
from Functional_Fusion.dataset import DataSetLanguage
from scipy.stats import ttest_rel, ttest_1samp, sem
import nitools as nt
from OptimalBattery.global_config import save_dir, data_dir
import torch as pt

from DCBC.dcbc import compute_DCBC
from DCBC.utilities import compute_dist


# save figs?
save_plot = False

############## Load data ##############
space = 'SUIT3'
atlas, _ = am.get_atlas(atlas_str=space)
subj = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10',
        'sub-12', 'sub-13', 'sub-14', 'sub-15', 'sub-16', 'sub-17', 'sub-18', 'sub-19']

lang_dataset = DataSetLanguage(f'{data_dir}/FunctionalFusion_new/Language')

data_run, info_run = lang_dataset.get_data(space=space, ses_id='ses-localizer', type='CondRun', subj=subj)
data_run[np.isnan(data_run)] = 0

data_all, info_all = lang_dataset.get_data(space=space, ses_id='ses-localizer', type='CondAll', subj=subj)
data_all[np.isnan(data_all)] = 0

task_names_s1 = info_all['task_name'].unique()
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
coarse_parcelation = ut.combine_parcellation_regions(parcelation_32, labels, region_mapping)

# define field of view (right hemi of cerebellum)
ROI_to_include = np.arange(5, 10)
ROI_mask_np = np.isin(coarse_parcelation, ROI_to_include).astype(int)
ROI_indices = np.where(ROI_mask_np == 1)[0]

# torchify
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
data_all = pt.tensor(data_all, dtype=pt.float32, device=device)
data_run = pt.tensor(data_run, dtype=pt.float32, device=device)
parcelation = pt.tensor(coarse_parcelation, dtype=pt.float32, device=device)
ROI_mask = pt.tensor(ROI_mask_np, dtype=pt.float32, device=device)

# gets what the indices for each task are and the duration of each regressor
condition_df = ct.get_condition_indices(info_run, task_column_name='task_name', cond_column_name='task_name')

multi_combo = ['tongue_movement', 'theory_of_mind', 'demand_grid', 'sentence_reading', 'spatial_navigation']
multi_combination = info_all.index[info_all["task_name"].isin(multi_combo)].tolist()

############## Build the three localizer maps (same as 1d, matched avg size) ##############
# multitask localizer
comb_names, Uhats_multi_full, Uhats_multi_collapsed = ev.real_localization_multi(
    combination=multi_combination, task_names_s1=task_names_s1,
    condition_df=condition_df, ROI_mask=ROI_mask,
    data_train=data_run, data_vs=data_all, parcellation_vs=parcelation, parcel_interest_idx=4)

# average multitask roi size -> target size for the single-contrast thresholds
avg_size = Uhats_multi_collapsed.sum(axis=1).mean()

# single contrast (sentence_reading - nonword_reading)
single_contrast_names = ['sentence_reading', 'nonword_reading']
single_combo_indices = info_all.index[info_all["task_name"].isin(single_contrast_names)].tolist()
combination_regressors = ct.build_combination_regressors(single_combo_indices, condition_df=condition_df, localizer_time=8)
combination_regressors_sorted = [sorted(sublist) for sublist in combination_regressors]
sentence_data = data_run[:, combination_regressors_sorted[0], :].cpu().numpy()
nonword_data = data_run[:, combination_regressors_sorted[1], :].cpu().numpy()

# fixed (absolute t) threshold matched to avg multitask size
def f(th):
    pred_sizes = [ev.thresholded_t_contrast(sentence_data[i], nonword_data[i], threshold=th, mode='absolute')[0, :].sum().item()
                  for i in range(sentence_data.shape[0])]
    return np.mean(pred_sizes) - avg_size
best_th_fixed = brentq(f, 0.01, 50.0)
print(f"Best fixed threshold (matched to actual data): {best_th_fixed:.3f}")

# adaptive (percentile) threshold matched to avg multitask size
def f(th):
    pred_sizes = [ev.thresholded_t_contrast(sentence_data[i], nonword_data[i], threshold=th, mode='percentile')[0, :].sum().item()
                  for i in range(sentence_data.shape[0])]
    return np.mean(pred_sizes) - avg_size
best_th_adaptive = brentq(f, 1, 99)
print(f"Best adaptive threshold (matched to actual data): {best_th_adaptive:.3f}")

contrasts_fixed = pt.stack([ev.thresholded_t_contrast(sentence_data[i], nonword_data[i], threshold=best_th_fixed, mode='absolute')[0]
                            for i in range(sentence_data.shape[0])], axis=0) * ROI_mask
contrasts_adaptive = pt.stack([ev.thresholded_t_contrast(sentence_data[i], nonword_data[i], threshold=best_th_adaptive, mode='percentile')[0]
                               for i in range(sentence_data.shape[0])], axis=0) * ROI_mask

# collect binary masks per method (n_sub, n_vox)
masks = {
    "multitask": Uhats_multi_collapsed,
    "contrast_fixed": contrasts_fixed.cpu().numpy(),
    "contrast_adaptive": contrasts_adaptive.cpu().numpy(),
}

############## DCBC ##############
max_dist = 35
bin_width = 5

coords_fov = atlas.world.T[ROI_indices]                        # (N_fov, 3) right-hemi voxels
dist = compute_dist(coords_fov, backend='torch').to_sparse()

dcbc_conds = ['intact_passage', 'degraded_passage', 'n_back', 'rest']
heldout_idx = np.where(info_all['task_name'].isin(dcbc_conds).values)[0]
print(f"DCBC conditions ({len(heldout_idx)}): {list(info_all['task_name'].iloc[heldout_idx])}")
func_all = data_all.detach().cpu().numpy()[:, heldout_idx, :]


def dcbc_for_mask(mask):
    vals = []
    for s in range(mask.shape[0]):
        parc = mask[s, ROI_indices]
        func = func_all[s][:, ROI_indices].T
        D = compute_DCBC(maxDist=max_dist, binWidth=bin_width,
                         parcellation=parc, func=func,
                         dist=dist, weighting=True, backend='torch')
        vals.append(float(D['DCBC']))
    return vals


dcbc = {name: dcbc_for_mask(m) for name, m in masks.items()}

print("\nDCBC (mean +/- sem, t vs 0):")
for name, vals in dcbc.items():
    v = np.array(vals, dtype=float)
    t, p = ttest_1samp(v, 0.0, nan_policy='omit')
    print(f"  {name:18s}: {np.nanmean(v):.4f} +/- {sem(v, nan_policy='omit'):.4f}   t={t:.2f}, p={p:.2e}")

print("\nPaired tests (multi vs single):")
for single in ["contrast_fixed", "contrast_adaptive"]:
    t, p = ttest_rel(dcbc["multitask"], dcbc[single], nan_policy='omit')
    print(f"  multitask vs {single}: t={t:.2f}, p={p:.2e}")

# Bar plot
palette = {"multitask": "#A34700", "contrast_fixed": "#005788", "contrast_adaptive": "#007656"}

plot_labels = ["contrast_fixed", "contrast_adaptive", "multitask"]
means = [np.nanmean(dcbc[l]) for l in plot_labels]
sems = [sem(np.array(dcbc[l], dtype=float), nan_policy='omit') for l in plot_labels]
colors = [palette[l] for l in plot_labels]

fig, ax = plt.subplots(figsize=(5, 4))
x = np.arange(len(plot_labels))
ax.bar(x, means, yerr=sems, capsize=5, color=colors, alpha=0.85)
ax.axhline(0, color='k', lw=0.8)
ax.set_xticks(x)
ax.set_xticklabels(plot_labels, rotation=20)
ax.set_ylabel("DCBC")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{save_dir}/single_vs_multi/real_dcbc_barplot_testconds.pdf", bbox_inches="tight")
plt.show()
