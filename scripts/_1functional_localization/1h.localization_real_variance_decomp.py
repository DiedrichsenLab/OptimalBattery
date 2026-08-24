import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import OptimalBattery.evaluate as ev
import Functional_Fusion.atlas_map as am
import Functional_Fusion.reliability as rel
import OptimalBattery.util as ut
import OptimalBattery.construct as ct
from scipy.optimize import brentq
from Functional_Fusion.dataset import DataSetLanguage
from scipy.stats import ttest_rel, sem
import nitools as nt
from OptimalBattery.global_config import save_dir, data_dir
import torch as pt


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

############## Variance decomposition (group / individual / noise) ##############
loc_conds = set(multi_combo) | set(single_contrast_names)
heldout_mask = ~info_run['task_name'].isin(loc_conds)
print(f"Held-out conditions ({info_run.loc[heldout_mask, 'task_name'].nunique()}): "
      f"{list(info_run.loc[heldout_mask, 'task_name'].unique())}")

data_run_np = data_run.detach().cpu().numpy()
data_dec = data_run_np[:, heldout_mask.values, :][:, :, ROI_indices]   # (n_sub, n_trials, n_fov)
cond_vec = pd.factorize(info_run.loc[heldout_mask, 'task_name'])[0]
part_vec = info_run.loc[heldout_mask, 'run'].values

var = rel.decompose_subj_group(data_dec, cond_vec, part_vec, separate='voxel_wise', subtract_mean=True)

comp_names = ['group', 'individual', 'noise']


def decomp_for_mask(mask):
    """Per-subject mean [v_g, v_s, v_e] and the same as % of total, averaged over the ROI voxels."""
    raw, frac = [], []
    for s in range(mask.shape[0]):
        sel = mask[s, ROI_indices] > 0
        if sel.sum() < 1:
            raw.append([np.nan] * 3); frac.append([np.nan] * 3); continue
        comp = np.nanmean(var[sel], axis=0)         
        raw.append(comp)
        frac.append(comp / comp.sum() * 100)         
    return np.array(raw), np.array(frac)


results = {name: decomp_for_mask(m) for name, m in masks.items()}

print("\nVariance components in the ROI (raw mean +/- sem  |  % of total):")
for name, (raw, frac) in results.items():
    print(f"  {name}:")
    for i, c in enumerate(comp_names):
        print(f"    {c:11s}: {np.nanmean(raw[:, i]):.4f} +/- {sem(raw[:, i], nan_policy='omit'):.4f}"
              f"   ({np.nanmean(frac[:, i]):.1f}%)")

print("\nPaired tests multi vs single (% of total):")
for single in ["contrast_fixed", "contrast_adaptive"]:
    for i, c in enumerate(comp_names):
        t, p = ttest_rel(results['multitask'][1][:, i], results[single][1][:, i], nan_policy='omit')
        print(f"  {c:11s}: multi vs {single}: t={t:.2f}, p={p:.2e}")

############## Bar plot (grouped: component x method), % of ROI variance ##############
palette = {"multitask": "#A34700", "contrast_fixed": "#005788", "contrast_adaptive": "#007656"}
methods = ["contrast_fixed", "contrast_adaptive", "multitask"]
x = np.arange(len(comp_names))
width = 0.25

fig, ax = plt.subplots(figsize=(6, 4))
for j, m in enumerate(methods):
    frac = results[m][1]
    means = np.nanmean(frac, axis=0)
    sems = sem(frac, axis=0, nan_policy='omit')
    ax.bar(x + (j - 1) * width, means, width, yerr=sems, capsize=3, color=palette[m], alpha=0.85, label=m)

ax.set_xticks(x)
ax.set_xticklabels(comp_names)
ax.set_ylabel("% of ROI variance")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{save_dir}/single_vs_multi/real_variance_decomp_barplot.pdf", bbox_inches="tight")
plt.show()
