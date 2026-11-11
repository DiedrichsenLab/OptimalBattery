"""
Temporal autocorrelation of the noise in the MDTB data, expressed as the variance of a contrast:
how noisy would the contrast between two tasks be if they were always presented `lag` positions apart?
"""
import numpy as np
import matplotlib.pyplot as plt
from Functional_Fusion.dataset import DataSetMDTB
import OptimalBattery.util as ut
from OptimalBattery.global_config import data_dir, save_dir

# Load the activity patterns per task and run (cortex), relative to rest
dataset = DataSetMDTB(f'{data_dir}/FunctionalFusion_new/MDTB')
data, info = dataset.get_data(space='fs32k', ses_id='ses-s1', type='TaskRun', subj=None)
data, info = ut.recenter_data(data, info, center_full_code='rest_task', keep_center=True)
data = np.nan_to_num(data)

# Remove the mean of each task across runs, leaving the run-specific residuals
for cond in np.unique(info['cond_num']):
    rows = (info['cond_num'] == cond).values
    data[:, rows, :] -= data[:, rows, :].mean(axis=1, keepdims=True)

# Sort the tasks of each run into the order in which they were presented
data, info = ut.sort_by_presentation_order(data, info, ut.load_mdtb_task_order())
runs = np.unique(info['run'])
n_sub = data.shape[0]
n_tasks = len(np.unique(info['task_num_orig']))

# Covariance (across vertices) between the tasks of a run: subjects x runs x position x position
cov = np.zeros((n_sub, len(runs), n_tasks, n_tasks))
for s in range(n_sub):
    for r, run in enumerate(runs):
        cov[s, r] = np.cov(data[s, info['run'].values == run, :])
cov[cov == 0] = np.nan         # rest is all zeros after recentering: ignore it

# Mean variance of the contrast between all pairs of tasks that were `lag` positions apart, per subject.
lags = np.arange(1, n_tasks)
var_contrast_by_lag = np.zeros((n_sub, len(lags)))
for s in range(n_sub):
    var_contrast_runs = []
    for r in range(len(runs)):
        var = np.diagonal(cov[s, r])                    # variance of each position in this run
        var_contrast_runs.append([])
        for lag in lags:
            var_a = var[:-lag]                          # first task of each pair
            var_b = var[lag:]                           # second task, `lag` positions later
            cov_ab = np.diagonal(cov[s, r], offset=lag) # covariance of each pair
            var_contrast_runs[r].append(np.nanmean(var_a + var_b - 2 * cov_ab))
    var_contrast_by_lag[s] = np.nanmean(var_contrast_runs, axis=0)  # average across runs
mean = np.nanmean(var_contrast_by_lag, axis=0)
sem = np.nanstd(var_contrast_by_lag, axis=0) / np.sqrt(n_sub)

plt.figure(figsize=(6, 4))
plt.errorbar(lags, mean, yerr=sem, marker='o')
plt.xlabel("Task position difference (lag)")
plt.ylabel("Mean variance of contrast")
plt.title("Contrast variance by task distance (Cortex)")
plt.ylim(bottom=0)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.savefig(f"{save_dir}/carryover_tempAutocorrelation/temp_auto_contrast.pdf", bbox_inches='tight')
plt.show()
