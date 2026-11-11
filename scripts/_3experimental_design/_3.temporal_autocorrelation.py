"""
Temporal autocorrelation of the noise in the MDTB data (Fig 4a):
covariance between the residual activity patterns of two tasks in the same run,
as a function of how many task positions they were apart.
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
cov = np.nanmean(cov, axis=1)  # average across runs

# Mean covariance of all pairs of tasks that were `lag` positions apart, per subject
lags = np.arange(1, n_tasks)
cov_by_lag = np.zeros((n_sub, len(lags)))
for s in range(n_sub):
    for j, lag in enumerate(lags):
        pairs = np.diagonal(cov[s], offset=lag)  # covariances of all task pairs `lag` positions apart
        cov_by_lag[s, j] = np.nanmean(pairs)
mean = np.nanmean(cov_by_lag, axis=0)
sem = np.nanstd(cov_by_lag, axis=0) / np.sqrt(n_sub)

plt.figure(figsize=(6, 4))
plt.errorbar(lags, mean, yerr=sem, marker='o')
plt.xlabel("Task position difference (lag)")
plt.ylabel("Mean covariance")
plt.title("Covariance decay with task position (Cortex)")
plt.ylim(bottom=0)
plt.gca().spines[['top', 'right']].set_visible(False)
plt.savefig(f"{save_dir}/carryover_tempAutocorrelation/temp_auto.pdf", bbox_inches='tight')
plt.show()
