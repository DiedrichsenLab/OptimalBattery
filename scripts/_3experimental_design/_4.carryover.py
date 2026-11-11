"""
Carryover effects in the MDTB data (Fig 4b):
is the activity pattern of a task more similar across runs when it was preceded by the same task?
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import ttest_rel, sem
from Functional_Fusion.dataset import DataSetMDTB
import OptimalBattery.util as ut
from OptimalBattery.global_config import data_dir, save_dir

fig_dir = f"{save_dir}/carryover_tempAutocorrelation"


def make_prev_cond(cond_vec, part_vec):
    """Condition that came before each condition within its run (-1 for the first task of a run)."""
    prev_vec = np.roll(cond_vec, 1)
    prev_vec[np.r_[True, part_vec[1:] != part_vec[:-1]]] = -1
    return prev_vec


def carryover_covariance(data, cond_vec, part_vec, include_start=True, level='group'):
    """
    Covariance between the patterns of the same task in different runs,
    split by whether the task was preceded by a different or by the same task.
        different previous task: task variance
        same previous task:      task variance + carryover variance

    Params:
        data (np.ndarray): (n_subjects, n_rows, n_voxels), rows sorted by run and presentation order
        cond_vec (np.ndarray): condition of each row
        part_vec (np.ndarray): run of each row
        include_start (bool): include the first task of each run (which all share the "previous task" -1)
        level (str): 'group' compares patterns between subjects, 'individual' within subject
    Returns:
        cov_diff_prev, cov_same_prev (np.ndarray): one value per subject
    """
    prev_vec = make_prev_cond(cond_vec, part_vec)
    if not include_start:
        keep = prev_vec != -1
        data, cond_vec, part_vec, prev_vec = data[:, keep, :], cond_vec[keep], part_vec[keep], prev_vec[keep]
    n_subj, n_rows, n_vox = data.shape

    def same(vec):  # which pairs of rows (of the stacked subjects) share the same value?
        vec = np.tile(vec, n_subj)
        return vec[:, None] == vec[None, :]

    subj_vec = np.repeat(np.arange(n_subj), n_rows)
    same_subj = subj_vec[:, None] == subj_vec[None, :]
    pairs = same(cond_vec) & ~same(part_vec) & (same_subj if level == 'individual' else ~same_subj)
    diff_prev = pairs & ~same(prev_vec)
    same_prev = pairs & same(prev_vec)

    # Inner products between all patterns of all subjects
    Y = data.reshape(n_subj * n_rows, n_vox)
    YY = Y @ Y.T

    v_task = np.nanmean(YY[diff_prev])
    v_carry = np.nanmean(YY[same_prev]) - v_task
    total = v_task + v_carry
    print(f"proportions: task(non-carry): {v_task / total:.3f}, carry: {v_carry / total:.3f}")

    cov_diff_prev = np.array([np.nanmean(YY[diff_prev & (subj_vec == s)]) for s in range(n_subj)])
    cov_same_prev = np.array([np.nanmean(YY[same_prev & (subj_vec == s)]) for s in range(n_subj)])
    return cov_diff_prev, cov_same_prev


# Transition matrix: how often did each task follow each other task?
task_order = ut.load_mdtb_task_order()
prev, curr = [], []
for _, run in task_order.groupby('run'):
    names = run['task_name'].tolist()
    prev += ['start'] + names[:-1]
    curr += names
tasks = sorted(task_order['task_name'].unique())
trans_counts = pd.crosstab(pd.Series(prev, name='prev'), pd.Series(curr, name='curr'))
trans_counts = trans_counts.reindex(index=['start'] + tasks, columns=tasks, fill_value=0)

im = plt.imshow(trans_counts, cmap="Blues", aspect="auto")
plt.xticks(range(len(trans_counts.columns)), trans_counts.columns, rotation=90)
plt.yticks(range(len(trans_counts.index)), trans_counts.index)
plt.colorbar(im, label="Transition count")
plt.savefig(f"{fig_dir}/transition_matrix.pdf", bbox_inches='tight')
plt.close()
# the max transition repeat is 3. And you get about 3/4 of the possible transitions.

# Load the activity patterns per task and run (cortex), sorted into the order of presentation
dataset = DataSetMDTB(f'{data_dir}/FunctionalFusion_new/MDTB')
data, info = dataset.get_data(space='fs32k', ses_id='ses-s1', type='TaskRun', subj=None)
data, info = ut.sort_by_presentation_order(np.nan_to_num(data), info, task_order)
cond_vec = info['cond_num'].values
part_vec = info['run'].values

# How often do two repetitions of a task share the same previous task? (excluding the first task of each run)
prev_vec = make_prev_cond(cond_vec, part_vec)
curr, prev = cond_vec[prev_vec != -1], prev_vec[prev_vec != -1]
same_task = curr[:, None] == curr[None, :]
same_prev = prev[:, None] == prev[None, :]
np.fill_diagonal(same_task, False)
print(f"{100 * np.mean(same_prev[same_task]):.1f}% of same-task pairs share the same previous task")

# Carryover between and within subjects, with and without the first task of each run
bars = []  # for each of the 4 analyses: covariance with different, then with same previous task
for include_start in [True, False]:
    print('Including start task:' if include_start else 'Excluding start task:')
    for level in ['group', 'individual']:
        print(f'{level} carryover')
        cov_diff, cov_same = carryover_covariance(data, cond_vec, part_vec, include_start, level)
        t, p = ttest_rel(cov_same, cov_diff)
        print(f"Carry-over difference t={t:.3f}, p={p:.3e}")

        v_carry, v_task = cov_same - cov_diff, cov_diff
        ratio = v_carry / (v_carry + v_task)
        print(f"carryover relative to carryover + pure task effect: mean={np.mean(ratio)}, SEM={sem(ratio)}")
        bars += [cov_diff, cov_same]

# Bar plot
color_diff, color_same = "#1f77b4", "#ff7f0e"
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar([0, 1, 3, 4, 7, 8, 10, 11],
       [np.mean(b) for b in bars],
       yerr=[sem(b) for b in bars],
       width=0.8, color=[color_diff, color_same] * 4, capsize=5, alpha=0.9)
ax.set_xticks([0.5, 3.5, 7.5, 10.5])
ax.set_xticklabels(["Group\n(start)", "Individual\n(start)", "Group\n(no start)", "Individual\n(no start)"])
ax.set_ylabel("Covariance")
ax.spines[['top', 'right']].set_visible(False)
ax.legend(handles=[Patch(facecolor=color_diff, label="Different previous"),
                   Patch(facecolor=color_same, label="Same previous")], frameon=False)
plt.tight_layout()
fig.savefig(f"{fig_dir}/carryover.pdf", bbox_inches="tight")
plt.show()
