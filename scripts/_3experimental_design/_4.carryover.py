import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import Functional_Fusion.atlas_map as am
from Functional_Fusion.dataset import DataSetMDTB
import os
import OptimalBattery.util as ut
from scipy.stats import ttest_1samp, ttest_rel
from OptimalBattery.global_config import data_dir, save_dir
from matplotlib.patches import Patch



# build transition matrix for visualization only...
dat_file = df = pd.read_csv(f"{data_dir}/Cerebellum/super_cerebellum/sc1/data/s16/sc1_s16.dat", delim_whitespace=True, header=0)
filtered_dat = dat_file[dat_file["runNum"] >= 51].copy()
filtered_dat["run"] = filtered_dat["runNum"] - 50
filtered_dat = filtered_dat.drop(columns=["runNum"])
filtered_dat = filtered_dat.rename(columns={"taskName": "task_name"})

# collect (prev_task, curr_task) pairs across runs
pairs = []
for _, g in filtered_dat.groupby("run"):
    names = g["task_name"].tolist()
    pairs.append(("start",names[0]))  # start of run
    pairs.extend(zip(names[:-1], names[1:]))          # immediate transitions only

length = len(pairs)

# Build counts matrix
trans_counts = pd.crosstab(
    pd.Series([a for a, b in pairs], name="prev"),
    pd.Series([b for a, b in pairs], name="curr"),
    dropna=False
).fillna(0).astype(int)
tasks = sorted(filtered_dat["task_name"].unique())
trans_counts = trans_counts.reindex(
    index=["start"] + tasks,
    columns=tasks,            
    fill_value=0
)

im = plt.imshow(trans_counts, cmap="Blues", aspect="auto")
plt.xticks(range(len(trans_counts.columns)), trans_counts.columns, rotation=90)
plt.yticks(range(len(trans_counts.index)), trans_counts.index)
plt.colorbar(im, label="Transition count")
plt.savefig(f"{save_dir}/carryover_tempAutocorrelation/transition_matrix.pdf", bbox_inches='tight')
plt.close()
#the max transition repeat is 3. And you get about 3/4 of the possible transitions.


# Variance decomp (carry over)
def make_prev_cond(cond_vec, part_vec):
    """
    Make a vector of previous global condition IDs (per run).
    First task in each run gets -1.
    """
    prev_vec = np.full_like(cond_vec, -1)  # fill with -1
    for t in range(1, len(cond_vec)):
        if part_vec[t] == part_vec[t-1]:       # same run → use previous cond
            prev_vec[t] = cond_vec[t-1]
        # else → leave as -1 (first task in run)
    return prev_vec


 
def decompose_carryover(data,info,cond_vec,part_vec, include_start=True, type = 'group', criterion="prev"):
    """
    Decompose data into task, systematic carryover, subject, noise (idosyncratic carryover + movement noise)
    Params:
    data : np.ndarray
        Array of shape (n_subjects, n_tasks, n_voxels)
    info : pd.DataFrame
        MDTB info dataframe with columns ["run", "task_name", "task_num", ...]
    cond_vec : np.ndarray
        Array of shape (n_tasks,) with global condition IDs (task_num)
    part_vec : np.ndarray
        Array of shape (n_tasks,) with partition/run IDs
    include_start : bool
        Whether to include first tasks in each run (default: True)
    within_individual : str
        Whether to compute group level carryover or idiosyncratic carryover (default: group) - "individual"
    criterion : str
        Whether to do over runs or previous tasks (default: "runs") - "prev"
    Returns
    -------
    """

    # make a vector of previous conditions (-1 for first task in run)
    prev_vec = make_prev_cond(cond_vec, part_vec)

    # filter out first tasks if not including start
    if not include_start:
        valid_mask = prev_vec != -1
        info = info.loc[valid_mask].reset_index(drop=True)
        cond_vec = cond_vec[valid_mask]
        part_vec = part_vec[valid_mask]
        prev_vec = prev_vec[valid_mask]
        data = data[:, valid_mask, :]

    n_subj, n_regs, n_vox = data.shape
    # make design vectors
    subj_vec = np.repeat(np.arange(n_subj), n_regs)
    curr_vec = np.tile(cond_vec, n_subj)
    prev_vec = np.tile(prev_vec, n_subj)
    run_vec = np.tile(part_vec, n_subj)

    # Compute inner product matrix
    Y = data.reshape(n_subj*n_regs, n_vox)
    YY = Y @ Y.T

    # Create masks for different comparison types
    same_subj   = subj_vec[:, None] == subj_vec[None, :]
    same_run    = run_vec[:, None] == run_vec[None, :]
    same_curr   = curr_vec[:, None] == curr_vec[None, :]
    same_prev   = prev_vec[:, None] == prev_vec[None, :]

    # Subject-specific carryover
    if type == 'individual':
        mask_within_nsameprev = same_subj & ~same_run & same_curr & ~same_prev # sigma (task) + sigma (subject)
        cov_within_nsameprev = np.nanmean(YY[mask_within_nsameprev])

        mask_within_sameprev = same_subj & ~same_run & same_curr & same_prev # sigma (task) + sigma (Carry) + sigma (subject) + sigma (idosyncratic carry)
        cov_within_sameprev = np.nanmean(YY[mask_within_sameprev])

        if criterion == "runs":
            # stop and print invalid cant do runs for within individual
            raise ValueError("Invalid criterion 'runs' for within_individual=True. Use 'prev' instead.")
    
        elif criterion == "prev":
            # Decompose variance components (using same / different previous tasks)
            v_t = cov_within_nsameprev # reliable task  variance 
            v_c = cov_within_sameprev - cov_within_nsameprev # carry-over variance

            mask_carry = mask_within_sameprev
            mask_noncarry = mask_within_nsameprev
        
    # Between subjects
    elif type == 'group':
        # Carry-over (cross-subject, same task, same/diff runs) (the sum of carry and non-carry is the group effect)
        mask_between_samerun    = ~same_subj & same_curr & same_run # sigma(task) + sigma (carry)
        cov_between_samerun= np.nanmean(YY[mask_between_samerun])

        mask_between_acrossruns = ~same_subj & same_curr & ~same_run # sigma (task)
        cov_between_acrossruns   = np.nanmean(YY[mask_between_acrossruns])


        # Carry-over (cross-subject, same task, same/diff runs) (the sum of carry and non-carry is the group effect)
        mask_between_sameprev    = ~same_subj & same_curr & ~same_run & same_prev # sigma(task) + sigma (carry)
        cov_between_sameprev= np.nanmean(YY[mask_between_sameprev])

        mask_between_nsameprev = ~same_subj & same_curr & ~same_run & ~same_prev # sigma (task)
        cov_between_nsameprev   = np.nanmean(YY[mask_between_nsameprev])


        if criterion == "runs":
            # Decompose variance components (using same / different runs)
            v_t = cov_between_acrossruns # reliable task  variance 
            v_c = cov_between_samerun - cov_between_acrossruns # carry-over variance

            mask_carry = mask_between_samerun
            mask_noncarry = mask_between_acrossruns
        
        elif criterion == "prev":
            # Decompose variance components (using same / different previous tasks)
            v_t = cov_between_nsameprev # reliable task  variance 
            v_c = cov_between_sameprev - cov_between_nsameprev # carry-over variance

            mask_carry = mask_between_sameprev
            mask_noncarry = mask_between_nsameprev

    # Express thew proportions as fraction of the total reliable variance (within subject)
    total = v_c + v_t
    print(f"proportions: task(non-carry): {v_t / total:.3f}, carry: {v_c / total:.3f}")

    #Subject-specific carry-over (within-subject, same task)
    v_t_per_subj = np.empty(n_subj)
    v_t_c_per_subj = np.empty(n_subj)
    

    for s in range(n_subj):
        mask_subj = (subj_vec == s)

        subj_mask_noncarry     = mask_noncarry & mask_subj
        subj_mask_carry      = mask_carry    & mask_subj
        
        v_t_per_subj[s] = np.nanmean(YY[subj_mask_noncarry])
        v_t_c_per_subj[s] = np.nanmean(YY[subj_mask_carry])
    

    return v_t_per_subj, v_t_c_per_subj

# define atlas and dirs
space = 'fs32k'
atlas,_= am.get_atlas(atlas_str=space)
func_fus_dir = os.path.join(data_dir, 'FunctionalFusion_new')

MDTB_dataset = DataSetMDTB(f'{func_fus_dir}/MDTB')
subj = None
data_mdtb_s1_run,info_mdtb_1_run  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='TaskRun',subj=subj)
# nans to 0
data_mdtb_s1_run = np.nan_to_num(data_mdtb_s1_run)

# add true order to info
info_with_order = ut.add_original_order(info_mdtb_1_run,filtered_dat)

# get the sort order based on (run, task_num_orig)
sort_order = info_with_order.sort_values(by=["run", "task_num_orig"]).index


# resort the info and data using the task_num_orig
info_sorted = info_with_order.loc[sort_order].reset_index(drop=True)
data_sorted = data_mdtb_s1_run[:, sort_order, :]


cond_vec = info_sorted["cond_num"].values   # global condition ID
part_vec  = info_sorted["run"].values

# percentage of same-previous-task pairs
prev_vec = make_prev_cond(cond_vec, part_vec)

# exclude first task in each run
valid = prev_vec != -1
curr = cond_vec[valid]
prev = prev_vec[valid]

# same-task pairs across runs
same_task = curr[:, None] == curr[None, :]
same_prev = prev[:, None] == prev[None, :]

# remove self-pairs
np.fill_diagonal(same_task, False)
np.fill_diagonal(same_prev, False)

# restrict to same-task comparisons
mask = same_task

percent_same_prev = 100 * np.mean(same_prev[mask])
print(f"{percent_same_prev:.1f}% of same-task pairs share the same previous task")


print('Including start task:')
# including start task for group
print('group carryover')
cov_dif_group, cov_same_group = decompose_carryover(
    data_sorted,
    info_sorted,
    cond_vec,
    part_vec, include_start=True, type= "group", criterion="prev"
)

t_group,p_group = ttest_rel(cov_same_group, cov_dif_group)
print(f"Carry-over difference group t={t_group:.3f}, p={p_group:.3e}")

v_c = cov_same_group - cov_dif_group
v_t = cov_dif_group
ratio = v_c / (v_c + v_t)
print("mean percentage of carryover relative to carryover + pure task effect:")
print(np.mean(ratio))
print("standard error of the mean (SEM) of the percentage:")
se = np.std(ratio, ddof=1) / np.sqrt(len(ratio))
print(se)


# including start task for individual
print('individual carryover')
cov_diff_indi,cov_same_indi = decompose_carryover(
    data_sorted,
    info_sorted,
    cond_vec,
    part_vec, include_start=True, type= "individual", criterion="prev"
)

t_indi,p_indi = ttest_rel(cov_same_indi, cov_diff_indi)
print(f"Carry-over difference individual t={t_indi:.3f}, p={p_indi:.3e}")

v_c = cov_same_indi - cov_diff_indi
v_t = cov_diff_indi
ratio = v_c / (v_c + v_t)
print("mean percentage of carryover relative to carryover + pure task effect:")
print(np.mean(ratio))
print("standard error of the mean (SEM) of the percentage:")
se = np.std(ratio, ddof=1) / np.sqrt(len(ratio))
print(se)


print('Excluding start task:')
# excluding start task for group
print('group carryover')
cov_dif_group_nos, cov_same_group_nos = decompose_carryover(
    data_sorted,
    info_sorted,
    cond_vec,
    part_vec, include_start=False, type= "group", criterion="prev"
)
t_group,p_group = ttest_rel(cov_same_group_nos, cov_dif_group_nos)
print(f"Carry-over difference t={t_group:.3f}, p={p_group:.3e}")

v_c = cov_same_group_nos - cov_dif_group_nos
v_t = cov_dif_group_nos
ratio = v_c / (v_c + v_t)
print("mean percentage of carryover relative to carryover + pure task effect:")
print(np.mean(ratio))
print("standard error of the mean (SEM) of the percentage:")
se = np.std(ratio, ddof=1) / np.sqrt(len(ratio))
print(se)

print('individual carryover')
cov_diff_indi_nos,cov_same_indi_nos = decompose_carryover(
    data_sorted,
    info_sorted,
    cond_vec,
    part_vec, include_start=False, type= "individual", criterion="prev"
)


t_indi,p_indi = ttest_rel(cov_same_indi_nos, cov_diff_indi_nos)
print(f"Carry-over difference t={t_indi:.3f}, p={p_indi:.3e}")

v_c = cov_same_indi_nos - cov_diff_indi_nos
v_t = cov_diff_indi_nos
ratio = v_c / (v_c + v_t)
print("mean percentage of carryover relative to carryover + pure task effect:")
print(np.mean(ratio))
print("standard error of the mean (SEM) of the percentage:")
se = np.std(ratio, ddof=1) / np.sqrt(len(ratio))
print(se)


# data
means = [
    np.mean(cov_dif_group),        np.mean(cov_same_group),        # Group (start)
    np.mean(cov_diff_indi),        np.mean(cov_same_indi),         # Individual (start)
    np.mean(cov_dif_group_nos),    np.mean(cov_same_group_nos),    # Group (no start)
    np.mean(cov_diff_indi_nos),    np.mean(cov_same_indi_nos)      # Individual (no start)
]

sems = [
    np.std(cov_dif_group, ddof=1) / np.sqrt(len(cov_dif_group)),
    np.std(cov_same_group, ddof=1) / np.sqrt(len(cov_same_group)),
    np.std(cov_diff_indi, ddof=1) / np.sqrt(len(cov_diff_indi)),
    np.std(cov_same_indi, ddof=1) / np.sqrt(len(cov_same_indi)),
    np.std(cov_dif_group_nos, ddof=1) / np.sqrt(len(cov_dif_group_nos)),
    np.std(cov_same_group_nos, ddof=1) / np.sqrt(len(cov_same_group_nos)),
    np.std(cov_diff_indi_nos, ddof=1) / np.sqrt(len(cov_diff_indi_nos)),
    np.std(cov_same_indi_nos, ddof=1) / np.sqrt(len(cov_same_indi_nos))
]

# x positions (extra spacing between blocks) 
x = np.array([0, 1,   3, 4,   7, 8,   10, 11])
width = 0.8

color_diff = "#1f77b4"   # different previous (light gray)
color_same = "#ff7f0e"   # same previous (dark)

colors = [
    color_diff, color_same,
    color_diff, color_same,
    color_diff, color_same,
    color_diff, color_same
]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    x, means, yerr=sems,
    width=width,
    color=colors,
    capsize=5,
    alpha=0.9
)


ax.set_xticks([0.5, 3.5, 7.5, 10.5])
ax.set_xticklabels([
    "Group\n(start)",
    "Individual\n(start)",
    "Group\n(no start)",
    "Individual\n(no start)"
])
ax.set_ylabel("Covariance")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# legend (only what matters)
legend_elements = [
    Patch(facecolor=color_diff, label="Different previous"),
    Patch(facecolor=color_same, label="Same previous")
]
ax.legend(handles=legend_elements, frameon=False)

plt.tight_layout()
fig.savefig(
    f"{save_dir}/carryover_tempAutocorrelation/carryover.pdf",
    bbox_inches="tight"
)
plt.show()
