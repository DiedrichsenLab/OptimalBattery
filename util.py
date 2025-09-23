"""
Module for utility functions
Author: Bassel Arafat
"""
import torch as pt
import numpy as np

def center_matrix(X,axis =0):
    """Center the matrix by subtracting the mean.
    Args:
        X: matrix to center
        axis: Axis along which to center the data
    return:
        X: Centered matrix
    """
    if type(X) == np.ndarray:
        mean = np.nanmean(X, axis=axis, keepdims=True)
    else:
        mean = pt.nanmean(X, axis=axis, keepdims=True)
    X = X - mean
    return X

def normalize_matrix(X,axis = 0):
    """Normalize the matrix by dividing by the norm.
    Args:
        X: matrix to normalize
        axis: Axis along which to normalize the data
    return:
        X: Normalized matrix
    """
    if type(X) == np.ndarray:
        norm = np.sqrt(np.nansum(X**2, axis=axis, keepdims=True))
        norm = np.where(norm == 0, 1.0, norm)
    else:
        norm = pt.sqrt(pt.nansum(X**2, axis=axis, keepdims=True))
        norm = pt.where(norm == 0, 1.0, norm)
    X = X / norm
    return X

def recenter_fmri_data(data , info ,task_column_name = 'cond_name',center_condition='rest'): # tested and works but needs review..
    """
    Recenter fMRI data by subtracting the 'rest' condition from each run
    Parameters:
        data(np.ndarray): fMRI data of shape (n_subjects, n_conditions, n_voxels)
        info(pd.DataFrame): task information tsv
        task_column_name(str): column name in info that contains task names
        center_condition(str): name of the condition to center the data around
    Returns:
        processed_data(np.ndarray): recentered fMRI data
        updated_info(pd.DataFrame): updated task information tsv
    """

    n_subjects, n_conditions, n_voxels = data.shape
    if 'run' in info.columns:
        runs = info['run'].unique()
    else:
        runs = [1]

    # Find indices of 'centering' condition for the different runs
    center_condition_indices= info[info[task_column_name] == center_condition].index

    # initlize data
    processed_data = np.zeros((n_subjects, n_conditions, n_voxels))

    for subject_idx in range(n_subjects):
        subject_data = data[subject_idx] 
        subject_recentered_data = []
        
        # Subtract centering conition from each run
        for run in runs:
            if 'run' in info.columns:
                run_mask = info['run'] == run
            else:
                run_mask = np.ones(len(info), dtype=bool)
            run_data = subject_data[run_mask]
             
            # Subtract centering condition from the run
            centering_cond_idx_run = center_condition_indices[0]
            rest_data = run_data[centering_cond_idx_run]
            adjusted_run_data = run_data - rest_data

            # Append recentered run data and updated info
            subject_recentered_data.append(adjusted_run_data)

        # Combine processed runs for the subject
        processed_data[subject_idx] = np.vstack(subject_recentered_data)
    return processed_data

def combine_parcellation_regions(fine_parcellation, labels, region_mapping):
    """
    Combines parcellation labels into coarser ones based on a mapping and make it into hard parcellation.

    Parameters:
    - fine_parcellation (np.ndarray): shape (n_parcels, n_voxels), , must be probabilistic 
    - labels (list of str): names of the n_labels, should the same number as the first dimension of fine_parcellation
    - region_mapping (dict): keys are new region IDs (int), values are lists of label names (str)

    Returns:
    - hard_parcellation (np.ndarray): shape (n_voxels,), int labels
    """
    n_voxels = fine_parcellation.shape[1]
    n_groups = len(region_mapping)

    coarse_parcellation = np.zeros((n_groups, n_voxels))

    for group_id, group_labels in region_mapping.items():
        idx = [labels.index(lab) for lab in group_labels if lab in labels]
        coarse_parcellation[group_id - 1, :] = np.nansum(fine_parcellation[idx, :], axis=0)

    # some voxels are not assigned to any region, set them to 0
    is_unassigned = np.all(np.isnan(fine_parcellation), axis=0)
    hard_parcellation = np.argmax(coarse_parcellation, axis=0) + 1
    hard_parcellation[is_unassigned] = 0  # label 0 = background / unassigned
    hard_parcellation = hard_parcellation.astype(np.int32)

    return hard_parcellation


def add_original_order(info, original_dat):
    """
    add "task_num_orig" column to info dataframe, which contains the original task order
    
    Params:
    info : pd.DataFrame
        Functional_Fusion info dataframe with columns ["run", "task_name", ...]
    original_dat : pd.DataFrame
        Experimental timing dataframe with columns ["run", "task_name", ...]
    Returns
    -------
    info_reordered : pd.DataFrame
        Same info, with extra column "task_num_orig" for original order
    """
    
    info = info.copy()
    
    # merge on run + task_name so each row in info gets its "original" order index
    original_dat = original_dat.copy()
    original_dat["task_num_orig"] = original_dat.groupby("run").cumcount() + 1
    
    info = info.merge(
        original_dat[["run", "task_name", "task_num_orig"]], on=["run", "task_name"], how="left")
    
    return info

# def add_original_order(info, filtered_dat):
#     """
#     add "task_num_orig" column to info dataframe, which contains the original task order
    
#     Params:
#     data : np.ndarray
#         Array of shape (n_subjects, n_tasks, n_voxels)
#     info : pd.DataFrame
#         MDTB info dataframe with columns ["run", "task_name", "task_num", ...]
#     filtered_dat : pd.DataFrame
#         Experimental timing dataframe with columns ["runNum", "taskName", ...]
#     run_map : dict, optional
#         Mapping from runNum in filtered_dat to run in info, if they differ.
#         If None, assumes they are identical. (mdtb dat file has runNum starting from 51)
#     Returns
#     -------
#     data_reordered : np.ndarray
#         Same as input data, but rows reordered to match true task order
#     info_reordered : pd.DataFrame
#         Same as input info, with extra column "task_num_mod" for corrected order
#     """
    
#     # Default run mapping (identity if not provided)
#     if run_map is None:
#         run_map = {rn: rn for rn in filtered_dat['runNum'].unique()}
    
#     info = info.copy()
#     info['task_num_orig'] = -1  # placeholder
    
#     # Build dictionary: run -> ordered task list
#     true_orders = {}
#     for runNum, df_run in filtered_dat.groupby('runNum'):
#         run_id = run_map.get(runNum, runNum)
#         true_orders[run_id] = df_run['taskName'].tolist()
    
#     # Assign corrected task numbers
#     for run in info['run'].unique():

#         true_order = true_orders[run]
#         run_mask = info['run'] == run
        
#         for i, tname in enumerate(true_order, start=1):
#             mask = run_mask & (info['task_name'] == tname)
#             info.loc[mask, 'task_num_orig'] = i
    
#     # Sort indices by corrected task order within each run
#     sort_idx = []
#     for run in sorted(info['run'].unique()):
#         run_mask = info['run'] == run
#         order = info.loc[run_mask].sort_values('task_num_orig').index
#         sort_idx.extend(order)
    
#     # Apply reordering
#     data_reordered = data[:, sort_idx, :]
#     info_reordered = info.loc[sort_idx].reset_index(drop=True)
    
#     return data_reordered, info_reordered