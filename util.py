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

def recenter_data(data, info, center_full_code='rest_task', keep_center=False):
    """
    Recenter  data by subtracting the center condition from each partition.
    Works with CondAll, CondHalf, and CondRun data types.
    
    Parameters:
        data (np.ndarray):  data of shape (n_subjects, n_conditions x repititions, n_voxels)
        info (pd.DataFrame): condition information with columns like 'task_code', 'cond_code', 'half', 'run'
        center_full_code (str): full task condition code to center around as 'taskcode_condcode' (default: 'rest_task')
        keep_center (bool): if True, keep center condition as zeros; if False, remove it
    
    Returns:
        processed_data (np.ndarray): recentered fMRI data
        updated_info (pd.DataFrame): updated info
    """
    n_subjects, _, _ = data.shape
    
    # Split on first underscore only
    center_task, center_cond = center_full_code.split('_', 1)
    
    # Determine partition column
    if 'half' in info.columns:
        partition_col = 'half'
        partitions = info['half'].unique()
    elif 'run' in info.columns:
        partition_col = 'run'
        partitions = info['run'].unique()
    else:
        partition_col = None
        partitions = [None]
    
    # Find center condition indices (match both task_code and cond_code)
    center_mask = (info['task_code'] == center_task) & (info['cond_code'] == center_cond)
    
    if not center_mask.any():
        raise ValueError(f"Center condition with task_code='{center_task}' and cond_code='{center_cond}' not found in data")
    
    # Process each subject
    processed_data_list = []
    for subject_idx in range(n_subjects):
        subject_data = data[subject_idx]
        subject_recentered = []
        
        for partition in partitions:
            # Get partition mask
            if partition_col is not None:
                partition_mask = (info[partition_col] == partition)
            else:
                partition_mask = np.ones(len(info), dtype=bool)
            
            # Find center condition for this partition
            center_idx_partition = np.where(partition_mask & center_mask)[0]
            
            if len(center_idx_partition) == 0:
                raise ValueError(f"No center condition '{center_full_code}' found for partition {partition}")
            
            # Get center condition data
            center_data = subject_data[center_idx_partition[0]]
            
            if keep_center:
                # Keep all conditions including center (as zeros)
                partition_indices = np.where(partition_mask)[0]
                recentered = subject_data[partition_indices] - center_data
            else:
                # Remove center condition
                task_mask = partition_mask & ~center_mask
                partition_indices = np.where(task_mask)[0]
                recentered = subject_data[partition_indices] - center_data
            
            subject_recentered.append(recentered)
        
        processed_data_list.append(np.vstack(subject_recentered))
    
    processed_data = np.stack(processed_data_list, axis=0)
    
    # Update info
    if keep_center:
        updated_info = info.copy()
    else:
        updated_info = info[~center_mask].reset_index(drop=True)
    
    return processed_data, updated_info

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
