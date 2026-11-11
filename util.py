"""
Module for utility functions
Author: Bassel Arafat
"""
import torch as pt
import os
import numpy as np
import pandas as pd

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

def recenter_data(data, info, center_full_code='rest_task', keep_center=True):
    """
    Recenter data by subtracting the center condition (e.g. rest) within each partition (half or run).
    Works with the All, Half and Run data types.

    Parameters:
        data (np.ndarray): data of shape (n_subjects, n_rows, n_voxels)
        info (pd.DataFrame): one row per data row, with columns 'task_code', 'cond_code' and optionally 'half' or 'run'
        center_full_code (str): center condition as 'taskcode_condcode' (default: 'rest_task')
        keep_center (bool): if True, keep the center condition (as zeros); if False, remove it

    Returns:
        data (np.ndarray): recentered data
        info (pd.DataFrame): matching info
    """
    center_task, center_cond = center_full_code.split('_', 1)
    is_center = ((info['task_code'] == center_task) & (info['cond_code'] == center_cond)).values
    if not is_center.any():
        raise ValueError(f"Center condition with task_code='{center_task}' and cond_code='{center_cond}' not found in data")

    # Rows are recentered within their half or run; without either, all rows form one partition
    partition_col = next((col for col in ['half', 'run'] if col in info.columns), None)
    partition = info[partition_col].values if partition_col else np.zeros(len(info))

    data = data.copy()
    for p in np.unique(partition):
        rows = partition == p
        center = np.where(rows & is_center)[0]
        if len(center) == 0:
            raise ValueError(f"No center condition '{center_full_code}' found for partition {p}")
        data[:, rows, :] -= data[:, center[:1], :]

    if keep_center:
        return data, info.copy()
    return data[:, ~is_center, :], info[~is_center].reset_index(drop=True)

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


def load_mdtb_task_order():
    """
    Load the order in which the tasks were presented in the MDTB scanning runs (task set A, subject s16),
    from eval_tsvs/sc1_s16.dat in this repo.

    Returns:
        task_order (pd.DataFrame): one row per task block, in presentation order, with columns ['run', 'task_name']
    """
    dat = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_tsvs', 'sc1_s16.dat'), sep=r'\s+')
    dat = dat[dat['runNum'] >= 51]  # runs 51-66 are the 16 scanning runs
    return pd.DataFrame({'run': dat['runNum'] - 50, 'task_name': dat['taskName']}).reset_index(drop=True)


def sort_by_presentation_order(data, info, task_order):
    """
    Sort data and info by run, and within each run by the order in which the tasks were presented.

    Parameters:
        data (np.ndarray): data of shape (n_subjects, n_rows, n_voxels)
        info (pd.DataFrame): info with columns ["run", "task_name", ...]
        task_order (pd.DataFrame): presentation order, from load_mdtb_task_order

    Returns:
        data (np.ndarray): sorted data
        info (pd.DataFrame): sorted info, with the position within the run in column "task_num_orig"
    """
    info = add_original_order(info, task_order)
    order = info.sort_values(by=["run", "task_num_orig"]).index.to_numpy()
    return data[:, order, :], info.loc[order].reset_index(drop=True)
