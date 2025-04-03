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