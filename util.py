"""
Module for optimal battery construction
Author: Bassel Arafat
"""

import numpy as np
import pandas as pd
from numpy.linalg import eigh
import torch as pt


def eigenval_crit(G, center=True):
    """Computes various criteria based on the eigenvalues and mutual information of a matrix G.
    Assumes that G is symmetric."""

    N = G.shape[0]
    # Center the G matrix
    if center: 
        H = np.eye(N) - np.ones((N, N)) / N
        G_mc = H @ G @ H
    else:
        G_mc = G

    # Compute eigenvalues for both centered and uncentered G matrices
    l, _ = eigh(G)
    l = l[::-1]  # Reverse order
    l[l < 1e-12] = 1e-12 # Set small eigenvalues to a threshold

    l_mc, _ = eigh(G_mc)
    l_mc = l_mc[::-1]
    l_mc[l_mc < 1e-12] = 1e-12
        

    # Create a dictionary of criteria
    d = {
        'variance': np.sum(l),  # Sum of uncentered eigenvalues
        'variance_mc': np.sum(l_mc),  # Sum of mean-centered eigenvalues
        'inverse_trace': - np.sum(1 / l),
        'inverse_trace_mc': - np.sum(1 / l_mc),
        'log_det': np.sum(np.log(l)),
        'log_det_mc': np.sum(np.log(l_mc)),
        'num_eigenvalues': len(l_mc)
    }
    
    return d


def build_combinations(G_lib, strategy='random',n_iter=1000,n_tasks=4,seed=1,replacement=True): 
    """ Builds a set of task-batteries and evalates them 
    G_lib: second moment matrices of task-library
    strategy: 'random' or 'exhaustive'
    n_iter: number of iterations for random strategy
    """
    np.random.seed(seed)
    D_list = []
    n_lib_task = G_lib.shape[0]

    comb = []
    if strategy == 'random':
        for _ in range(n_iter):
            candidate = tuple(sorted(np.random.choice(n_lib_task, size=n_tasks, replace=replacement)))
            comb.append(candidate)
        comb = list(set(comb))   
    else:
        raise ValueError('Invalid strategy')
        
    for i in range(len(comb)):
        n_unique = len(set(comb[i]))
        d = eigenval_crit(G_lib[comb[i],:][:,comb[i]],center=True)
        d['combination'] = [comb[i]]
        d['n_unique'] = [n_unique]
        D_list.append(pd.DataFrame(d))
    D = pd.concat(D_list)
    return D 

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
        updated_info = []
        
        # Subtract centering conition from each run
        for run in runs:
            if 'run' in info.columns:
                run_mask = info['run'] == run
            else:
                run_mask = np.ones(len(info), dtype=bool)
            run_data = subject_data[run_mask]
            run_info = info[run_mask] 
             
            # Subtract centering condition from the run
            centering_cond_idx_run = center_condition_indices[0]
            rest_data = run_data[centering_cond_idx_run]
            adjusted_run_data = run_data - rest_data

            # Remove 'rest' condition from the run's metadata
            adjusted_run_info = run_info.drop(index=run_info.index[centering_cond_idx_run])

            # Append recentered run data and updated info
            subject_recentered_data.append(adjusted_run_data)
            updated_info.append(adjusted_run_info)

        # Combine processed runs for the subject
        processed_data[subject_idx] = np.vstack(subject_recentered_data)
        
    # remove the rest condition from the data by removing center_condition_indices
    processed_data = np.delete(processed_data,center_condition_indices,axis=1)

    # Combine info across all subjects and runs
    updated_info = pd.concat(updated_info, ignore_index=True)

    return processed_data, updated_info

def translate_battery(info,battery_indices):
    """
    Translate battery from indices to names
    Parameters:
        info(pd.DataFrame): task information tsv
        battery_indices(np.ndarray): indices of tasks in the battery
    Returns:
        battery_names(np.ndarray): names of tasks in the battery
    """
    names = info['names'].unique()
    battery_names = names[battery_indices]
    return battery_names


def build_battery_dataset(YLib, info, combination, n_repeats=1):
    """
    Constructs a dataset based on a task battery combination.
    
    Parameters:
        YLib (numpy.ndarray): The full data array of shape (subjects, regressors, voxels).
        info (pandas.DataFrame): Information about regressors.
        combination (list): List of indices to include in battery dataset
        n_repeats (int): how much data you want for that combination, default is 1 meaning just make one artificial run of the combination
    
    Returns:
        final_dataset (numpy.ndarray): The constructed task battery dataset of shape (subjects, regressors, voxels).
    """
    # list to include n_repeats of the combination dataset (artificial runs)
    Y_subset_list = []
    total_runs = info['run'].nunique()

    task_groups = info.groupby(['task_num_uni', 'cond_num'])
    task_only_groups = info.groupby(['task_num_uni'])
    
    # to make sure no regressor is chosen twice across n_repeats
    selected_indices = []
    for _ in range(n_repeats):
        Y_subset = []
        
        for idx in combination:
            task_num = info.loc[idx, 'task_num_uni']
            cond_num = info.loc[idx, 'cond_num']
            
           # Get regressor list for the 'task'
            task_group = task_only_groups.get_group(task_num)
            task_indices = task_group.index.tolist()
            
            # Get regressor list for the 'condition of interest'
            condition_group = task_groups.get_group((task_num, cond_num))
            condition_indices = condition_group.index.tolist()
            
            # Determine if it's a task or condition
            num_task_regressors = len(task_indices) // total_runs  # if result is > 1, it's a task with multiple conditions
            
            # Never choose the same regressor twice
            chosen_indices = []
            while len(chosen_indices) < num_task_regressors:
                chosen_idx = condition_indices[pt.randint(len(condition_indices), (1,)).item()]
                if chosen_idx not in selected_indices and chosen_idx not in chosen_indices: #hasn't been chosen for the preview repeat and hasn't been chosen if it's a condition
                    chosen_indices.append(chosen_idx)
                    selected_indices.append(chosen_idx)

            # if it's a 10s condition then average 3x10s to get 30s for example
            averaged_vector = pt.mean(YLib[:,chosen_indices, :], axis=1)
            Y_subset.append(averaged_vector)
        
        # make dataset for current repeat and append to the list
        Y_subset = pt.stack(Y_subset, axis=1)  
        Y_subset_list.append(Y_subset)
    
    # Average repeats
    stacked_Y = pt.stack(Y_subset_list, axis=0)
    final_dataset = pt.mean(stacked_Y, axis=0)
    return final_dataset

if __name__ == "__main__":
    N = 8 
    U = np.random.normal(0,1,(N,10))
    G = U @ U.T
    D = build_combinations(G, strategy='random',n_iter=100,n_tasks=4)
    pass