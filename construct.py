"""
Module for optimal battery construction
Author: Bassel Arafat
"""

import numpy as np
import pandas as pd
from numpy.linalg import eigh
import torch as pt
import PcmPy as pcm

def eigenval_crit(G, center=True):
    """Computes various criteria based on the eigenvalues and mutual information of a matrix G.
    Assumes that G is symmetric.
    Args:
        G: Second moment matrix
        center: Whether to center the matrix
    Returns:
        d: Dictionary of results for the criteria
    """

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
        'num_eigenvalues': len(l_mc),
    }
    
    return d

def build_combinations(G_lib, strategy='random',n_batteries=1000,n_tasks=4,seed=None,replacement=True,rest_idx=None): 
    """ Builds a set of task-batteries and evalates them 
    Parameters:
        G_lib(np.ndarray): Second moment matrix
        strategy(str): Strategy for building combinations
        n_batteries(int): Number of batteries to build
        n_tasks(int): Number of tasks in the battery
        seed(int): Random seed
        replacement(bool): Whether to sample with replacement
        rest_idx(int): Index of the rest condition to be added to each combination
    Returns:
        D(pd.DataFrame): DataFrame containing the combinations and the eigenmetrics for each combination
    """

    np.random.seed(seed)
    # deal with having rest in every combination
    n_lib_task = G_lib.shape[0] - 1 # number of tasks excluding rest since its always added
    if rest_idx is not None:
        rest_idx_tuple=(rest_idx,)
    else:
        rest_idx_tuple = ()

    D_list = []
    comb = []
    if strategy == 'random':
        for _ in range(n_batteries):
            if rest_idx is None:
                candidate = tuple(sorted(np.random.choice(n_lib_task, size=n_tasks, replace=replacement)))
            else:
                candidate = tuple(sorted(np.random.choice(n_lib_task, size=n_tasks-1, replace=replacement)))
            candidate = candidate + rest_idx_tuple # add rest to the combination
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
    D = D.reset_index(drop=True)
    return D

def get_G(data,n_cond = 29,n_part = 16):
    """get the crossvalided covariance matrix of the data across all subjects"""

    # make conditiion and partition vectors
    cond_vec = np.tile(np.arange(1, n_cond + 1), n_part)
    part_vec = np.repeat(np.arange(1, n_part + 1), n_cond)

    # calculate G per subject and average across subjects
    Gs_list = []
    for i in range(data.shape[0]):
        Gs,_ = pcm.util.est_G_crossval(data[i] , cond_vec, part_vec)
        Gs_list.append(Gs)

    Gs_list = np.stack(Gs_list, 0)
    G_Lib = np.mean(Gs_list, axis=0)

    return G_Lib


def choose_combination(D,metric):
    """  choose the best battery based on some metric"""
    # sample random battery
    rng = np.random.default_rng()
    index = rng.integers(0,D.shape[0])
    if metric == 'random':
        D_best = D.iloc[[index]]
    else:
        D_best = D.iloc[[D[metric].idxmax()]]
        D_best = D_best.reset_index(drop=True)
    return D_best


def get_condition_indices(df,task_column_name = 'task_name',cond_column_name = 'cond_name'):
    """
    Get condition indices from a dataframe and record the duration of each condition
    Parameters:
        df(pd.DataFrame): dataframe containing condition indices needs to include:
            - 'cond_name': name of the condition
            - 'run': run number
            - 'task_name': name of the task
    Returns:
        condition_indices(np.ndarray): condition indices
    """
    unique_conditions = df[cond_column_name].unique()
    new_df = pd.DataFrame(columns=[cond_column_name, 'indices', 'duration'])
    
    # Filter only the first run
    first_run_df = df[df['run'] == df['run'].min()]
    task_run_counts = first_run_df.groupby(task_column_name)[cond_column_name].nunique()
    duration_map = {1: 30, 2: 15, 3: 10}
    
    # Populate the new dataframe
    for condition in unique_conditions:
        indices = df[df[cond_column_name] == condition].index.tolist()
        
        # Identify task_name for the condition from the original dataframe
        task_name = df[df[cond_column_name] == condition][task_column_name].values[0]
        num_conditions = task_run_counts.get(task_name, 1)
        duration = duration_map.get(num_conditions, 30)
        
        new_row = {cond_column_name: condition, 'indices': indices, 'duration': duration}
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
    
    return new_df

def build_combination_regressors(combination, condition_df, localizer_time=12, seed=None):
    """
    Constructs a regressor list for the given condition combination,
    ensuring the total scanning time is distributed approximately equally across selected conditions.
    Pads remaining time by sampling additional regressors as much as possible.

    Parameters:
        combination (list): List of condition indices for the current combination.
        condition_df (pd.DataFrame): Must include:
            - 'cond_name': name of the condition
            - 'indices': list of regressor indices
            - 'duration': duration of each regressor in seconds
        localizer_time (int): Total allowed time in minutes.
        seed (int): Optional seed for reproducibility.

    Returns:
        tuple: (List of lists of regressors per condition, total time in seconds)
    """
    if seed is not None:
        np.random.seed(seed)

    total_seconds = localizer_time * 60  # Convert minutes to seconds
    allocated_time_per_condition = total_seconds // len(combination)

    comb_regressors = []
    total_combination_time = 0
    chosen_set = set()

    # Step 1: Equally allocate time per condition
    for cond_idx in combination:
        row = condition_df.iloc[cond_idx]
        duration = row['duration']
        indices = row['indices']
        num_to_sample = allocated_time_per_condition // duration

        sampled = np.random.choice(indices, size=num_to_sample, replace=False)

        comb_regressors.append(list(sampled))
        chosen_set.update(sampled)
        total_combination_time += len(sampled) * duration

    # Step 2: Fill any leftover time with unsampled regressors
    remaining_time = total_seconds - total_combination_time

    # Prepare pool of all unused regressors across conditions
    all_remaining = []
    for i, cond_idx in enumerate(combination):
        row = condition_df.iloc[cond_idx]
        duration = row['duration']
        indices = row['indices']
        available = list(set(indices) - chosen_set)
        for idx in available:
            all_remaining.append((idx, duration, i)) 

    np.random.shuffle(all_remaining)

    for idx, dur, list_pos in all_remaining:
        if dur <= remaining_time:
            comb_regressors[list_pos].append(idx)
            remaining_time -= dur
            total_combination_time += dur
            if remaining_time < min([condition_df.iloc[i]['duration'] for i in combination]):
                break

    return comb_regressors

def average_regressors(run_data, regressor_groups):
    """
    Computes the average of selected regressors.

    Args:
        run_data : Input tensor of shape (subjects, regressors, voxels).
        regressor_groups (list of list of int): A list containing lists of regressor indices to be averaged.
    Returns:
        Ysubset : Averaged regressors of shape (subjects, number of tasks, voxels).
    """

    subjects, _, voxels = run_data.shape
    num_groups = len(regressor_groups)
    
    if type(run_data) is pt.Tensor:
        # initialize
        Ysubset = pt.empty((subjects, num_groups, voxels), dtype=run_data.dtype, device=run_data.device)
        # Compute the average for each group
        for i, indices in enumerate(regressor_groups):
            selected = run_data[:, indices, :]  # Gather the required regressors
            Ysubset[:, i, :] = selected.mean(dim=1)  # Average across regressors
    else:
        # initialize
        Ysubset = np.zeros((subjects, num_groups, voxels))
        # Compute the average for each group
        for i, indices in enumerate(regressor_groups):
            selected = run_data[:, indices, :]  # Gather the required regressors
            Ysubset[:, i, :] = np.nanmean(selected, axis=1)  # Average the selected regressors

    return Ysubset

if __name__ == "__main__":
    N = 8 
    U = np.random.normal(0,1,(N,10))
    G = U @ U.T
    D = build_combinations(G, strategy='random',n_iter=100,n_tasks=4)
    pass