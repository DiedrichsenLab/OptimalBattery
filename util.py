"""
Module for optimal battery construction
Author: Bassel Arafat
"""

import numpy as np
import pandas as pd
from numpy.linalg import eigh



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
    # l = l[l > 1e-8]  # Remove very small eigenvalues
    l[l < 1e-12] = 1e-12

    l_mc, _ = eigh(G_mc)
    l_mc = l_mc[::-1]
    # l_mc = l_mc[l_mc > 1e-8]
    l_mc[l_mc < 1e-12] = 1e-12
        

    # Create a dictionary of criteria
    d = {
        'variance': np.sum(l),  # Sum of uncentered eigenvalues
        'variance_mc': np.sum(l_mc),  # Sum of mean-centered eigenvalues
        'inverse_trace': - np.sum(1 / l),
        'inverse_trace_mc': - np.sum(1 / l_mc),
        'log_det': np.sum(np.log(l)),
        'log_det_mc': np.sum(np.log(l_mc)),
        'eigenvalues_pre': [l.tolist()],
        'eigenvalues': [l_mc.tolist()],
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

    elif strategy == 'exhaustive':
        pass 
    else:
        raise ValueError('Invalid strategy')
        
    for i in range(len(comb)):
        n_unique = len(set(comb[i]))
        d = eigenval_crit(G_lib[comb[i],:][:,comb[i]],center=True)
        d['n_tasks'] = [len(comb[i])]
        d['combination'] = [comb[i]]
        d['n_unique'] = [n_unique]
        D_list.append(pd.DataFrame(d))
    D = pd.concat(D_list)
    return D 

def recenter_fmri_data(data , info ,task_column_name = 'cond_name',center_condition='rest'):

    n_subjects, n_conditions, n_voxels = data.shape
    if 'run' in info.columns:
        runs = info['run'].unique()
    else:
        runs = [1]

    # Find indices of 'rest' condition for the different runs
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


def max_value_distribution_analysis(df, num_batteries, num_iterations, eval_metric):
    metrics = ['variance', 'variance_mc', 'inverse_trace', 'inverse_trace_mc', 'log_det', 'log_det_mc']
    results = []

    for _ in range(num_iterations):
        # Randomly sample task batteries without replacement
        sampled_df = df.sample(n=num_batteries, replace=False)

        iteration_results = {}
        for metric in metrics:
            # Find the row with the highest value for the current metric
            max_metric_row = sampled_df.loc[sampled_df[metric].idxmax()]
            # Record the evaluation metric (e.g., 'cos') value
            iteration_results[metric] = max_metric_row[eval_metric]

        results.append(iteration_results)
        
    result_df = pd.DataFrame(results)

    return result_df

def translate_battery(info,battery_indices):
    """
    Translate battery name to task name
    """
    names = info['names'].unique()
    battery_names = names[battery_indices]
    return battery_names

if __name__ == "__main__":
    N = 8 
    U = np.random.normal(0,1,(N,10))
    G = U @ U.T
    D = build_combinations(G, strategy='random',n_iter=100,n_tasks=4)
    pass