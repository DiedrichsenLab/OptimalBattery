# Module for functions used for optimal battery construction
# Author: Bassel Arafat
# Date: Oct 1st 2024

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

    # Compute eigenvalues and eigenvectors of both centered and uncentered G matrices
    l, _ = eigh(G)
    l = l[::-1]  # Reverse order
    # l = l[l > 1e-8]  # Remove very small eigenvalues

    # mc= mean centered
    l_mc, _ = eigh(G_mc)
    l_mc = l_mc[::-1]  
    # l_mc = l_mc[l_mc > 1e-8]


    # Create a dictionary of criteria
    d = {
        'variance': np.sum(l),
        'variance_mc': np.sum(l_mc),
        'inverse_trace': - np.sum(1 / l),
        'inverse_trace_mc': - np.sum(1 / l_mc),
        'log_det': np.sum(np.log(l)),
        'log_det_mc': np.sum(np.log(l_mc)),
        'eigenvalues':[l_mc.tolist()],
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



if __name__ == "__main__":
    N = 8 
    U = np.random.normal(0,1,(N,10))
    G = U @ U.T
    D = build_combinations(G, strategy='random',n_iter=100,n_tasks=4)
    pass