import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge


def generate_Us(s=24, k=16, p=40, type='hard', seed=1):
    """
    Generate the true U matrices for simulation using a normal distribution.
    
    Parameters:
    s: number of subjects
    k: number of parcels
    p: number of voxels
    type: type of Us
    seed: random seed for reproducibility

    Returns:
    Us: ndarray, shape (s, k, p)

    """
    np.random.seed(seed)

    if type == 'hard':
        values = np.random.normal(0, 1, (s, k, p))  # Shape: (s, k, p)
        
        # Find the max indices for each voxel
        max_indices = np.argmax(values, axis=1)  # Shape: (s, p)

        Us = np.zeros((s, k, p))

        for subj in range(s):
            Us[subj][max_indices[subj], np.arange(p)] = 1

    elif type == 'prob':
        # Generate soft assignments using a normal distribution
        Us = np.random.normal(0, 1, (s, k, p))  # Shape: (s, k, p)
        Us = np.exp(Us)  # Apply exponential function
        Us_sum = np.sum(Us, axis=1, keepdims=True)  # Calculate sum along parcel dimension
        Us = Us / Us_sum  # Normalize to get probabilities 

    return Us

 
def estimate_Us_ols(Y,V):
    """
    get U_hat using OLS regression
    Args:
        Y: Individual fMRI data (n_subjects, n_tasks, n_voxels)
        V: Functional Profile (n_tasks, n_components)
    Returns:
        U_hat: Individual parcellations (n_subjects, n_components, n_voxels)
    """
    # Uhat =  (V^T V)^-1 V^T Yyy


    U_hats = np.linalg.inv(V.T @ V) @ V.T @ Y
    return U_hats # s, k, p

def estimate_Us_ols(Y,V):
    """
    get U_hat using OLS regression
    Args:
        Y: Individual fMRI data (n_subjects, n_tasks, n_voxels)
        V: Functional Profile (n_tasks, n_components)
    Returns:
        U_hat: Individual parcellations (n_subjects, n_components, n_voxels)
    """
    # Uhat =  (V^T V)^-1 V^T Y

    VtV = V.T @ V
    
    if np.linalg.matrix_rank(VtV) < VtV.shape[0]: # mean subtracted so thats one less rank
        VtV += np.eye(VtV.shape[0]) * 0.001

    U_hats = np.linalg.inv(VtV) @ V.T @ Y
    return U_hats # s, k, p


def estimate_Us_NNLS(Y, V):
    """
    Estimate the U matrices using NNLS regression.

    Args:
        Y: Individual fMRI data (n_subjects, n_tasks, n_voxels)
        V: Functional Profile (n_tasks, n_components)

    Returns:
        U_hat: Individual parcellations (n_subjects, n_components, n_voxels)
    """
    n_subjects, n_tasks, n_voxels = Y.shape
    _, n_components = V.shape

    U_hat = np.zeros((n_subjects, n_components, n_voxels))

    for subj in range(n_subjects):
        for voxel in range(n_voxels):
            y_voxel = Y[subj, :, voxel]  # Shape: (n_tasks,)
            u_voxel, error = nnls(V, y_voxel)
            U_hat[subj, :, voxel] = u_voxel

    return U_hat


def estimate_Us_NNLS_lasso(Y, V, alpha=0.1, max_iter=1000):
    """
    Estimate the U matrices using NNLS regression with Lasso regularization.

    Args:
        Y: Individual fMRI data (n_subjects, n_tasks, n_voxels)
        V: Functional Profile (n_tasks, n_components)
        alpha: Regularization strength
        max_iter: Maximum number of iterations

    Returns:
        U_hat: Individual parcellations (n_subjects, n_components, n_voxels)
    """

    n_subjects, n_tasks, n_voxels = Y.shape
    _, n_components = V.shape

    U_hat = np.zeros((n_subjects, n_components, n_voxels))

    for subj in range(n_subjects):
        for voxel in range(n_voxels):
            y_voxel = Y[subj, :, voxel]  # Shape: (n_tasks,)

            # Create and fit the Lasso model for each voxel
            lasso = Lasso(alpha=alpha, positive=True)
            lasso.fit(V, y_voxel) 

            # Store weigts
            U_hat[subj, :, voxel] = lasso.coef_
    return U_hat

def estimate_Us_NNLS_ridge(Y, V, alpha=0.1, max_iter=1000):
    """
    Estimate the U matrices using NNLS regression with Ridge regularization (L2).

    Args:
        Y: Individual fMRI data (n_subjects, n_tasks, n_voxels)
        V: Functional Profile (n_tasks, n_components)
        alpha: Regularization strength for Ridge
        max_iter: Maximum number of iterations (currently not used in Ridge)

    Returns:
        U_hat: Individual parcellations (n_subjects, n_components, n_voxels)
    """
    n_subjects, n_tasks, n_voxels = Y.shape
    _, n_components = V.shape

    U_hat = np.zeros((n_subjects, n_components, n_voxels))

    # Loop over each subject and each voxel
    for subj in range(n_subjects):
        for voxel in range(n_voxels):
            y_voxel = Y[subj, :, voxel]  # Shape: (n_tasks,)

            # Create and fit the Ridge model for each voxel
            ridge = Ridge(alpha=alpha)
            ridge.fit(V, y_voxel)

            # Store the weights
            U_hat[subj, :, voxel] = ridge.coef_

    return U_hat
