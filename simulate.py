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
    # Uhat =  (V^T V)^-1 V^T Y

    # ensure number of parcels is less than or equal to number of voxels
    U_hats = np.linalg.inv(V.T @ V) @ V.T @ Y
    return U_hats # s, k, p


def U_MSE(U_true, U_pred):
    MSE = []
    for subject in range(U_true.shape[0]):
        mse = np.mean((U_true[subject] - U_pred[subject])**2)
        MSE.append(mse)
    return np.mean(MSE)

