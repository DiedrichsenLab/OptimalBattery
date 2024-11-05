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

def random_matrix_normal(G, R, make_exact=False, rng=None):
    n_tasks = G.shape[0]
    n_parcels = R.shape[0]

    if rng is None:
        rng = np.random.default_rng()
    else:
        rng = rng
    V = rng.standard_normal((n_tasks, n_parcels))

    if make_exact:
        P_row = np.linalg.inv(V @ V.T)
        P_col = np.linalg.inv(V.T @ V)
        L_col = np.linalg.cholesky(P_col)
        L_row = np.linalg.cholesky(P_row)
        Vs = L_row.T @ V 
    else:
        Vs = V

    lam, eV = np.linalg.eigh(G)
    lam[lam < 1e-15] = 0
    lam = np.sqrt(lam)
    chol_G = eV * lam.reshape((1, eV.shape[1]))

    lam, eV = np.linalg.eigh(R)
    lam[lam < 1e-15] = 0
    lam = np.sqrt(lam)
    chol_R = eV * lam.reshape((1, eV.shape[1]))
    V = chol_G @ Vs @ chol_R.T

    return V

def U_MSE(U_true, U_pred):
    MSE = []
    # if its only two dimensions then add a dimension
    if len(U_true.shape) == 2:
        U_true = U_true.reshape(1, U_true.shape[0], U_true.shape[1])
        U_pred = U_pred.reshape(1, U_pred.shape[0], U_pred.shape[1])
    for subject in range(U_true.shape[0]):
        mse = np.mean((U_true[subject] - U_pred[subject])**2)
        MSE.append(mse)
    return np.mean(MSE)

if __name__=='__main__':
    N = 10
    R = np.random.normal(0,1,(4,4))
    C = np.random.normal(0,1,(4,4))
    cov_R = R @ R.T
    cov_V = C @ C.T 
    Vs = random_matrix_normal(cov_R, cov_V, make_exact=True)
    
    pass 