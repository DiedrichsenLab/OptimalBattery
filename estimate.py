import numpy as np



def estimate_Us_ols(Y,V,regularize = None):
    """
    get U_hat using OLS regression
    Args:
        Y: Individual fMRI data (n_subjects, n_tasks, n_voxels)
        V: Functional Profile (n_tasks, n_components)
    Returns:
        U_hat: Individual parcellations (n_subjects, n_components, n_voxels)
    """
    # Uhat =  (V^T V)^-1 V^T Y
    if regularize is not None:
        U_hats = np.linalg.inv(V.T @ V + regularize*np.eye(V.shape[1])) @ V.T @ Y
    else:
        U_hats = np.linalg.inv(V.T @ V) @ V.T @ Y
    return U_hats # s, k, p