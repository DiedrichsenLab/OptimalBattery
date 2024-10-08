import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import nnls
from sklearn.linear_model import Lasso


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
        Us = np.zeros((s, k, p))
        # For each subject, generate values using a normal distribution
        for subj in range(s):
            # Draw values from a normal distribution for each voxel with k dimensions
            values = np.random.normal(0, 1, (p, k))  # Shape: (p, k)

            # For each voxel, find the index of the maximum value and one-hot encode
            max_indices = np.argmax(values, axis=1)  # Shape: (p,)

            # Assign one-hot encoding: For each voxel, set the max index parcel to 1
            Us[subj, max_indices, np.arange(p)] = 1

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
    U_hats = np.linalg.inv(V.T @ V) @ V.T @ Y
    return U_hats # s, k, p

def estimate_Us_NNLS(Y, V, max_iter=1000, tol=1e-6, learning_rate=1e-3):
    """
    Estimate U_hat using Non-Negative Least Squares (NNLS) regression for multiple subjects and voxels.
     #uhat = argmin ||Y - VU||^2, U >= 0
    
    Args:
        Y (ndarray): Individual fMRI data, shape (n_subjects, n_tasks, n_voxels).
        V (ndarray): Functional Profile (dictionary matrix), shape (n_tasks, n_components).
        max_iter (int): Maximum number of iterations for gradient descent.
        tol (float): Convergence tolerance.
        learning_rate (float): Learning rate for gradient descent.
    
    Returns:
        U_hat (ndarray): Estimated individual parcellations, shape (n_subjects, n_components, n_voxels).
    """
    n_subjects, n_tasks, n_voxels = Y.shape
    _, n_components = V.shape

    U_hat = np.zeros((n_subjects, n_components, n_voxels))

    for subj in range(n_subjects):
        for voxel in range(n_voxels):
            y_voxel = Y[subj, :, voxel]  # Shape: (n_tasks,)

            u_voxel = np.zeros(n_components)
            for i in range(max_iter):
                # Compute the gradient: grad = -2 * V.T @ (Y - V @ U_hat)
                gradient = -2 * V.T @ (y_voxel - V @ u_voxel)

                # Update U_hat in the direction of the negative gradient
                u_voxel -= learning_rate * gradient

                # Project: Set negative values in U_hat to zero
                u_voxel = np.maximum(u_voxel, 0)

                # Check for convergence (if change in objective is small)
                error = np.linalg.norm(y_voxel - V @ u_voxel) ** 2
                if error < tol:
                    break

            # Store the result for the current subject and voxel
            U_hat[subj, :, voxel] = u_voxel

    return U_hat


def estimate_Us_l2_regularization(Y, V, alpha=0.1, max_iter=1000, tol=1e-6, learning_rate=1e-3):
    """
    Estimate U_hat using L2 Regularization (Ridge Regression)
    
    Args:
        Y (ndarray): Individual fMRI data, shape (n_subjects, n_tasks, n_voxels).
        V (ndarray): Functional Profile (dictionary matrix), shape (n_tasks, n_components).
        alpha (float): Regularization parameter for L2 norm.
        max_iter (int): Maximum number of iterations for gradient descent.
        tol (float): Convergence tolerance.
        learning_rate (float): Learning rate for gradient descent.
    
    Returns:
        U_hat (ndarray): Estimated individual parcellations, shape (n_subjects, n_components, n_voxels).
    """
    n_subjects, n_tasks, n_voxels = Y.shape
    _, n_components = V.shape

    U_hat = np.zeros((n_subjects, n_components, n_voxels))

    for subj in range(n_subjects):
        for voxel in range(n_voxels):
            y_voxel = Y[subj, :, voxel]  # Shape: (n_tasks,)
            u_voxel = np.zeros(n_components)
            
            for i in range(max_iter):
                # Compute the gradient of the squared error term: -2 * V.T @ (Y - V @ U_hat)
                gradient = -2 * V.T @ (y_voxel - V @ u_voxel)

                # Add the L2 regularization term to the gradient: + 2 * alpha * u_voxel
                gradient += 2 * alpha * u_voxel

                # Update U_hat in the direction of the negative gradient
                u_voxel -= learning_rate * gradient

                u_voxel = np.maximum(u_voxel, 0)

                # Check for convergence (if change in objective is small)
                error = np.linalg.norm(y_voxel - V @ u_voxel) ** 2 + alpha * np.linalg.norm(u_voxel) ** 2
                if error < tol:
                    break
            U_hat[subj, :, voxel] = u_voxel

    return U_hat





