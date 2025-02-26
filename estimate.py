"""
Module for estimating  parcellations and functional profiles 
Author: Bassel Arafat
"""

import torch as pt
import numpy as np
import OptimalBattery.evaluate as ev
import OptimalBattery.util as ut

######################### Parcellation Estimation #########################
def estimate_Us(Y, V, method='correlation', alpha=1e-3, hard=False):
    """
    Estimate U_hat using different projection methods: 'correlation', 'ols', or 'ridge'.

    Args:
        Y (torch.Tensor): fMRI data of shape (n_subjects, n_tasks, n_voxels)
        V (torch.Tensor): Functional profile of shape (n_tasks, n_parcels)
        method (str): Choice of {'correlation', 'ols', 'ridge'}
        alpha (float): Regularization for ridge (ignored unless method == 'ridge')
        hard (bool): If True, returns one-hot assignment for each voxel to one parcel

    Returns:
        U_hats (torch.Tensor): 
          If hard=False, shape = (n_subjects, n_parcels, n_voxels) with continuous weights
          If hard=True,  shape = (n_subjects, n_parcels, n_voxels) with 0/1 assignments
    """
    # 1) Compute weights depending on method
    if method == 'correlation':
        # correlation ~ (V^T @ Y)
        U_hats = V.T @ Y

    elif method == 'ols':
        # OLS: (V^T V)^(-1) V^T @ Y
        A = V.T @ V
        A_inv = pt.linalg.inv(A)
        U_hats = A_inv @ (V.T @ Y)

    elif method == 'ridge':
        # Ridge: (V^T V + alpha*I)^(-1) V^T @ Y
        A = V.T @ V
        alpha_eye = pt.eye(A.shape[0], device=A.device) * alpha
        A_inv = pt.linalg.inv(A + alpha_eye)
        U_hats = A_inv @ (V.T @ Y)
    else:
        raise ValueError(f"Invalid method")

    # 2) Return continuous or hard assignments
    if hard:
        max_indices = pt.argmax(U_hats, dim=1)  # (n_subjects, n_voxels)
        U_hard = pt.zeros_like(U_hats)
        U_hard.scatter_(1, max_indices.unsqueeze(1), 1)
        return U_hard

    return U_hats


######################### V estimation #####################################

def estimate_Vs(data, parcellation, ROI_mask = None):
    """
    Compute Vs by averaging data within the parcels from a given parcellation. can be restricted to a ROI + parcellation overlap
    Parameters:
        data (torch.Tensor): fMRI data of shape (n_subjects, n_tasks, n_voxels).
        parcellation (torch.Tensor): Parcellation indices of shape (n_voxels,).
        ROI_mask (torch.Tensor): Binary mask of shape (n_voxels,) indicating the region of interest. (optional)
    Returns:
        Vs (torch.Tensor): Averaged values for each condition within selected parcels, shape (n_tasks, n_selected_parcels).
    """
    # Average across subjects
    avg_data = data.mean(dim=0)

    # Get the values of the unique parcels in the parcellation
    parcel_list = pt.unique(parcellation)

    Vs = []
    for p in parcel_list:
        if ROI_mask is None:
            # Get the voxels that are in the parcel
            overlap_indices  = pt.where(parcellation == p)[0]
        else:
            # Get the voxels that are in both the parcel and the ROI
            overlap_indices  = pt.where((parcellation == p) & (ROI_mask>0))[0]
        # if there are no voxels in the parcel that are in the ROI, skip
        if len(overlap_indices) == 0:
            continue
        
        # Get the data for the voxels in the parcel that are in the ROI
        parcel_data = avg_data[:, overlap_indices]
        parcel_data = parcel_data.mean(dim=1)
        Vs.append(parcel_data)

    Vs = pt.stack(Vs, dim=1)
    return Vs


if __name__ == "__main__":
    # Vs = pt.rand(29, 5)
    # data = pt.rand(24,29,100)
    # Us = estimate_Us(data, Vs,method = 'correlation', hard=True)

    pass
