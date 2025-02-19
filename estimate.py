"""
Module for estimating  parcellations and functional profiles 
Author: Bassel Arafat
"""

import torch as pt
import numpy as np
import OptimalBattery.evaluate as ev

######################### Parcellation Estimation #########################

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
    U_hats = pt.linalg.inv(V.T @ V) @ V.T @ Y
    return U_hats # k, p

def estimate_Us_projection(Y,V):
    """
    get U_hat using projection
    Args:
        Y: Individual fMRI data (n_tasks, n_voxels)
        V: Functional Profile (n_tasks, n_components)
    Returns:
        U_hat: Individual parcellations (n_components, n_voxels)
    """
    U_hats= V.T @ Y
    return U_hats # k, p

######################### V estimation #####################################

def get_Vs(data, parcellation, ROI, parcel_list=None):
    """
    Compute Vs matrix by averaging data within the given parcels.
    If parcel_list is None, it selects all parcels in the ROI.

    Parameters:
        data (torch.Tensor): fMRI data of shape (n_subjects, n_conditions, n_voxels).
        parcellation (torch.Tensor): Parcellation indices of shape (n_voxels,).
        ROI (torch.Tensor): Binary mask (0s and 1s) of shape (n_voxels,).
        parcel_list (list, optional): List of parcel indices to include. If None, selects all parcels in ROI.

    Returns:
        Vs (torch.Tensor): Averaged values for each condition within selected parcels, shape (n_conditions, n_selected_parcels).
        selected_parcels (list): List of selected parcel indices.
    """
    avg_data = data.mean(dim=0)  # Average across subjects
    ROI_voxels = pt.where(ROI == 1)[0]
    
    # Select all unique parcels in ROI if parcel_list is not provided
    if parcel_list is None:
        parcel_list = pt.unique(parcellation[ROI_voxels]).tolist()

    Vs = pt.stack([
        avg_data[:, pt.tensor(np.intersect1d(pt.where(parcellation == p)[0].cpu().numpy(), ROI_voxels.cpu().numpy()), device=avg_data.device)].mean(dim=1)
        for p in parcel_list], dim=1)

    return Vs


def get_largest_parcels(data, Vs, ROI_mask):
    """
    Compute the voxel count for each parcel within the ROI across subjects.
    
    Args:
        data (torch.Tensor): fMRI dataset for all subjects.
        Vs (torch.Tensor): Functional Profile matrix for all parcels
        ROI (torch.Tensor): Binary mask of the ROI
    
    Returns:
        torch.Tensor: Ordered indices of parcels based on voxel count.
    """
    ROI_indices = pt.where(ROI_mask == 1)[0]
    total_parcel_counts = pt.zeros(Vs.shape[1])
    for subject_data in data:
        data = subject_data[:, ROI_indices]
        data_projected = estimate_Us_projection(data, Vs)
        data_projected_onehot = ev.get_U_hat_one_hot(data_projected)[0]
        total_parcel_counts += pt.sum(data_projected_onehot, axis=1)
    
    top_parcels = pt.argsort(total_parcel_counts, descending=True)
    
    return top_parcels
