"""
Module for estimating  parcellations and functional profiles 
Author: Bassel Arafat
"""

import torch as pt
import numpy as np
import OptimalBattery.evaluate as ev

######################### Parcellation Estimation #########################

def estimate_Us(Y,V,method = 'correlation', hard = False):
    """
    get U_hat using projection
    Args:
        Y: Individual fMRI data (n_subjects,n_tasks, n_voxels)
        V: Functional Profile (n_tasks, n_parcels)
        method: 'correlation' or 'ols'
        hard: If True, returns one-hot encoding of the projection
    Returns:
        U_hats: Individual parcellations (n_subjects, n_parcels, n_voxels)
    """
    if method == 'correlation':
        U_hats= V.T @ Y
    elif method == 'ols':
        pt.linalg.inv(V.T @ V) @ V.T @ Y
    else:
        raise ValueError('Invalid method')
    if hard:
        max_indices = pt.argmax(U_hats, axis=1)
        U_hats = pt.zeros_like(U_hats)
        U_hats.scatter_(1, max_indices[:, None, :], 1)
    
    return U_hats # k, p

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
