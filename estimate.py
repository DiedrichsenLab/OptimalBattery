"""
Module for estimating  parcellations and functional profiles 
Author: Bassel Arafat
"""

import torch as pt
import numpy as np

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
        data (np.ndarray): fMRI data of shape (n_subjects, n_conditions, n_voxels).
        parcellation (np.ndarray): Parcellation indices of shape (n_voxels,).
        ROI (np.ndarray): Binary mask (0s and 1s) of shape (n_voxels,).
        parcel_list (list, optional): List of parcel indices to include. If None, selects all parcels in ROI.

    Returns:
        Vs (np.ndarray): Averaged values for each condition within selected parcels, shape (n_conditions, n_selected_parcels).
        selected_parcels (list): List of selected parcel indices.
    """
    avg_data = np.mean(data, axis=0)  # Average across subjects
    ROI_voxels = np.where(ROI == 1)[0]  # Get voxel indices in ROI
    
    # If no parcel_list is given, use all parcels in ROI
    if parcel_list is None:
        parcel_list = np.unique(parcellation[ROI_voxels])

    Vs = np.array([
        avg_data[:, np.intersect1d(np.where(parcellation == p)[0], ROI_voxels)].mean(axis=1)
        for p in parcel_list
    ]).T

    return Vs, parcel_list  


def get_training_Vs(data, parcellation, ROI, n_parcels=None):
    """
    Compute Vs using top n_parcels based on mean activation or all parcels if n_parcels is None.

    Parameters:
        data (np.ndarray): fMRI data of shape (n_subjects, n_conditions, n_voxels).
        parcellation (np.ndarray): Parcellation indices of shape (n_voxels,).
        ROI (np.ndarray): Binary mask (0s and 1s) of shape (n_voxels,).
        n_parcels (int, optional): Number of top parcels to use. If None, uses all parcels in ROI.

    Returns:
        Vs (np.ndarray): Averaged values for each condition within selected parcels, shape (n_conditions, n_selected_parcels).
        selected_parcels (list): List of selected parcel indices.
    """
    # Compute Vs for all parcels in ROI
    Vs, all_parcels = get_Vs(data, parcellation, ROI)  

    # Select top n_parcels based on mean activation
    if n_parcels is not None:  
        mean_activations = Vs.mean(axis=0)  
        top_indices = np.argsort(mean_activations)[::-1][:n_parcels]
        Vs = Vs[:, top_indices]
        selected_parcels = [all_parcels[i] for i in top_indices]  # Get corresponding parcel indices
    else:
        selected_parcels = all_parcels  # or just use all parcels

    return Vs, selected_parcels

def get_testing_Vs(data, parcellation, ROI, selected_parcels):
    """
    Compute Vs using selected_parcels based on training data.

    Parameters:
        data (np.ndarray): fMRI data of shape (n_subjects, n_conditions, n_voxels).
        parcellation (np.ndarray): Parcellation indices of shape (n_voxels,).
        ROI (np.ndarray): Binary mask (0s and 1s) of shape (n_voxels,).
        selected_parcels (list): List of selected parcel indices.

    Returns:
        Vs (np.ndarray): Averaged values for each condition within selected parcels, shape (n_conditions, n_selected_parcels).
    """
    Vs, _ = get_Vs(data, parcellation, ROI, parcel_list=selected_parcels)  # Use selected_parcels from training data
    return Vs