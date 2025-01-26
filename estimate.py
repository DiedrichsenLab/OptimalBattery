"""
Module for estimating the parcellations using different methods
Author: Bassel Arafat
"""

import numpy as np
from HierarchBayesParcel.util import indicator
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
import torch as pt

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

def estiamte_HBP_U(train_data,cond_vec,part_vec,ar_model):
    x_matrix = indicator(cond_vec)
    
    
    em_model = em.MixVMF(K=ar_model.K, P=ar_model.P, X=x_matrix, part_vec=part_vec,
                         subject_specific_kappa=False, parcel_specific_kappa=False, 
                         subjects_equal_weight=True)
    
    M_1 = fm.FullMultiModel(arrange=ar_model, emission=[em_model])
    M_1.initialize([train_data])

    M_1, ll,_,U_individual = M_1.fit_em(iter=200, tol=0.01,
                                     fit_arrangement=False,
                                     fit_emission= True,
                                     first_evidence=False)    

    # Get the data-only parcellation
    emloglik = M_1.emissions[0].Estep()
    U_data = pt.softmax(emloglik, dim=1) # get data only parcellation

    return U_data