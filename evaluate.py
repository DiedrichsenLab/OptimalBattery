"""
Module for evaluating the performance of task batteries for both simulations and real data.
Author: Bassel Arafat
"""
import torch as pt
import OptimalBattery.estimate as et
import OptimalBattery.util as ut
import numpy as np
import OptimalBattery.construct as ct
import Functional_Fusion.atlas_map as am
import cortico_cereb_connectivity.model as model
import Functional_Fusion.dataset as fdata
import os
import cortico_cereb_connectivity.evaluation as con_ev
import pandas as pd
import HierarchBayesParcel.util as util
import OptimalBattery.simulate as sim

# define paths
base_dir = 'Y:/data/'
if not os.path.exists(base_dir):
    base_dir = '/cifs/diedrichsen/data/'
func_fus_dir = os.path.join(base_dir, 'FunctionalFusion')


def get_prediction_error(ytest, vtest, U_hat, indices=None):
    """Compute the prediction error using

    Args:
        ytest (ndarray): Test data (subjects,conditions, voxels).
        vtest (ndarray): Test Vs
        U_hat (ndarray): Estimated Us.
        indices (list or None): Indices of the voxels to evaluate.

    Returns:
        cos_err (ndarray): Cosine error per subject.
        cos_mean (ndarray): Mean cosine error across all subjects.
    """
    # Ensure correct dimensions (batching subjects if needed)
    if U_hat.ndimension() == 2:
        U_hat = U_hat.unsqueeze(0)
    if ytest.ndimension() == 2:
        ytest = ytest.unsqueeze(0)

    # Compute yhat
    yhat = pt.matmul(vtest, U_hat)

    # Compute cosine error across all voxels
    if indices is not None:
        cosine_error_vox = 1 - pt.nansum(ytest[:, :, indices] * yhat[:, :, indices], dim=1)
    else:
        cosine_error_vox = 1 - pt.nansum(ytest * yhat, dim=1)

    # compute mean error per subject
    cos_err = pt.nanmean(cosine_error_vox, dim=1)
    cos_mean = pt.mean(cos_err)

    return cos_err, cos_mean

def get_prediction_error_cv(ytest, U_hat, indices=None):
    """
    Compute cross-validated prediction error using leave-one-subject-out approach,
    comparing voxel-wise activity profiles across conditions.

    Args:
        ytest (Tensor): Test data (subjects, conditions, voxels), shape (S, C, V).
        U_hat (Tensor): Estimated U matrices per subject, shape (S, P, V).
        indices (list or Tensor, optional): Subset of voxel indices to evaluate.

    Returns:
        cos_err (Tensor): Cosine error per subject.
        cos_mean (Tensor): Mean cosine error across all subjects.
    """
    if U_hat.ndimension() == 2:
        U_hat = U_hat.unsqueeze(0)
    if ytest.ndimension() == 2:
        ytest = ytest.unsqueeze(0)

    n_subjects = ytest.shape[0]
    cos_err = []

    for subj in range(n_subjects):
        # Leave-one-subject-out
        other_subs_idx = [i for i in range(n_subjects) if i != subj]
        other_subs_ytest = ytest[other_subs_idx]                 
        other_subs_uhats = U_hat[other_subs_idx]                

        # Compute V across n-1 subjects
        Y_train_t = other_subs_ytest.permute(0, 2, 1)       
        V = pt.matmul(other_subs_uhats, Y_train_t).mean(dim=0) 
        V = V.T                                  

        # Center and normalize each parcel's condition profile
        V = ut.center_matrix(V, axis=0)         
        V = ut.normalize_matrix(V, axis=0)        

        # Predict for left-out subject
        U_subj = U_hat[subj]                   
        yhat = pt.matmul(V, U_subj)               
        y_true = ytest[subj]                        

        if indices is not None:
            yhat = yhat[:, indices]
            y_true = y_true[:, indices]

        # Center and normalize each voxel's condition profile
        yhat = ut.center_matrix(yhat, axis=0)        
        yhat = ut.normalize_matrix(yhat, axis=0)    

        y_true = ut.center_matrix(y_true, axis=0)      
        y_true = ut.normalize_matrix(y_true, axis=0)  

        # Cosine similarity per voxel, then mean
        cos_sim_voxels = pt.sum(y_true * yhat, dim=0)      
        cos_sim_subj =pt.mean(cos_sim_voxels)

        cos_err.append(cos_sim_subj)

    cos_err = pt.stack(cos_err)
    cos_mean = pt.mean(cos_err)

    return cos_err, cos_mean



def fit_model(xtrain,ytrain,X_atlas,train_label_image):
    """Fit connectivity model for each subject
    Args:
        xtrain (ndarray): Cortical training data (subjects, conditions, vertices).
        ytrain (ndarray): Cerebellar training data (subjects, conditions, voxels).
        X_atlas (AtlasMap): Atlas map for cortical data needed getting cortical data into icosahedron parcels.
        train_label_image (list): List of label images for left and right hemispheres.
        Returns:
        conn_model_list (list): List of fitted connectivity models for each subject.
    """

    # initialize training dict
    conn_model_list = []
    # Loop over subjects
    for i in range(xtrain.shape[0]):
        Y  = ytrain[i]
        # get the mean across tessels for cortical data
        X = xtrain[i] # cortical training data in vertices   
        X_atlas.get_parcel(train_label_image, unite_struct = False)
        X, _ = fdata.agg_parcels(X, X_atlas.label_vector,fcn=np.nanmean) # cortical training data in icosahedron parcels

        # loop over subjects and train models
        alpha = np.exp(8) # based on connectivity paper?
        conn_model = getattr(model, 'L2regression')(alpha)

        # Fit model, get train and validate metrics
        conn_model.fit(X, Y)
        conn_model_list.append(conn_model)

    return conn_model_list

def evaluate_model(xtest,ytest,X_atlas,train_label_image,conn_model_list):
    """Evaluate connectivity model for each subject
    Args:
        xtest (ndarray): Cortical test data (subjects, conditions, vertices).
        ytest (ndarray): Cerebellar test data (subjects, conditions, voxels).
        X_atlas (AtlasMap): Atlas map for cortical data needed getting cortical data into icosahedron parcels.
        train_label_image (list): List of label images for left and right hemispheres.
        conn_model_list (list): List of fitted connectivity models for each subject.
    Returns:
        R_list (list): List of correlation values for each subject.
    """
    R_list = []
    # loop over subjects
    for i in range(xtest.shape[0]):
        # ytrain 
        Y = ytest[i]

        # get x test,mean across tessels for cortical data
        X = xtest[i] # cortical test data in vertices
        X_atlas.get_parcel(train_label_image, unite_struct = False)
        X, _ = fdata.agg_parcels(X, X_atlas.label_vector,fcn=np.nanmean) # cortical test data in icosahedron parcels

        # subject specific fitted model
        fitM = conn_model_list[i]

        # Get model predictions
        Y_pred = fitM.predict(X)
        R,_ = con_ev.calculate_R(Y, Y_pred)
        R_list.append(R)
    return R_list


def real_connectivity(G_library, condition_df,
                      cortical_train,cerebellar_train, # training data
                      cortical_test,cerebellar_test, # test data
                      battery_sizes = [3,4,5,6,7,8,9,10,12,14,16],
                      metrics  = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc'],
                      n_batteries = 1000,
                      rest_idx = 28,
                      n_iter=5,
                      scan_duration=8):
    """Evaluate the quality of a cortex-cerebellar connectivity model using different metrics
    
    Args:
        G_library (ndarray): task x task second moment matrix for the task library.
        condition_df (DataFrame): DataFrame containing condition information.
        cortical_train (ndarray): Cortical training data (subjects, conditions, vertices).
        cerebellar_train (ndarray): Cerebellar training data (subjects, conditions, voxels).
        cortical_test (ndarray): Cortical test data (subjects, conditions, vertices).
        cerebellar_test (ndarray): Cerebellar test data (subjects, conditions, voxels).
        battery_sizes (list): List of battery sizes to evaluate.
        metrics (list): List of metrics to use for selecting batteries.
        n_batteries (int): Number of batteries to sample for each size.
        rest_idx (int): Index of the rest condition in the task library.
        n_iter (int): Number of iterations to run for each battery size and metric.
        Returns:
        results_df (DataFrame): DataFrame containing evaluation results.
            """
    # get the label files for the icosahedron parcels
    train_label_image = []
    for hemi in ['L', 'R']:
        train_label_image.append(func_fus_dir + '/Atlases' + f'/tpl-fs32k' + f'/Icosahedron1002.{hemi}.label.gii')

    # get the atlases needed to average within the icosahedron parcels
    fs32k_atlas,_ = am.get_atlas('fs32k')
    
    results_df = pd.DataFrame()
    for n_task in battery_sizes:
        print(f"Evaluating battery size: {n_task}")
        for n in range(n_iter):
            print(f"Iteration: {n}")
            D = ct.build_combinations(G_library, strategy='random',n_batteries=n_batteries,n_tasks=n_task,seed = None,replacement=False,rest_idx= rest_idx)
            for metric in metrics:
                print(f"Evaluating metric: {metric}")
                D_best = ct.choose_combination(D,metric)
                top_comb = D_best['combination'].values[0]

                # get the regressors for training data
                combination_regressors = ct.build_combination_regressors(top_comb, condition_df=condition_df, localizer_time=scan_duration)

                # average, center and normalize the training data for cortex for this combination
                xtrain = ct.average_regressors(cortical_train, combination_regressors)
                xtrain = ut.center_matrix(xtrain, axis=1)
                xtrain = ut.normalize_matrix(xtrain, axis=1)

                # average, center and normalize the training data for cerebellum for this combination
                ytrain = ct.average_regressors(cerebellar_train, combination_regressors)
                ytrain = ut.center_matrix(ytrain, axis=1)
                ytrain = ut.normalize_matrix(ytrain, axis=1)

                # center and normalize the test data for cortex
                xtest = ut.center_matrix(cortical_test, axis=1)
                xtest = ut.normalize_matrix(xtest, axis=1)

                # center and normalize the test data for cerebellum
                ytest = ut.center_matrix(cerebellar_test, axis=1)
                ytest = ut.normalize_matrix(ytest, axis=1)
                
                # fit the models
                connectivity_models_subjects = fit_model(xtrain=xtrain, ytrain=ytrain,
                                                         X_atlas=fs32k_atlas,train_label_image=train_label_image)
                
                # evaluate the models
                R_list = evaluate_model(xtest=xtest, ytest=ytest,
                                         X_atlas=fs32k_atlas, train_label_image=train_label_image,
                                         conn_model_list=connectivity_models_subjects)
                # record
                D_ev = pd.DataFrame()
                D_ev['n_task'] = [n_task]
                D_ev['metric'] = [metric]
                D_ev['correlation'] = [R_list]
                D_ev['roi'] = ['cortex-cereebellum']
                D_ev['iteration'] = [n]
                results_df = pd.concat([results_df,D_ev],axis=0)
                results_df.reset_index(drop=True, inplace=True)
            

    return results_df


def real_parcellation(G_library,condition_df,
                        YLib,Ytest,
                        VLib,
                        evaluation_indices = None,
                        battery_sizes = [3,4,5,6,7,8,9,10,12,14,16],
                        metrics  = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc'],
                        n_batteries = 1000,
                        n_iter=5,
                        rest_idx = 28,
                        localizer_duration=8):
    """ Evaluate the parcellation performance for each combination in the DataFrame D

    Args:
        G_library (ndarray): task x task second moment matrix for the task library.
        condition_df (DataFrame): DataFrame containing condition information.
        YLib (ndarray): training data used for estimating parcellations (subjects, conditions, voxels). based on CondRun
        Ytest (ndarray): test data used for evaluating parcellation quality (subjects, conditions, voxels). based on CondAll
        VLib (ndarray): Vs for training data estimated from CondAll
        evaluation_indices (list or None): Indices of the brain vertices/voxels to evaluate.
        battery_sizes (list): List of battery sizes to evaluate.
        metrics (list): List of metrics to use for selecting batteries.
        n_batteries (int): Number of batteries to sample for each size.
        n_iter (int): Number of iterations to run for each battery size and metric.
        rest_idx (int): Index of the rest condition in the task library.
        localizer_duration (int): Duration of the localizer in seconds.
    Returns:
        results_df (DataFrame): DataFrame containing evaluation results.

    """
    # Center & Normalize ytest
    Ytest = ut.center_matrix(Ytest, axis=1)
    Ytest = ut.normalize_matrix(Ytest, axis=1)

    results_df = pd.DataFrame()
    for n_task in battery_sizes:
        print(f"Evaluating battery size: {n_task}")
        for n in range(n_iter):
            print(f"Iteration: {n}")
            D = ct.build_combinations(G_library, strategy='random',n_batteries=n_batteries,n_tasks=n_task,seed = None,replacement=False,rest_idx= rest_idx)
            for metric in metrics:
                print(f"Evaluating metric: {metric}")
                D_best = ct.choose_combination(D,metric)
                top_comb = D_best['combination'].values[0]

                # get the regressors for training data
                combination_regressors = ct.build_combination_regressors(top_comb, condition_df=condition_df, localizer_time=localizer_duration)

                # average, center and normalize the data used for the parcellation
                Ysubset = ct.average_regressors(YLib, combination_regressors)
                Ysubset = ut.center_matrix(Ysubset, axis=1)
                Ysubset = ut.normalize_matrix(Ysubset, axis=1)

                Vsubset = VLib[top_comb,:]
                Vsubset = ut.center_matrix(Vsubset, axis=0)
                Vsubset = ut.normalize_matrix(Vsubset, axis=0)

                Uhats =  et.estimate_Us(Ysubset, Vsubset, method='correlation',hard=True)

                cos_subjects, cos_mean = get_prediction_error_cv(Ytest, Uhats, indices=evaluation_indices)
                cos_subjects = cos_subjects.cpu().numpy().tolist()
               
                # record
                D_ev = pd.DataFrame()
                D_ev['n_task'] = [n_task]
                D_ev['metric'] = [metric]
                D_ev['cos_sim'] = [cos_subjects]
                D_ev['iteration'] = [n]
                results_df = pd.concat([results_df,D_ev],axis=0)
                results_df.reset_index(drop=True, inplace=True)
    return results_df

def find_single_contrast(Vs, regionA, regionB):
    """Find the indices of the two tasks that maximize the contrast between two regions.(in this case region 1 is functional region of interest and 2 is everything else)
    Args:
        Vs (ndarray): shape (n_tasks, parcels), task activation patterns.
        regionA (list or ndarray): Indices of parcels representing region A.
        regionB (list or ndarray): Indices of parcels representing region B.
    Returns:
        contrast_indices (list): Indices of the two tasks that maximize the contrast [max_task_idx, min_task_idx].
    """
    difference = Vs[:, regionA ] - Vs[:, regionB]
    sorted_idx = pt.argsort(difference) 

    min_idx = sorted_idx[0].item()
    max_idx = sorted_idx[-1].item()

    return [max_idx, min_idx]

def thresholded_contrast(task1, task2, threshold=0.85):
    """Compute the contrast between two tasks and apply thresholding.
    Args:
        task1 (Tensor): shape (n_subjects, n_voxels), activation patterns for task 1.
        task2 (Tensor): shape (n_subjects, n_voxels), activation patterns for task 2.
        threshold (float or None): Quantile threshold to apply. If None, no thresholding is applied.
    Returns:
        thresholded_data (Tensor): shape (n_subjects, n_voxels), thresholded contrast data.
    """
    contrast_data = task1 - task2  # Compute contrast
    if threshold is not None:
        # Compute per-subject thresholds
        subject_thresholds = pt.quantile(contrast_data, threshold, dim=1, keepdim=True) 
        
        # Apply thresholding: values below threshold -> -1, above threshold -> 1
        thresholded_data = pt.where(contrast_data < subject_thresholds, 0, 1).float()
    return thresholded_data

def size_matched_contrast(contrast_map, reference_mask, roi_indices):
    """
    For each subject, selects the top-N voxels from the contrast map within the ROI,
    where N is the number of voxels in the reference binary mask.

    Parameters:
    - contrast_map (np.ndarray): shape (n_subjects, n_voxels), contrast values per subject
    - reference_mask (np.ndarray): shape (n_subjects, n_voxels), binary mask (e.g., multitask)
    - roi_indices (np.ndarray): indices of ROI voxels to consider

    Returns:
    - binary_masks (np.ndarray): shape (n_subjects, n_voxels), new binary masks
    """
    n_subjects, n_voxels = contrast_map.shape
    binary_masks = np.zeros_like(contrast_map)

    for i in range(n_subjects):
        n_voxels_to_select = int(reference_mask[i].sum())
        contrast_vals = contrast_map[i, roi_indices]

        sorted_indices = np.argsort(contrast_vals)
        top_indices = sorted_indices[-n_voxels_to_select:]

        temp_mask = np.zeros_like(contrast_vals)
        temp_mask[top_indices] = 1.0

        full_mask = np.zeros(n_voxels, dtype=np.float32)
        full_mask[roi_indices] = temp_mask

        binary_masks[i] = full_mask

    return binary_masks

def real_localization_multi(G_Lib,combination=None,task_names_s1=None,
                        condition_df= None, ROI_mask=None,
                        data_train=None,full_vs_train=None,parcel_interest_idx=None):
    if combination is None:
        D = ct.build_combinations(G_lib=G_Lib, strategy='random',n_batteries=10000,n_tasks=4,seed = None,replacement=False,rest_idx= 0)
        D_best = ct.choose_combination(D,'log_det_mc')
        comb = D_best['combination'].values[0]
    else:
        comb = combination

    # store top_comb names
    comb_names = [task_names_s1[i] for i in comb]

    # get the regressors for training data
    combination_regressors = ct.build_combination_regressors(comb, condition_df=condition_df, localizer_time=8,seed=1)

    # average, center and normalize the data used for the parcellation
    Ysubset = ct.average_regressors(data_train, combination_regressors)
    Ysubset = ut.center_matrix(Ysubset, axis=1)
    Ysubset = ut.normalize_matrix(Ysubset, axis=1)

    Vsubset = full_vs_train[comb,:]
    Vsubset = ut.center_matrix(Vsubset, axis=0)
    Vsubset = ut.normalize_matrix(Vsubset, axis=0)

    Uhats_multi =  et.estimate_Us(Ysubset, Vsubset, method='correlation',hard=True)
    Uhats_multi = pt.argmax(Uhats_multi,axis=1) + 1
    Uhats_multi_masked = Uhats_multi * ROI_mask # field of view mask
    Uhats_multi_masked = Uhats_multi_masked.cpu().numpy().astype(np.float32)
    Uhats_multi_collapsed = np.where(Uhats_multi_masked == parcel_interest_idx, 1, 0)

    return comb_names, Uhats_multi_masked, Uhats_multi_collapsed


def calculate_interaction_matrix(multi_mask, single_mask, contrast1, contrast2):
    """
    Calculates a 2x2 interaction matrix for each subject and localizer.

    For each subject:
        - Rows = contrasts: contrast1 (e.g. language), contrast2 (e.g. n-back)
        - Columns = inside and outside the mask

    Parameters:
    - multi_mask: (n_subjects, n_voxels)
    - single_mask: (n_subjects, n_voxels)
    - contrast1: (n_subjects, n_voxels)
    - contrast2: (n_subjects, n_voxels)

    Returns:
    - interaction_matrix: (n_subjects, 2, 2, 2)
        [subject, localizer (multi=0/single=1), contrast (1=lang/2=nback), in/out (0=in, 1=out)]
    """
    n_subs = multi_mask.shape[0]
    interaction_matrix = np.zeros((n_subs, 2, 2, 2))

    for i in range(n_subs):
        for loc_idx, mask in enumerate([multi_mask, single_mask]):
            for con_idx, contrast in enumerate([contrast1, contrast2]):
                in_vals = contrast[i][mask[i] == 1]
                out_vals = contrast[i][mask[i] == 0]
                interaction_matrix[i, loc_idx, con_idx, 0] = np.nanmean(in_vals)
                interaction_matrix[i, loc_idx, con_idx, 1] = np.nanmean(out_vals)

    return interaction_matrix

def compute_interaction_scores(interaction_matrix):
    """
    Computes interaction scores from a precomputed interaction matrix.

    Interaction score is:
    (contrast1_in - contrast1_out) - (contrast2_in - contrast2_out)

    Parameters:
    - interaction_matrix: (n_subjects, 2, 2, 2)
        [subject, localizer, contrast, in/out]

    Returns:
    - scores: (n_subjects, 2)
        scores[:, 0] = multitask
        scores[:, 1] = single
    """
    contrast1_in = interaction_matrix[:, :, 0, 0]
    contrast1_out = interaction_matrix[:, :, 0, 1]
    contrast2_in = interaction_matrix[:, :, 1, 0]
    contrast2_out = interaction_matrix[:, :, 1, 1]

    scores = (contrast1_in - contrast1_out) - (contrast2_in - contrast2_out)
    return scores

def calculate_spatial_overlap(U_binary):
    """
    Compute average Dice coefficient across all pairs of subjects.

    Args:
        U_binary (np.ndarray): shape (n_subjects, n_voxels), binary masks

    Returns:
        mean_dice (float): Mean Dice coefficient across all subject pairs
    """
    n_subs = U_binary.shape[0]
    scores = []
    for i in range(n_subs):
        for j in range(i + 1, n_subs):
            dice = sim.get_dice_binary(
                pt.tensor(U_binary[i][None, None, :]), 
                pt.tensor(U_binary[j][None, None, :])
            )
            scores.append(dice)
    return np.mean(scores)


if __name__=='__main__':
    # U_hat = pt.random.rand(3,10,6000)
    real_connectivity()
    pass

