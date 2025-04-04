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


def fit_model(xtrain,ytrain,X_atlas,train_label_image):
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
                      n_iter=5):
    """Evaluate the quality of a cortex-cerebellar connectivity model using different metrics"""
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
                combination_regressors = ct.build_combination_regressors(top_comb, condition_df=condition_df, localizer_time=8)

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
                D_ev['iteration'] = [n]
                results_df = pd.concat([results_df,D_ev],axis=0)
                results_df.reset_index(drop=True, inplace=True)
            

    return results_df


def real_parcellation(G_library,condition_df,
                        YLib,Ytest,
                        VLib,Vtest,
                        evaluation_indices = None,
                        battery_sizes = [3,4,5,6,7,8,9,10,12,14,16],
                        metrics  = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc'],
                        n_batteries = 1000,
                        n_iter=5,
                        rest_idx = 28,
                        localizer_duration=8):
    """ Evaluate the parcellation performance for each combination in the DataFrame D.

    """
    # Center & Normalize vtest
    Vtest = ut.center_matrix(Vtest, axis=0)
    Vtest = ut.normalize_matrix(Vtest, axis=0)

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

                cos_subjects, cos_mean = get_prediction_error(Ytest, Vtest, Uhats, indices=evaluation_indices)
                cos_subjects = cos_subjects.cpu().numpy().tolist()
               
                # record
                D_ev = pd.DataFrame()
                D_ev['n_task'] = [n_task]
                D_ev['metric'] = [metric]
                D_ev['cos_err'] = [cos_subjects]
                D_ev['iteration'] = [n]
                results_df = pd.concat([results_df,D_ev],axis=0)
                results_df.reset_index(drop=True, inplace=True)
    return results_df

def compute_region_profiles(test_data, contrast_data):
    n_subj, n_tasks, n_vox = test_data.shape
    region_profiles = pt.zeros((n_subj, n_tasks,1), device=test_data.device)

    for s in range(n_subj):
        # Subject-specific mask
        mask_1 = (contrast_data[s] == 1)
        for t in range(n_tasks):
            # Take the average activation for all voxels in mask_1
            region_profiles[s, t, 0] = test_data[s, t, mask_1].mean()
    return region_profiles

def average_pairwise_correlation(region_profiles):

    # flatten the region_profiles (subj,taskprofiles)
    n_subj =region_profiles.shape[0]
    X = region_profiles.reshape(n_subj, -1).cpu().numpy()

    # get the correlation matrix among all subjects
    subjects_corr_matrix = np.corrcoef(X, rowvar=True)

    # get cross sub corrs
    i_upper, j_upper = np.triu_indices(n_subj, k=1)
    corrs = subjects_corr_matrix[i_upper, j_upper]

    return np.mean(corrs)

def find_single_contrast(Vs, regionA, regionB):
    difference = Vs[:, regionA ] - Vs[:, regionB]
    sorted_idx = pt.argsort(difference) 

    min_idx = sorted_idx[0].item()
    max_idx = sorted_idx[-1].item()

    return [max_idx, min_idx]

def thresholded_contrast(task1, task2, threshold=0.85):

    contrast_data = task1 - task2  # Compute contrast
    if threshold is not None:
        # Compute per-subject thresholds
        subject_thresholds = pt.quantile(contrast_data, threshold, dim=1, keepdim=True) 
        
        # Apply thresholding: values below threshold -> -1, above threshold -> 1
        thresholded_data = pt.where(contrast_data < subject_thresholds, 0, 1).float()
    return thresholded_data

def real_localization_multi(G_library,condition_df,
                        YLib,VLib,Ytest,
                        evaluation_indices = None,
                        battery_sizes = [3,4,5,6,7,8,9,10,12,14,16],
                        metric = 'log_det_mc',
                        n_batteries = 1000,
                        n_iter=5,
                        rest_idx = 28,
                        localizer_duration=8,
                        target_parcel_idx = 0):
    """ Evaluate the localization performance multitask battery

    """
    results_df = pd.DataFrame()
    for n_task in battery_sizes:
        print(f"Evaluating battery size: {n_task}")
        for n in range(n_iter):
            print(f"Iteration: {n}")
            D = ct.build_combinations(G_library, strategy='random',n_batteries=n_batteries,n_tasks=n_task,seed = None,replacement=False,rest_idx= rest_idx)
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
            U_hats_collpased = sim.collapse_U(Uhats, target_parcel_idx=target_parcel_idx)
            U_binary  = U_hats_collpased[:,0,:]

            test_profile = compute_region_profiles(Ytest, U_binary)
            corr = average_pairwise_correlation(test_profile)
        
            # record
            D_ev = pd.DataFrame()
            D_ev['n_task'] = [n_task]
            D_ev['metric'] = [metric]
            D_ev['iteration'] = [n]
            D_ev['corr'] = [corr]
            results_df = pd.concat([results_df,D_ev],axis=0)
            results_df.reset_index(drop=True, inplace=True)

    return results_df

def real_localization_single(test_data,regiona_idx,regionb_idx,
                             full_vs_train,condition_df,data_train,ROI_indices,
                             thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95,0.99]):
    
    test_data = test_data[:,:,ROI_indices]
    # find top two tasks for the single contrast
    single_combination = find_single_contrast(full_vs_train, regiona_idx, regionb_idx)
    task_names = condition_df['cond_name'].values
    task_names = np.array(task_names)[single_combination]
    print(f"Single contrast tasks: {task_names}")

    # make the contrast data
    combination_regressors = ct.build_combination_regressors(single_combination, condition_df,localizer_time=8)
    contrast_data = ct.average_regressors(data_train, combination_regressors)
    contrast_data = ut.normalize_matrix(contrast_data, axis=1)
    masked_contrast_data = contrast_data[:,:,ROI_indices]

    results_df = pd.DataFrame()
    for threshold in thresholds:
        mask = thresholded_contrast(masked_contrast_data[:,0,:],masked_contrast_data[:,1,:], threshold=threshold)
        test_profile = compute_region_profiles(test_data, mask)
        corr = average_pairwise_correlation(test_profile)

        # record
        D_ev = pd.DataFrame()
        D_ev['threshold'] = [threshold]
        D_ev['corr'] = [corr]
        D_ev['regiona_idx'] = [regiona_idx]
        D_ev['regionb_idx'] = [regionb_idx]
        D_ev['single_combination'] = [single_combination]
        results_df = pd.concat([results_df,D_ev],axis=0)
        results_df.reset_index(drop=True, inplace=True)
    return results_df

                    



if __name__=='__main__':
    # U_hat = pt.random.rand(3,10,6000)
    real_connectivity()
    pass

