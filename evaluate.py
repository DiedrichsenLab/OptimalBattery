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


def evluate_dataframe_parcellation(D,condition_df,
                        YLib,VLib,
                        Ytest, Vtest,
                        indices = None,method='correlation',hard = True,alpha =1e-3,localizer_time=8):
    """ Evaluate the parcellation performance for each combination in the DataFrame D.

            Args:
                D: DataFrame containing the combinations to evaluate
                condition_df: dataframe that contains how long each condition is and it's indices
                YLib: The training data (all tasks all voxels) (subjects, conditions x repitions, voxels)
                VLib: Activity profiles for training data (tasks,parcels)
                Ytest: The test data (all tasks all voxels) (subjects, conditions, voxels)
                Vtest: Activity profiles for test data (tasks,parcels)
                indices: The indices of the voxels to evaluate in
                method: The method to use for estimating the Us
                localizer_time: The scanning time of the localizers in seconds
    return:
        D: DataFrame with the computed prediction error
        """
    D['cos_subjects'] = None
    D['cos_mean'] = None

    # Center & Normalize vtest
    Vtest = ut.center_matrix(Vtest, axis=0)
    Vtest = ut.normalize_matrix(Vtest, axis=0)

    # Center & Normalize ytest
    Ytest = ut.center_matrix(Ytest, axis=1)
    Ytest = ut.normalize_matrix(Ytest, axis=1)

    for i in range(len(D)):
        if i % 1000 == 0:
            print(f"Processing combination: {i}")
        # Get the combination
        combination = D['combination'].iloc[i]
        combination = list(combination)

        # construct the regressors
        combination_regressors = ct.build_combination_regressors(combination, condition_df,localizer_time=localizer_time)

        # build the actual artificial localizer data
        YLib_subset = ct.average_regressors(YLib, combination_regressors)
        YLib_subset = ut.center_matrix(YLib_subset, axis=1)
        YLib_subset = ut.normalize_matrix(YLib_subset, axis=1)

        # get the Vs for the combination
        VLib_subset = VLib[combination, :]
        VLib_subset = ut.center_matrix(VLib_subset, axis=0)
        VLib_subset = ut.normalize_matrix(VLib_subset, axis=0)

        # Build the parcellation
        U_hats = et.estimate_Us(YLib_subset, VLib_subset,method,hard= hard,alpha=alpha)

        # evaluate the parcellation
        cos_subs, cos_mean = get_prediction_error(Ytest, Vtest, U_hats, indices=indices)
        D.at[i, 'cos_subjects'] = cos_subs.cpu().numpy().tolist()
        D.loc[i, 'cos_mean'] = cos_mean.item()
    return D

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
                



if __name__=='__main__':
    # U_hat = pt.random.rand(3,10,6000)
    real_connectivity()
    pass

