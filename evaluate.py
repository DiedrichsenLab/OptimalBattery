"""
Module for evaluating the performance of task batteries for both simulations and real data.
Author: Bassel Arafat
"""
import torch as pt
import OptimalBattery.estimate as et
import OptimalBattery.util as ut
import numpy as np
import construct as ct

def get_percentage_correct(U_true, U_pred): 
    """Compute the percentage of correctly classified voxels.
    Args:
        U_true: True Us
        U_pred: Estimated Us
    return:
        percentage: Percentage of correctly classified voxels
    """
    correct_voxels = pt.sum(U_true * U_pred)
    total_voxels = U_true.shape[2]
    percentage = (correct_voxels / total_voxels) * 100
    return percentage


def get_prediction_error(ytest, vtest, U_hat, indices=None):
    """Compute the prediction error using PyTorch (supports GPU for speedup because this was a bottleneck).
    
    Args:
        ytest (ndarray): Test data (subjects,conditions, voxels).
        vtest (ndarray): Test Vs
        U_hat (ndarray): Estimated Us.
        indices (list or None): Indices of the voxels to evaluate.
        use_cuda (bool): Whether to use GPU.
    
    Returns:
        avg_cos (float): Mean prediction error across subjects.
        cos_std (float): Standard deviation of prediction error.
    """
    # Ensure correct dimensions (batching subjects if needed)
    if U_hat.ndimension() == 2:
        U_hat = U_hat.unsqueeze(0)  
    if ytest.ndimension() == 2:
        ytest = ytest.unsqueeze(0) 

    # Compute yhat
    yhat = pt.bmm(vtest.unsqueeze(0).expand(U_hat.shape[0], -1, -1), U_hat)

    # Compute cosine error across all voxels
    if indices is not None:
        cosine_error_vox = 1 - pt.nansum(ytest[:, :, indices] * yhat[:, :, indices], dim=1)
    else:
        cosine_error_vox = 1 - pt.nansum(ytest * yhat, dim=1)

    # compute mean error per subject
    cos_err = pt.nanmean(cosine_error_vox, dim=1)
    cos_mean = pt.mean(cos_err)

    return cos_err, cos_mean

def sim_evaluate_combination_multiregion(combination,
                                     Ytrue,Vr,Ur,
                                     n_iter=10,
                                     sig_e=0.04,
                                     vtest = None,ytest = None):
    """Evaluate the parcellation performance for a single combination of tasks.
    Args:
        combination: The combination of tasks to evaluate
        Ytrue: True tuning functions of all voxels across all tasks (generated using the fine 25 region parcellation)
        Vr: The reduced task matrix for the regions you want to discover
        Ur: The reduced parcellation (correct answer)
        n_iter: Number of iterations to run
        sig_e: Standard deviation of the noise to add to the data
        vtest: Test v vectors
        ytest: Test data

    return:
        cos_mean: Mean prediction error
        perc_mean: Mean percentage of correctly classified voxels
        perc_sem: Standard error of the mean of the percentage of correctly classified voxels
    """
    # Get the task subset indices and corresponding data
    task_subset_indices = list(combination)

    V_subset = Vr[task_subset_indices,:]
    V_subset = center_matrix(V_subset,axis=0)
    V_subset = normalize_matrix(V_subset,axis=0)
    y_subset = Ytrue[task_subset_indices,:]
    
    perc = pt.zeros((n_iter,))
    cos = pt.zeros((n_iter,))
    for i in range(n_iter):
        y = y_subset + pt.random.normal(0, sig_e, y_subset.shape)
        y_norm = center_matrix(y,axis=0)
        y_norm = normalize_matrix(y_norm,axis=0)

        U_hat = et.estimate_Us_projection(y_norm, V_subset)
        U_hat_one_hot = get_U_hat_one_hot(U_hat)
    
        #eval
        perc[i] = get_percentage_correct(Ur, U_hat_one_hot)
        cos[i],_ = get_prediction_error(ytest,vtest,U_hat_one_hot)

    perc_mean = pt.mean(perc)
    perc_sem = pt.std(perc)/ pt.sqrt(n_iter) 
    cos_mean = pt.mean(cos)

    return  cos_mean, perc_mean,perc_sem


def sim_evaluate_dataframe_multiregion(D,
                                               Ytrue,Vr, Ur,
                                               sig_e=1,
                                               vtest = None,ytest = None):
    """ Evaluate the parcellation performance for each combination in the DataFrame D.

        Args:
            D: DataFrame containing the combinations to evaluate
            Ytrue: True tuning functions of all voxels across all tasks (generated using the fine 25 region parcellation)
            Vr: The reduced task matrix for the regions you want to discover
            Ur: The reduced parcellation (correct answer)  
            estimation_method: The method to estimate the parcellation
            sig_e: Standard deviation of the noise to add to the data
            vtest: Test v vectors
            ytest: Test data

    return:
        D: DataFrame with the computed percentage of correctly classified voxels and prediction error
        """
    D_eval = D.copy()
    D_eval['perc'] = None
    D_eval['perc_sem'] = None
    D_eval['cos'] = None

    for i in range(len(D)):
        combination = D_eval['combination'].iloc[i]
        cos,perc,perc_sem = sim_evaluate_combination_multiregion(combination,Ytrue,Vr, Ur,sig_e=sig_e,vtest = vtest,ytest = ytest)
        
        D_eval.loc[i, 'cos'] = cos
        D_eval.loc[i, 'perc'] = perc
        D_eval.loc[i, 'perc_sem'] = perc_sem
    return D_eval

def evluate_dataframe(D,condition_df,
                        YLib,VLib,
                        Ytest, Vtest,
                        indices = None):
    """ Evaluate the parcellation performance for each combination in the DataFrame D.
    
            Args:
                D: DataFrame containing the combinations to evaluate
                condition_df: dataframe that contains how long each condition is and it's indices
                YLib: The training data (all tasks all voxels) (subjects, conditions x repitions, voxels)
                VLib: Activity profiles for training data (tasks,parcels)
                Ytest: The test data (all tasks all voxels) (subjects, conditions, voxels)
                Vtest: Activity profiles for test data (tasks,parcels)
                indices: The indices of the voxels to evaluate in
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
        combination_regressors = ct.build_combination_regressors(combination, condition_df,localizer_time=8)

        # build the actual artificial localizer data
        YLib_subset = ct.average_regressors(YLib, combination_regressors)
        YLib_subset = ut.center_matrix(YLib_subset, axis=1)
        YLib_subset = ut.normalize_matrix(YLib_subset, axis=1)

        # get the Vs for the combination
        VLib_subset = VLib[combination, :]
        VLib_subset = ut.center_matrix(VLib_subset, axis=0)
        VLib_subset = ut.normalize_matrix(VLib_subset, axis=0)

        # Build the parcellation
        U_hats = et.estimate_Us(YLib_subset, VLib_subset,method='correlation',hard= True)

        # evaluate the parcellation
        cos_subs, cos_mean = get_prediction_error(Ytest, Vtest, U_hats, indices=indices)
        D.at[i, 'cos_subjects'] = cos_subs.cpu().numpy().tolist()
        D.loc[i, 'cos_mean'] = cos_mean.item()
    return D

    











    

if __name__=='__main__':
    U_hat = pt.random.rand(3,10,6000)
    U_hat_one = get_U_hat_one_hot(U_hat)
    pass

