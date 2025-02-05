"""
Module for evaluating the performance of task batteries for both simulations and real data.
Author: Bassel Arafat
"""

import numpy as np
import OptimalBattery.estimate as et

def center_matrix(X,axis =0):
    """Center the matrix by subtracting the mean.
    Args:
        X: matrix to center
        axis: Axis along which to center the data
    return:
        X: Centered matrix
    """
    mean = np.nanmean(X, axis=axis, keepdims=True)
    X = X - mean
    return X

def normalize_matrix(X,axis = 0):
    """Normalize the matrix by dividing by the norm.
    Args:
        X: matrix to normalize
        axis: Axis along which to normalize the data
    return:
        X: Normalized matrix
    """
    norm = np.sqrt(np.nansum(X**2, axis=axis, keepdims=True))
    norm = np.where(norm == 0, 1.0, norm)
    X = X / norm
    return X

def get_U_hat_one_hot(U_hat):
    """Convert the estimated Us to one-hot encoding
    Args:
        U_hat: Estimated Us
    return:
        U_hat_one_hot: One-hot encoding of the estimated Us
    """
    if U_hat.ndim == 2:
        U_hat = U_hat[np.newaxis, :, :]

    max_indices = np.argmax(U_hat, axis=1)
    U_hat_one_hot = np.zeros_like(U_hat)
    for subject in range(U_hat.shape[0]):
        U_hat_one_hot[subject, max_indices[subject], np.arange(U_hat.shape[2])] = 1
    return U_hat_one_hot

def get_percentage_correct(U_true, U_pred): 
    """Compute the percentage of correctly classified voxels.
    Args:
        U_true: True Us
        U_pred: Estimated Us
    return:
        percentage: Percentage of correctly classified voxels
    """
    correct_voxels = np.sum(U_true * U_pred)
    total_voxels = U_true.shape[2]
    percentage = (correct_voxels / total_voxels) * 100
    return percentage


def get_prediction_error(ytest,vtest,U_hat,indices = None):
    """Compute the prediction error for simulated data.
    Args:
        ytest: Test data
        vtest: Test v vectors
        U_hat: Estimated Us
        indices: The indices of the voxels to evaluate
    return:
        avg_cos: Mean prediction error across subjects
        cos_std: Standard deviation of the prediction error across subjects
    """
    if U_hat.ndim == 2:
        U_hat = U_hat[np.newaxis,:,:]
    if ytest.ndim == 2:
        ytest = ytest[np.newaxis,:,:]

    #normalize vtest
    vtest_normalized = normalize_matrix(vtest,axis = 0)
    # center and normalize ytest
    ytest_centered = center_matrix(ytest,axis = 1)
    ytest_normalized = normalize_matrix(ytest_centered,axis = 1)

    cos_err = np.zeros((U_hat.shape[0],))
    for i in range(U_hat.shape[0]):
        # get reconstructed y
        yhat = np.matmul(vtest_normalized,U_hat[i])
        if indices is not None:
            cosine_error_vox = 1 - np.nansum(ytest_normalized[i][:, indices] * yhat[:, indices], axis=0)
        else:
            cosine_error_vox = 1 - np.nansum(ytest_normalized[i] * yhat, axis=0)
        # mean across voxels within a subject
        cos_err[i] = np.nanmean(cosine_error_vox)
    
    #avg across subjects
    avg_cos = np.nanmean(cos_err)
    # std across subjects
    cos_std = np.nanstd(cos_err)

    return avg_cos, cos_std

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
    
    perc = np.zeros((n_iter,))
    cos = np.zeros((n_iter,))
    for i in range(n_iter):
        y = y_subset + np.random.normal(0, sig_e, y_subset.shape)
        y_norm = center_matrix(y,axis=0)
        y_norm = normalize_matrix(y_norm,axis=0)

        U_hat = et.estimate_Us_projection(y_norm, V_subset)
        U_hat_one_hot = get_U_hat_one_hot(U_hat)
    
        #eval
        perc[i] = get_percentage_correct(Ur, U_hat_one_hot)
        cos[i],_ = get_prediction_error(ytest,vtest,U_hat_one_hot)

    perc_mean = np.mean(perc)
    perc_sem = np.std(perc)/ np.sqrt(n_iter) 
    cos_mean = np.mean(cos)

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

def real_evaluate_combination_multiregion(combination,
                                           YLib,VLib,
                                           ytest, vtest,
                                           indices = None):
    """Evaluate the parcellation performance for a single combination of tasks.
    Args:
        combination: The combination of tasks to evaluate
        YLib: The data for all tasks
        VLib: Activity profiles for regions of interest
        ytest: The test data
        vtest: The test task matrix
        Indices: The indices of the voxels to evaluate 
    return:
        cos: Prediction error
    """
    # Get the task subset indices and corresponding data
    task_subset_indices = list(combination)
    V_subset = VLib[task_subset_indices, :]
    V_subset = center_matrix(V_subset,axis = 0)
    V_subset = normalize_matrix(V_subset,axis = 0)

    y_subset = YLib[:,task_subset_indices, :]
    y_subset = center_matrix(y_subset,axis = 1)
    y_subset = normalize_matrix(y_subset,axis = 1)

    U_hats = et.estimate_Us_projection(y_subset, V_subset)
    U_hat_one_hot = get_U_hat_one_hot(U_hats)
    
    cos,cos_std = get_prediction_error(ytest,vtest,U_hat_one_hot,indices = indices)
    return cos,cos_std


def real_evaluate_dataframe_multiregion(D,
                                         YLib,VLib,
                                         ytest, vtest,
                                         indices = None):
    """ Evaluate the parcellation performance for each combination in the DataFrame D.
    
            Args:
                D: DataFrame containing the combinations to evaluate
                YLib: The data for all tasks
                VLib: Activity profiles for regions of interest
                ytest: The test data
                vtest: The test task matrix
                Indices: The indices of the voxels to evaluate
    return:
        D: DataFrame with the computed prediction error
        """
    D_eval = D.copy()
    D_eval['cos'] = None
    D_eval['cos_std'] = None

    for i in range(len(D)):
        if i % 10 == 0:
            print(f"Processing combination: {i}")
        combination = D_eval['combination'].iloc[i]
        cos,cos_std= real_evaluate_combination_multiregion(combination, YLib,VLib,ytest, vtest, indices = indices)
        D_eval.loc[i, 'cos'] = cos
        D_eval.loc[i, 'cos_std'] = cos_std
    return D_eval


if __name__=='__main__':
    U_hat = np.random.rand(3,10,6000)
    U_hat_one = get_U_hat_one_hot(U_hat)
    pass

