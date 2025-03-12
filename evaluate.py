"""
Module for evaluating the performance of task batteries for both simulations and real data.
Author: Bassel Arafat
"""
import torch as pt
import OptimalBattery.estimate as et
import OptimalBattery.util as ut
import numpy as np
import construct as ct

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


def evluate_dataframe(D,condition_df,
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
    

if __name__=='__main__':
    U_hat = pt.random.rand(3,10,6000)
    U_hat_one = get_U_hat_one_hot(U_hat)
    pass

