import numpy as np
import OptimalBattery.estimate as et

def center_data(X,axis =0):
    mean = np.nanmean(X, axis=axis, keepdims=True)
    X = X - mean
    return X

def normalize_data(X,axis = 0):
    norm = np.sqrt(np.nansum(X**2, axis=axis, keepdims=True))
    norm = np.where(norm == 0, 1.0, norm)
    X = X / norm
    return X

def get_U_hat_one_hot(U_hat):
    """Convert the estimated Us to one-hot encoding."""
    if U_hat.ndim == 2:
        U_hat = U_hat[np.newaxis, :, :]

    max_indices = np.argmax(U_hat, axis=1)
    U_hat_one_hot = np.zeros_like(U_hat)
    for subject in range(U_hat.shape[0]):
        U_hat_one_hot[subject, max_indices[subject], np.arange(U_hat.shape[2])] = 1
    return U_hat_one_hot

def sim_percentage_correct(U_true, U_pred):
    """Compute the percentage of correctly classified voxels."""
    correct_voxels = np.sum(U_true * U_pred)
    total_voxels = U_true.shape[2]
    percentage = (correct_voxels / total_voxels) * 100
    return percentage


def sim_prediction_error(ytest,vtest,U_hat):
    if U_hat.ndim == 2:
        U_hat = U_hat[np.newaxis,:,:]
    if ytest.ndim == 2:
        ytest = ytest[np.newaxis,:,:]
    vtest_stand = vtest / np.sum(vtest**2,axis = 0)

    # get norm of data across 2nd dimension
    ytest_norm_2 = np.nansum(ytest**2,axis = 1)
    ytest_norm = np.sqrt(ytest_norm_2)
    ytest_norm_reshaped = ytest_norm[:,np.newaxis,:]

    # normalize data
    ytest_normalized = ytest / ytest_norm_reshaped


    cos_err = np.zeros((U_hat.shape[0],))
    for i in range(U_hat.shape[0]):
        yhat = np.matmul(vtest_stand,U_hat[i])
        cosine_error_vox = 1 - np.nansum(ytest_normalized[i] * yhat,axis = 0)
        cos_err[i] = np.nanmean(cosine_error_vox)
    
    final_cos_err = np.nanmean(cos_err)
    return final_cos_err

def real_prediction_error(ytest,vtest,U_hat,indices = None):
    if U_hat.ndim == 2:
        U_hat = U_hat[np.newaxis,:,:]
    if ytest.ndim == 2:
        ytest = ytest[np.newaxis,:,:]
    vtest_stand = vtest / np.sum(vtest**2,axis = 0)

    # get norm of data across 2nd dimension
    ytest= center_data(ytest,axis = 1)
    ytest = normalize_data(ytest,axis = 1)


    cos_err = np.zeros((U_hat.shape[0],))
    for i in range(U_hat.shape[0]):
        yhat = np.matmul(vtest_stand,U_hat[i])
        cosine_error_vox = 1 - np.nansum(ytest[i][:,indices] * yhat[:,indices],axis = 0)
        cos_err[i] = np.nanmean(cosine_error_vox)
    
    final_cos_err = np.nanmean(cos_err)
    return final_cos_err



def sim_evaluate_combination_multiregion(combination,
                                     Ytrue,Vr,Ur,
                                     n_iter=100,
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
    """
    # Get the task subset indices and corresponding data
    task_subset_indices = list(combination)

    V_subset = Vr[task_subset_indices,:]
    V_subset = center_data(V_subset,axis=0)
    V_subset = normalize_data(V_subset,axis=0)
    y_subset = Ytrue[task_subset_indices,:]
    
    perc = np.zeros((n_iter,))
    cos = np.zeros((n_iter,))
    for i in range(n_iter):
        y = y_subset + np.random.normal(0, sig_e, y_subset.shape)
        y_norm = center_data(y,axis=0)
        y_norm = normalize_data(y_norm,axis=0)

        U_hat = et.estimate_Us_projection(y_norm, V_subset)
        U_hat_one_hot = get_U_hat_one_hot(U_hat)
    
        #eval
        perc[i] = sim_percentage_correct(Ur, U_hat_one_hot)
        cos[i] = sim_prediction_error(ytest,vtest,U_hat_one_hot)

    perc_mean = perc.mean()
    perc_sem = perc.std(ddof=1) / np.sqrt(n_iter) 
    cos_mean = cos.mean()


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
        """
    # Create a new column with combinations as tuples to make them hashable
    D['combination_tuple'] = D['combination'].apply(lambda x: tuple(x)) 
    # Get unique combinations
    unique_combinations = D['combination_tuple'].unique()

    perc_dict= {}
    perc_sem_dict = {}
    cos_dict = {}
    # Loop over each unique combination
    for i, comb_tuple in enumerate(unique_combinations):
        # if i % 10 == 0:
        #     print(f"Processing combination: {i}")
        cos,perc,perc_sem = sim_evaluate_combination_multiregion(comb_tuple,Ytrue,Vr, Ur,sig_e=sig_e,vtest = vtest,ytest = ytest)
        perc_dict[comb_tuple] = perc
        perc_sem_dict[comb_tuple] = perc_sem
        cos_dict[comb_tuple] = cos


    
    # Map the computed cos_HBP values back to the DataFrame
    D['perc'] = D['combination_tuple'].map(perc_dict)
    D['perc_sem'] = D['combination_tuple'].map(perc_sem_dict)
    D['cos'] = D['combination_tuple'].map(cos_dict)
    return D

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
    """
    # Get the task subset indices and corresponding data
    task_subset_indices = list(combination)
    V_subset = VLib[task_subset_indices, :]
    V_subset = center_data(V_subset,axis = 0)
    V_subset = normalize_data(V_subset,axis = 0)

    y_subset = YLib[:,task_subset_indices, :]
    y_subset = center_data(y_subset,axis = 1)
    y_subset = normalize_data(y_subset,axis = 1)

    U_hats = et.estimate_Us_projection(y_subset, V_subset)
    U_hat_one_hot = get_U_hat_one_hot(U_hats)
    
    cos = real_prediction_error(ytest,vtest,U_hat_one_hot,indices = indices)
    return cos


def evaluate_dataframe_real_multiregion(D,
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
        """
    # Create a new column with combinations as tuples to make them hashable
    D['combination_tuple'] = D['combination'].apply(lambda x: tuple(x)) 
    # Get unique combinations
    unique_combinations = D['combination_tuple'].unique()

    cos_dict = {}
    # Loop over each unique combination
    for i, comb_tuple in enumerate(unique_combinations):
        if i % 100 == 0:
            print(f"Processing combination: {i}")
        cos= real_evaluate_combination_multiregion(comb_tuple, YLib,VLib,ytest, vtest, indices = indices)
        cos_dict[comb_tuple] = cos
    
    # Map the computed cos values back to the DataFrame
    D['cos'] = D['combination_tuple'].map(cos_dict)
    return D


