import numpy as np
import torch as pt
import OptimalBattery.estimate as et
import HierarchBayesParcel.evaluation as hbpev
import OptimalBattery.util as ut
from HierarchBayesParcel.evaluation import calc_test_error


def center_normalize(X,axis=0):
    """Center and normalize the data along the specified axis."""
    X = X - X.mean(axis=axis, keepdims=True)
    X = X / np.linalg.norm(X, axis=axis, keepdims=True)
    return X

def get_U_hat_one_hot(U_hat):
    """Convert the estimated Us to one-hot encoding."""
    max_indices = np.argmax(U_hat, axis=0)
    U_hat_one_hot = np.zeros_like(U_hat)
    U_hat_one_hot[max_indices, np.arange(U_hat.shape[1])] = 1
    return U_hat_one_hot

def percentage_correct_parcellation(U_true, U_pred):
    """Compute the percentage of correctly classified voxels."""
    correct_voxels = np.sum(U_true * U_pred)
    total_voxels = U_true.shape[1]
    percentage = (correct_voxels / total_voxels) * 100
    return percentage


def percentage_correct_localization(U_true, U_pred):
    hits = np.sum(U_true * U_pred)
    false_positives = np.sum(U_pred * (1 - U_true))
    percentage = (hits / (hits + false_positives)) * 100
    if np.isnan(percentage):
        percentage = 0
    return percentage

def prediction_error(ytest,vtest,U_hat):
    yhat = np.matmul(vtest,U_hat)
    cosine_error_vox = 1 - np.sum(ytest * yhat,axis = 0)
    cos_err = np.mean(cosine_error_vox)
    return cos_err

def evaluate_single_simulation_multi(combination,
                                     Ytrue,Vr,Ur,
                                     n_iter=100,
                                     sig_e=0.04):
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
    V_subset = center_normalize(V_subset,axis=0)
    y_subset = Ytrue[task_subset_indices,:]
    
    perc = np.zeros((n_iter,))
    for i in range(n_iter):
        y = y_subset + np.random.normal(0, sig_e, y_subset.shape)
        y_norm = center_normalize(y,axis=0)

        U_hat = et.estimate_Us_projection(y_norm, V_subset)
        U_hat_one_hot = get_U_hat_one_hot(U_hat)
    
        #eval
        perc[i] = percentage_correct_parcellation(Ur, U_hat_one_hot)
    return perc.mean()


def evaluate_dataframe_simulation_multi(D, Ytrue,Vr, Ur):
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
    # Loop over each unique combination
    for i, comb_tuple in enumerate(unique_combinations):
        if i % 1000 == 0:
            print(f"Processing combination: {i}")
        perc = evaluate_single_simulation_multi(comb_tuple,Ytrue,Vr, Ur,sig_e=1)
        perc_dict[comb_tuple] = perc

    
    # Map the computed cos_HBP values back to the DataFrame
    D['perc'] = D['combination_tuple'].map(perc_dict)    
    return D

def evaluate_single_simulation_localization(combination,YLib,VLib, U_true,estimation_method = 'OLS',parcel_to_evaluate = None):
    # Get the task subset indices and corresponding data
    task_subset_indices = list(combination)

    V_subset = VLib[task_subset_indices,:]
    V_subset = center_normalize(V_subset,axis=0)

    y_subset = YLib[task_subset_indices, :]
    y_subset = y_subset + np.random.normal(0, 0.04, y_subset.shape)
    y_subset = center_normalize(y_subset,axis=0)

    # test task indices should be 20 to 24 if parcel index 4 for exmaple
    task_start = parcel_to_evaluate * 5
    task_end = task_start + 5
    ytest = YLib[task_start:task_end, :]
    ytest = YLib + np.random.normal(0, 0.04, YLib.shape)
    ytest = center_normalize(ytest,axis=0)
    vtest = VLib
    vtest = VLib[task_start:task_end,parcel_to_evaluate]
    vtest = center_normalize(vtest,axis=0)

    if estimation_method == 'OLS':
        U_hat = et.estimate_Us_ols(y_subset, V_subset)
        U_hat_one_hot = get_U_hat_one_hot(U_hat)
    elif estimation_method == 'projection':
        U_hat = et.estimate_Us_projection(y_subset, V_subset)
        U_hat_one_hot = get_U_hat_one_hot(U_hat)
    
    #eval
    U_true_eval = U_true[parcel_to_evaluate,:].reshape(1,-1)
    U_pred_eval = U_hat_one_hot[parcel_to_evaluate,:].reshape(1,-1)
    perc = percentage_correct_localization(U_true=U_true_eval, U_pred=U_pred_eval)
    cos  = prediction_error(ytest,vtest,U_hat_one_hot)
    
    return perc,cos,U_hat_one_hot

def evaluate_dataframe_simulation_localization(D, YLib,VLib, U_true,estimation_method = 'OLS',parcel_to_evaluate = None):
    # Create a new column with combinations as tuples to make them hashable
    D['combination_tuple'] = D['combination'].apply(lambda x: tuple(x)) 
    # Get unique combinations
    unique_combinations = D['combination_tuple'].unique()

    Us = []
    perc_correct= {}
    cos_dict = {}

    # Loop over each unique combination
    for i, comb_tuple in enumerate(unique_combinations):
        if i % 1000 == 0:
            print(f"Processing combination: {i}")
        perc,cos,U_hat_one_hot = evaluate_single_simulation_localization(comb_tuple,YLib,VLib, U_true,estimation_method,parcel_to_evaluate=parcel_to_evaluate)
        perc_correct[comb_tuple] = perc
        cos_dict[comb_tuple] = cos
        Us.append(U_hat_one_hot)

    
    # Map the computed cos_HBP values back to the DataFrame
    D['perc'] = D['combination_tuple'].map(perc_correct)    
    D['cos'] = D['combination_tuple'].map(cos_dict)
    return D,Us



def evaluate_single_real(combination, YLib,VLib,info,ytest, vtest,M_test,ar_model,parcels_to_evaluate,estimation_method = 'hbp'):
    ytest = pt.tensor(ytest,dtype=pt.float32)
    vtest = pt.tensor(vtest,dtype=pt.float32)

    # Get the task subset indices and corresponding data
    task_subset_indices = list(combination)

    if estimation_method == 'projection':
        V_subset = VLib[task_subset_indices, :]
        V_subset = center_normalize(V_subset,axis=0)

        y_subset_run_1 = YLib[:, task_subset_indices, :]
        n_tasks = VLib.shape[0]
        run_2_indices = [(i + n_tasks) for i in task_subset_indices]
        y_subset_run_2 = YLib[:, run_2_indices, :]

        # y subset is the mean of the two runs
        y_subset = (y_subset_run_1 + y_subset_run_2) / 2
        y_subset = center_normalize(y_subset,axis=1)
        
        U_hats = []
        for i in range(y_subset.shape[0]):
            U_hat = et.estimate_Us_projection(y_subset[i], V_subset)
            U_hats.append(U_hat)
        U_hats = np.stack(U_hats)
        U_hat_one_hot = get_U_hat_one_hot(U_hats)
        U_hat_evaluation = U_hat_one_hot[:,parcels_to_evaluate,:]
        U_hat_evaluation = pt.tensor(U_hat_evaluation,dtype=pt.float32)

        cos = hbpev.coserr(ytest,vtest,U_hat_evaluation,adjusted=True)
        cos = cos.mean().item()

    elif estimation_method == 'hbp':
        # leverage repeats for HBP
        HBP_data,HBP_cond_vec,HBP_part_vec = ut.make_dataset(YLib,info,task_subset_indices,n_repeats=2)
        HBP_data = center_normalize(HBP_data,axis=1)

        U_hat_HBP = et.estiamte_HBP_U(HBP_data, HBP_cond_vec, HBP_part_vec,ar_model)
        U_hat_one_hot = get_U_hat_one_hot(U_hat_HBP)

        U_hat_HBP_eval = U_hat_one_hot[:,parcels_to_evaluate,:]
        if type(U_hat_HBP_eval) == np.ndarray:
            U_hat_HBP_eval = pt.tensor(U_hat_HBP_eval,dtype=pt.float32)
        if U_hat_HBP_eval.ndim == 2:
            U_hat_HBP_eval = U_hat_HBP_eval.reshape(-1,1,U_hat_HBP_eval.shape[1])
        U_hat_HBP_eval = [U_hat_HBP_eval]

        # Compute cos
        cos = calc_test_error(M=M_test, tdata=ytest, U_hats=U_hat_HBP_eval, coserr_type = 'expected',fit_emission='use_Uhats').mean()

    
    return cos,U_hat_one_hot


def evaluate_dataframe_real(D, YLib,VLib,info,ytest, vtest,M_test,ar_model,parcels_to_evaluate,estimation_method = 'hbp'):
    # Create a new column with combinations as tuples to make them hashable
    D['combination_tuple'] = D['combination'].apply(lambda x: tuple(x)) 
    # Get unique combinations
    unique_combinations = D['combination_tuple'].unique()

    Us = []
    cos_dict = {}

    # Loop over each unique combination
    for i, comb_tuple in enumerate(unique_combinations):
        if i % 100 == 0:
            print(f"Processing combination: {i}")
        cos,U_hat_one_hot = evaluate_single_real(comb_tuple, YLib,VLib,info,ytest, vtest,M_test,ar_model,parcels_to_evaluate,estimation_method = estimation_method)
        cos_dict[comb_tuple] = cos
        Us.append(U_hat_one_hot)

    
    # Map the computed cos values back to the DataFrame
    D['cos'] = D['combination_tuple'].map(cos_dict)
    return D,Us