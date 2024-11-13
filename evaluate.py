import numpy as np
import pandas as pd
import torch as pt
import OptimalBattery.simulate as sim
import OptimalBattery.estimate as et
import HierarchBayesParcel.evaluation as hbpev
import OptimalBattery.util as ut
from HierarchBayesParcel.evaluation import calc_test_error,coserr



def U_MSE(U_true, U_pred):
    MSE = []
    # if its only two dimensions then add a dimension
    if len(U_true.shape) == 2:
        U_true = U_true.reshape(1, U_true.shape[0], U_true.shape[1])
        U_pred = U_pred.reshape(1, U_pred.shape[0], U_pred.shape[1])
    elif len(U_true.shape) == 1:
        U_true = U_true.reshape(1, U_true.shape[0])
        U_pred = U_pred.reshape(1, U_pred.shape[0])

    for subject in range(U_true.shape[0]):
        mse = np.mean((U_true[subject] - U_pred[subject])**2)
        MSE.append(mse)
    return np.mean(MSE)

def center_normalize(X,axis=0):
    X = X - X.mean(axis=axis, keepdims=True)
    X = X / np.linalg.norm(X, axis=axis, keepdims=True)
    return X

def get_U_hat_one_hot(U_hat):
    if len(U_hat.shape) == 2:
        U_hat = U_hat.reshape(1, U_hat.shape[0], U_hat.shape[1])
    if type(U_hat) == pt.Tensor:
        U_hat = U_hat.cpu().numpy()

    Us=[]
    for i in range(U_hat.shape[0]):
        max_indices = np.argmax(U_hat[i], axis=0)
        U_hat_one_hot = np.zeros_like(U_hat[i])
        U_hat_one_hot[max_indices, np.arange(U_hat[i].shape[1])] = 1
        Us.append(U_hat_one_hot)
    Us = np.stack(Us)
    return Us


def evaluate_single_simulation(combination,YLib,VLib, U_true,ytest,vtest,parcels_to_evaluate,estimation_method = 'OLS'):
    if type(U_true) == pt.Tensor:
        U_true = U_true.numpy()
    if len(U_true.shape) == 2:
        U_true = U_true.reshape(1, U_true.shape[0], U_true.shape[1])

    # Get the task subset indices and corresponding data
    task_subset_indices = list(combination)
    # task_subset_indices = np.arange(25)
    V_subset = VLib[task_subset_indices,:]
    V_subset = center_normalize(V_subset,axis=0)

    y_subset = YLib[task_subset_indices, :]
    y_subset = center_normalize(y_subset,axis=0)

    if estimation_method == 'OLS':
        U_hat = et.estimate_Us_ols(y_subset, V_subset)
        U_hat_one_hot = get_U_hat_one_hot(U_hat)
    elif estimation_method == 'projection':
        U_hat = et.estimate_Us_projection(y_subset, V_subset)
        U_hat_one_hot = get_U_hat_one_hot(U_hat)

    U_hat_evaluation = U_hat_one_hot[:,parcels_to_evaluate,:]
    U_true_evaluation = U_true[:,parcels_to_evaluate,:]
    mse = U_MSE(U_true_evaluation, U_hat_evaluation)
    # U_hat_evaluation = pt.tensor(U_hat_evaluation,dtype=pt.float32)
    # cos  =hbpev.coserr(ytest,vtest,U_hat_evaluation,adjusted=False,soft_assign=False)
    cos = np.nan # cos.item()
    return mse,cos,U_hat_one_hot


def evaluate_dataframe_simulation(D, YLib,VLib, U_true,ytest,vtest,parcels_to_evaluate,estimation_method = 'OLS'):
    # Create a new column with combinations as tuples to make them hashable
    D['combination_tuple'] = D['combination'].apply(lambda x: tuple(x))
    # Get unique combinations
    unique_combinations = D['combination_tuple'].unique()

    Us = []
    mse_dict = {}
    cos_dict = {}

    # Loop over each unique combination
    for i, comb_tuple in enumerate(unique_combinations):
        if i % 100 == 0:
            print(f"Processing combination: {i}")
        mse,cos,U_hat_one_hot = evaluate_single_simulation(comb_tuple,YLib,VLib, U_true,ytest,vtest,parcels_to_evaluate,estimation_method)
        mse_dict[comb_tuple] = mse
        cos_dict[comb_tuple] = cos
        Us.append(U_hat_one_hot)


    # Map the computed cos_HBP values back to the DataFrame
    D['mse'] = D['combination_tuple'].map(mse_dict)
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