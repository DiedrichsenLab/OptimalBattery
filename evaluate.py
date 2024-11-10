import numpy as np
import pandas as pd
import torch as pt
import OptimalBattery.simulate as sim
import OptimalBattery.estimate as et
import HierarchBayesParcel.evaluation as hbpev


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


def evaluate_combinations_simulation(D, YLib,VLib, U_true,ytest,vtest,parcels_to_evaluate,estimation_method = 'OLS'):
    # Create a new column with combinations as tuples to make them hashable
    D['combination_tuple'] = D['combination'].apply(lambda x: tuple(x)) 
    # Get unique combinations
    unique_combinations = D['combination_tuple'].unique()
    # Initialize a dictionary to store cos_HBP for each unique combination
    mse_key = f"mse_{estimation_method.lower()}"

    U_true = U_true.numpy()
    Us = []

    mse_dict = {}
    cos_dict = {}

    # Loop over each unique combination
    for i, comb_tuple in enumerate(unique_combinations):
        if i % 100 == 0:
            print(f"Processing combination: {i}")
        
        # Get the task subset indices and corresponding data
        task_subset_indices = list(comb_tuple)
        V_subset = VLib[task_subset_indices,:]
        V_subset = V_subset - V_subset.mean(axis=0)
        V_subset = V_subset / np.linalg.norm(V_subset, axis=0)


        y_subset = YLib[task_subset_indices, :]
        y_subset = y_subset - y_subset.mean(axis=0)
        y_subset = y_subset / np.linalg.norm(y_subset, axis=0)
        
        if estimation_method == 'OLS':
            U_hat = et.estimate_Us_ols(y_subset, V_subset,regularize=1e-4)
            max_indices = np.argmax(U_hat, axis=0)
            U_hat_one_hot = np.zeros_like(U_hat)
            U_hat_one_hot[max_indices, np.arange(U_hat.shape[1])] = 1
        elif estimation_method == 'projection':
            U_hat = V_subset.T @ y_subset
            max_indices = np.argmax(U_hat, axis=0)
            U_hat_one_hot = np.zeros_like(U_hat)
            U_hat_one_hot[max_indices, np.arange(U_hat.shape[1])] = 1
        Us.append(U_hat_one_hot)

        U_hat_evaluation = U_hat_one_hot[parcels_to_evaluate,:]
        U_true_evaluation = U_true[parcels_to_evaluate,:]
        mse = U_MSE(U_true_evaluation, U_hat_evaluation)

        U_hat_evaluation = pt.tensor(U_hat_evaluation,dtype=pt.float32)
        if len(U_hat_evaluation.shape) == 1:
            U_hat_evaluation = U_hat_evaluation.reshape(1,U_hat_evaluation.shape[0])
        cos  =hbpev.coserr(ytest,vtest,U_hat_evaluation,adjusted=False)
        cos = cos.item()

        mse_dict[comb_tuple] = mse
        cos_dict[comb_tuple] = cos
    
    # Map the computed cos_HBP values back to the DataFrame
    D[mse_key] = D['combination_tuple'].map(mse_dict)    
    D['cos'] = D['combination_tuple'].map(cos_dict)
    return D,Us

def evaluate_single_combination(YLib,VLib, U_true,combination,ytest,vtest,parcels_to_evaluate,estimation_method = 'OLS'):
    U_true = U_true.numpy()
        
    # Get the task subset indices and corresponding data
    task_subset_indices = list(combination)
    # task_subset_indices = np.arange(25)
    V_subset = VLib[task_subset_indices,:]
    V_subset = V_subset - V_subset.mean(axis=0)
    V_subset = V_subset / np.linalg.norm(V_subset, axis=0)

    y_subset = YLib[task_subset_indices, :]
    y_subset = y_subset - y_subset.mean(axis=0)
    y_subset = y_subset / np.linalg.norm(y_subset, axis=0)

    if estimation_method == 'OLS':
        U_hat = et.estimate_Us_ols(y_subset, V_subset,regularize=1e-4)
        max_indices = np.argmax(U_hat, axis=0)
        U_hat_one_hot = np.zeros_like(U_hat)
        U_hat_one_hot[max_indices, np.arange(U_hat.shape[1])] = 1
    elif estimation_method == 'projection':
        U_hat = V_subset.T @ y_subset
        max_indices = np.argmax(U_hat, axis=0)
        U_hat_one_hot = np.zeros_like(U_hat)
        U_hat_one_hot[max_indices, np.arange(U_hat.shape[1])] = 1

    U_hat_evaluation = U_hat_one_hot[parcels_to_evaluate,:]
    U_true_evaluation = U_true[parcels_to_evaluate,:]
    mse = U_MSE(U_true_evaluation, U_hat_evaluation)

    U_hat_evaluation = pt.tensor(U_hat_evaluation,dtype=pt.float32)
    if len(U_hat_evaluation.shape) == 1:
        U_hat_evaluation = U_hat_evaluation.reshape(1,U_hat_evaluation.shape[0])
    cos  =hbpev.coserr(ytest,vtest,U_hat_evaluation,adjusted=False)
    cos = cos.item()
    
    return mse,cos,U_hat_one_hot


