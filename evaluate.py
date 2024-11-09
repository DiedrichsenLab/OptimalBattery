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


def evaluate_combinations_simulation(D, YLib, VLib, U_true, ytest, vtest, parcels_to_evaluate, estimation_method='OLS'):
    # Create a new column with combinations as tuples to make them hashable
    D['combination_tuple'] = D['combination'].apply(lambda x: tuple(x)) 
    # Get unique combinations
    unique_combinations = D['combination_tuple'].unique()
    # Initialize a dictionary to store MSE and cosine error for each unique combination
    mse_key = f"mse_{estimation_method.lower()}"

    U_true = U_true.numpy()
    Us = []

    mse_dict = {}
    cos_dict = {}

    # **Extract voxel indices for the parcel(s) of interest**
    parcel_indices = []
    for parcel in parcels_to_evaluate:
        indices = np.where(U_true[parcel, :] == 1)[0]
        parcel_indices.extend(indices)
    parcel_indices = np.array(parcel_indices)

    # **Extract ytest for the parcel(s) of interest**
    ytest_parcel = ytest[:, parcel_indices]  # Assuming ytest is of shape (tasks, voxels)
    ytest_parcel = ytest_parcel - ytest_parcel.mean(axis=0)
    ytest_parcel = ytest_parcel / np.linalg.norm(ytest_parcel, axis=0)
    ytest_parcel = pt.tensor(ytest_parcel, dtype=pt.float32)
    # Ensure ytest_parcel has shape (1, tasks, voxels)
    if ytest_parcel.dim() == 2:
        ytest_parcel = ytest_parcel.unsqueeze(0)

    # **Prepare V_test for the parcel(s) of interest**
    V_test = vtest  # Assuming vtest corresponds to the parcel(s) of interest
    V_test = V_test - V_test.mean(axis=0)
    V_test = V_test / np.linalg.norm(V_test, axis=0)
    V_test = pt.tensor(V_test, dtype=pt.float32)

    # Loop over each unique combination
    for i, comb_tuple in enumerate(unique_combinations):
        if i % 100 == 0:
            print(f"Processing combination: {i}")
        
        # Get the task subset indices and corresponding data
        task_subset_indices = list(comb_tuple)
        V_subset = VLib[task_subset_indices, :]
        V_subset = V_subset - V_subset.mean(axis=0)
        V_subset = V_subset / np.linalg.norm(V_subset, axis=0)

        y_subset = YLib[task_subset_indices, :]
        y_subset = y_subset - y_subset.mean(axis=0)
        y_subset = y_subset / np.linalg.norm(y_subset, axis=0)
        
        if estimation_method == 'OLS':
            U_hat = et.estimate_Us_ols(y_subset, V_subset, regularize=1e-4)
            max_indices = np.argmax(U_hat, axis=0)
            U_hat_one_hot = np.zeros_like(U_hat)
            U_hat_one_hot[max_indices, np.arange(U_hat.shape[1])] = 1
        elif estimation_method == 'projection':
            U_hat = V_subset.T @ y_subset
            max_indices = np.argmax(U_hat, axis=0)
            U_hat_one_hot = np.zeros_like(U_hat)
            U_hat_one_hot[max_indices, np.arange(U_hat.shape[1])] = 1
        Us.append(U_hat_one_hot)

        # **Compute MSE using only the parcels of interest**
        U_hat_evaluation = U_hat_one_hot[parcels_to_evaluate, :]
        U_true_evaluation = U_true[parcels_to_evaluate, :]
        mse = U_MSE(U_true_evaluation, U_hat_evaluation)

        # **Extract U_hat for the parcel(s) of interest and the voxels of interest**
        U_hat_parcel = U_hat_one_hot[parcels_to_evaluate, :][:, parcel_indices]
        # Reshape and convert to tensor
        U_hat_parcel = U_hat_parcel.reshape(len(parcels_to_evaluate), -1)
        U_hat_parcel = pt.tensor(U_hat_parcel, dtype=pt.float32)
        U_hat_parcel = U_hat_parcel.unsqueeze(0)  #

        # **Compute cosine error using the data for the parcel(s) of interest**
        cos = hbpev.cosine_error(ytest_parcel, V_test, U_hat_parcel, adjusted=False)
        cos = cos.item()

        mse_dict[comb_tuple] = mse
        cos_dict[comb_tuple] = cos
    
    # Map the computed MSE and cosine error values back to the DataFrame
    D[mse_key] = D['combination_tuple'].map(mse_dict)    
    D['cos'] = D['combination_tuple'].map(cos_dict)
    return D, Us



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


def evaluate_single_combination(YLib, VLib, U_true, combination, ytest, vtest, parcels_to_evaluate, estimation_method='OLS'):
    U_true = U_true.numpy()
        
    # Get the task subset indices and corresponding data
    task_subset_indices = list(combination)
    V_subset = VLib[task_subset_indices, :]
    V_subset = V_subset - V_subset.mean(axis=0)
    V_subset = V_subset / np.linalg.norm(V_subset, axis=0)

    y_subset = YLib[task_subset_indices, :]
    y_subset = y_subset - y_subset.mean(axis=0)
    y_subset = y_subset / np.linalg.norm(y_subset, axis=0)

    # Estimate U_hat using the specified method
    if estimation_method == 'OLS':
        U_hat = et.estimate_Us_ols(y_subset, V_subset, regularize=1e-4)
        max_indices = np.argmax(U_hat, axis=0)
        U_hat_one_hot = np.zeros_like(U_hat)
        U_hat_one_hot[max_indices, np.arange(U_hat.shape[1])] = 1
    elif estimation_method == 'projection':
        U_hat = V_subset.T @ y_subset
        max_indices = np.argmax(U_hat, axis=0)
        U_hat_one_hot = np.zeros_like(U_hat)
        U_hat_one_hot[max_indices, np.arange(U_hat.shape[1])] = 1

    # Compute MSE using only the parcels of interest
    U_hat_evaluation = U_hat_one_hot[parcels_to_evaluate, :]
    U_true_evaluation = U_true[parcels_to_evaluate, :]
    mse = U_MSE(U_true_evaluation, U_hat_evaluation)

    # **Extract voxel indices for the parcel(s) of interest**
    parcel_indices = []
    for parcel in parcels_to_evaluate:
        indices = np.where(U_true[parcel, :] == 1)[0]
        parcel_indices.extend(indices)
    parcel_indices = np.array(parcel_indices)

    # **Extract ytest for the parcel(s) of interest**
    ytest_parcel = ytest[:, parcel_indices]
    ytest_parcel = ytest_parcel - ytest_parcel.mean(axis=0)
    ytest_parcel = ytest_parcel / np.linalg.norm(ytest_parcel, axis=0)
    ytest_parcel = pt.tensor(ytest_parcel, dtype=pt.float32)
    if ytest_parcel.dim() == 2:
        ytest_parcel = ytest_parcel.unsqueeze(0)  # Add subject dimension

    # **Prepare V_test for the parcel(s) of interest**
    V_test = vtest
    V_test = V_test - V_test.mean(axis=0)
    V_test = V_test / np.linalg.norm(V_test, axis=0)
    V_test = pt.tensor(V_test, dtype=pt.float32)

    # **Extract U_hat for the parcel(s) of interest and the voxels of interest**
    U_hat_parcel = U_hat_one_hot[parcels_to_evaluate, :][:, parcel_indices]
    U_hat_parcel = U_hat_parcel.reshape(len(parcels_to_evaluate), -1)
    U_hat_parcel = pt.tensor(U_hat_parcel, dtype=pt.float32)
    U_hat_parcel = U_hat_parcel.unsqueeze(0)  # Add subject dimension

    # **Compute cosine error using the data for the parcel(s) of interest**
    cos = hbpev.cosine_error(ytest_parcel, V_test, U_hat_parcel, adjusted=False)
    cos = cos.item()
    
    return mse, cos, U_hat_one_hot


