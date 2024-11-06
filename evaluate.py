import numpy as np
import pandas as pd
import torch as pt
import OptimalBattery.simulate as sim
import HierarchBayesParcel.evaluation as ev

def evaluate_combinations_simulation(D, YLib,VLib, U_true,parcels_to_evaluate):
    # Create a new column with combinations as tuples to make them hashable
    D['combination_tuple'] = D['combination'].apply(lambda x: tuple(x)) 
    # Get unique combinations
    unique_combinations = D['combination_tuple'].unique()
    # Initialize a dictionary to store cos_HBP for each unique combination
    mse_ols_dict = {}

    U_true = U_true.numpy()
    Us = []

    # Loop over each unique combination
    for i, comb_tuple in enumerate(unique_combinations):
        if i % 100 == 0:
            print(f"Processing combination: {i}")
        
        # Get the task subset indices and corresponding data
        task_subset_indices = list(comb_tuple)
        V_subset = VLib[task_subset_indices,:]
        V_subset = V_subset - V_subset.mean(axis=0)

        y_subset = YLib[task_subset_indices, :]
        y_subset = y_subset - y_subset.mean(axis=0)
        
        U_hat_ols = sim.estimate_Us_ols(y_subset, V_subset,regularize=1e-4)
        U_hat_ols = U_hat_ols.reshape(5, 5, 900).mean(axis=1)
        Us.append(U_hat_ols)

        U_hat_ols_evaluation = U_hat_ols[parcels_to_evaluate,:]
        U_true_evaluation = U_true[parcels_to_evaluate,:]
        
        mse_ols = sim.U_MSE(U_true_evaluation, U_hat_ols_evaluation)
        mse_ols_dict[comb_tuple] = mse_ols
    
    # Map the computed cos_HBP values back to the DataFrame
    D['mse_ols'] = D['combination_tuple'].map(mse_ols_dict)    
    return D,Us


def evaluate_single_combination(YLib,VLib, U_true,combination,parcels_to_evaluate):
    U_true = U_true.numpy()
        
    # Get the task subset indices and corresponding data
    task_subset_indices = list(combination)
    # task_subset_indices = np.arange(25)
    V_subset = VLib[task_subset_indices,:]
    V_subset = V_subset - V_subset.mean(axis=0)

    y_subset = YLib[task_subset_indices, :]
    y_subset = y_subset - y_subset.mean(axis=0)

   

    U_hat_ols = sim.estimate_Us_ols(y_subset, V_subset,regularize=1e-4)
    U_hat_ols = U_hat_ols.reshape(5, 5, 900).mean(axis=1)
    U_hat_ols_evaluation = U_hat_ols[parcels_to_evaluate,:]
    U_true_evaluation = U_true[parcels_to_evaluate,:]
    
    mse_ols = sim.U_MSE(U_true_evaluation, U_hat_ols_evaluation)
    print(mse_ols)

    
    return mse_ols,U_hat_ols