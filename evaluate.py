import numpy as np
import pandas as pd
import torch as pt
import OptimalBattery.simulate as sim
import HierarchBayesParcel.evaluation as ev

def evaluate_combinations_simulation(D, YLib,VLib, ytest, vtest, U_true_region):
    # Create a new column with combinations as tuples to make them hashable
    D['combination_tuple'] = D['combination'].apply(lambda x: tuple(x))
    
    # Get unique combinations
    unique_combinations = D['combination_tuple'].unique()
    
    # Initialize a dictionary to store cos_HBP for each unique combination
    mse_ols_dict = {}
    cos_ols_dict = {}

    ytest = pt.tensor(ytest,dtype=pt.float32)
    vtest = pt.tensor(vtest,dtype=pt.float32)
    U_true_region = U_true_region.numpy()
    
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

        #tensors to np
        y_subset = y_subset.numpy()
        V_subset = V_subset.numpy()
        
        
        U_hat_ols = sim.estimate_Us_ols(y_subset, V_subset)
        mse_ols = sim.U_MSE(U_true_region, U_hat_ols[20:,:])
        
        U_hat_ols = pt.tensor(U_hat_ols,dtype=pt.float32)
        cos_ols = ev.coserr(ytest,vtest,U_hat_ols[20:,:]).mean().cpu().numpy()
        cos_ols = cos_ols.item()
        
        # Store the result in the dictionary
        mse_ols_dict[comb_tuple] = mse_ols
        cos_ols_dict[comb_tuple] = cos_ols
    
    # Map the computed cos_HBP values back to the DataFrame
    D['mse_ols'] = D['combination_tuple'].map(mse_ols_dict)
    D['cos_ols'] = D['combination_tuple'].map(cos_ols_dict)
    
    return D


def evaluate_single_combination(YLib,VLib, ytest, vtest, U_true_region,combination):

    ytest = pt.tensor(ytest,dtype=pt.float32)
    vtest = pt.tensor(vtest,dtype=pt.float32)
    U_true_region = U_true_region.numpy()
        
    # Get the task subset indices and corresponding data
    task_subset_indices = list(combination)
    # task_subset_indices = np.arange(25)
    V_subset = VLib[task_subset_indices,:]
    V_subset = V_subset - V_subset.mean(axis=0)

    y_subset = YLib[task_subset_indices, :]
    y_subset = y_subset - y_subset.mean(axis=0)

    #tensors to np
    y_subset = y_subset.numpy()
    V_subset = V_subset.numpy()
    
    
    U_hat_ols = sim.estimate_Us_ols(y_subset, V_subset)
    mse_ols = sim.U_MSE(U_true_region, U_hat_ols[20:,:])
    print(mse_ols)

    
    return 