"""
Module for function used in the simulation of the task-battery construction problem.
Author: Bassel Arafat
"""
import numpy as np
import matplotlib.pyplot as plt
import OptimalBattery.util as ut
import OptimalBattery.estimate as et
import OptimalBattery.evaluate as ev
import OptimalBattery.construct as ct
import torch as pt

device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
def make_U_spatial(grid, centroids, K_main, K_subparcels): # ugly but works
    """
    Computes parcel labels for all pixels based on distances to centroids and divides them into subparcels.
    
    """
    # Compute positions of all pixels
    width, height = grid.width, grid.height
    X_coords, Y_coords = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    X_coords = X_coords.flatten()
    Y_coords = Y_coords.flatten()
    positions = np.column_stack((X_coords, Y_coords))
    
    # Compute distances from each pixel to each centroid
    D = np.zeros((grid.P, K_main))
    for k, (cx, cy) in enumerate(centroids):
        D[:, k] = np.sqrt((X_coords - cx)**2 + (Y_coords - cy)**2)
    
    # Initialize the parcel labels and define the size of each parcel
    parcel_labels = np.full(grid.P, -1, dtype=int)
    unassigned_nodes = set(range(grid.P))
    desired_size = grid.P // K_main
    
    for k in range(K_main - 1):
        unassigned_nodes_list = list(unassigned_nodes)
        distances = D[unassigned_nodes_list, k]
        sorted_indices = np.argsort(distances)
        nodes_to_assign = np.array(unassigned_nodes_list)[sorted_indices[:desired_size]]
        parcel_labels[nodes_to_assign] = k
        unassigned_nodes -= set(nodes_to_assign)
    
    # Assign the remaining pixels to the last parcel
    parcel_labels[list(unassigned_nodes)] = K_main - 1
    
    # Initialize new parcel labels
    new_parcel_labels = np.full(grid.P, -1, dtype=int)
    
    for k in range(K_main):  # For each main parcel
        nodes_in_parcel = np.where(parcel_labels == k)[0]
        # Split nodes_in_parcel into K_subparcels of equal size
        subparcel_nodes = np.array_split(nodes_in_parcel, K_subparcels)
        for sub_k, nodes in enumerate(subparcel_nodes):
            new_parcel_label = k * K_subparcels + sub_k
            new_parcel_labels[nodes] = new_parcel_label
    
    # Convert new parcel labels to a matrix U_true
    K_total = K_main * K_subparcels
    U_true = np.zeros((K_total, grid.P))
    for k in range(K_total):
        U_true[k, new_parcel_labels == k] = 1
    
    return U_true

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


def get_dice_coefficient(U_true, U_pred):
    """
    Compute Dice coefficient
    
    Args:
        U_true (Tensor): True Us)
        U_pred (Tensor): Estimated Us
    Returns:
        mean_dice (float): Average Dice across all parcels
    """

    intersection = (U_true * U_pred).sum(dim=1) 
    size_true = U_true.sum(dim=1)               
    size_pred = U_pred.sum(dim=1)          

    dice_scores = (2 * intersection ) / (size_true + size_pred )  
    mean_dice = dice_scores.mean().item()
    return mean_dice


def evluate_dataframe_simulation(D, YLib, VLib, n_iter, noise, U_true, method, noise_method='fixed',max_n_task = 16):
    """ Evaluate the parcellation performance for each combination in the DataFrame D.
    
    Args:
        D: DataFrame containing the combinations to evaluate
        YLib: The training data (subjects, conditions x repetitions, voxels)
        VLib: Activity profiles for training data (tasks, parcels)
        n_iter: Number of iterations to run the simulation for each task battery
        noise: The base noise level to add to battery data
        U_true: The true parcellation matrix (parcels, voxels)
        method: Method used for parcellation estimation
        noise_method: If 'weighted', noise increases with the number of tasks

    Returns:
        D: DataFrame with the computed percentage of correct voxels classified
    """
    D.loc[:, 'percent_correct'] = None


    for i in range(len(D)):
        if i % 1000 == 0:
            print(f"Processing combination: {i}")

        combination = list(D['combination'].iloc[i])
        n_task = len(combination)

        # Apply weighted noise if specified
        if noise_method == 'weighted':
            weighted_noise = np.sqrt(noise * (n_task / max_n_task))
        elif noise_method == 'fixed':
            weighted_noise = noise  # Default noise level

        # Normalize VLib subset
        VLib_subset = VLib[combination, :]
        VLib_subset = ut.center_matrix(VLib_subset, axis=0)
        VLib_subset = ut.normalize_matrix(VLib_subset, axis=0)

        perc_correct_li = []
        for j in range(n_iter):
            # Add noise based on the weighted or fixed method
            YLib_subset = YLib[:, combination, :]
            YLib_subset = YLib_subset + pt.normal(0, weighted_noise, YLib_subset.shape, device=device)
            YLib_subset = ut.center_matrix(YLib_subset, axis=1)
            YLib_subset = ut.normalize_matrix(YLib_subset, axis=1)

            # Build the parcellation
            U_hats = et.estimate_Us(YLib_subset, VLib_subset, method=method, hard=True)

            # Evaluate the parcellation
            perc_correct = get_percentage_correct(U_true, U_hats)
            perc_correct_li.append(perc_correct.item())

        # Store the averaged percentage correct
        D.loc[i, 'percent_correct'] = np.mean(perc_correct_li)

    return D



def sim_connectivity(num_task_lib = 32,
                     n_parcels = 5,
                     n_voxels_y = 100,
                     battery_sizes = [3,4,6,8,10,14,18,24,28], 
                     n_batteries = 50000,
                     seed = None): 
    """ Single simulation for the connectivity estimation. 
    """

    # Make new task battery 
    if seed is not None:
        rng= np.random.default_rng(seed=seed)
    else: 
        rng= np.random.default_rng()
    V_lib = rng.normal(0,1,(num_task_lib, n_parcels))
    V_lib = V_lib - V_lib.mean(axis=0,keepdims=True) 
    G_lib = V_lib @ V_lib.T

    W_true = rng.normal(0,1,(n_parcels, n_voxels_y))

 
    metrics = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc']
    max_n_task = max(battery_sizes)
    
    # Number of batteries to compile for each size 
    for n_task in battery_sizes:
        print(f"Processing battery size: {n_task}")

        # Generate possible battery combinations for current battery size and evaluate each battery
        D = ct.build_combinations(G_lib, strategy='random',n_batteries=n_batteries,n_tasks=n_task,replacement=False)

        for metric in metrics:
            
            D_best = ct.choose_battery(D,metric) 
            top_comb = D_best['combination'].values[0]         
            xtrain = V_lib[top_comb,:]
            xtrain = ut.center_matrix(xtrain,axis=0)

            # 
            ytrain = xtrain @ W_true + rng.normal(0,noise_sd,(num_task_lib,n_voxels_y))
            ytrain = ut.center_matrix(ytrain,axis=0)
            conn_model = getattr(model, 'L2regression')(1000)

            # Fit model, correlate with original weights
            conn_model.fit(xtrain, ytrain)
            coef= conn_model.coef_
            coef_flat = coef.flatten()
            W_flat = W.flatten()
            corrcoef_matrix = np.corrcoef(coef_flat, W_flat)
            pearson_corr = corrcoef_matrix[0, 1]

            D_ev = pd.DataFrame()
            D_ev['iteration'] = [i]
            D_ev['n_task'] = [n_task]
            D_ev['metric'] = [metric]
            D_ev['correlation'] = pearson_corr
            results_df = pd.concat([results_df,D_ev],axis=0)


if __name__=='__main__':
    test_produce_V()
    pass