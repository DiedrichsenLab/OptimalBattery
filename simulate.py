"""
Module for function used in the simulation of the task-battery construction problem.
Author: Bassel Arafat
"""
import numpy as np
import matplotlib.pyplot as plt
import OptimalBattery.util as ut
import OptimalBattery.estimate as et
import OptimalBattery.evaluate as ev
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
    D['percent_correct'] = None

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


if __name__=='__main__':
    test_produce_V()
    pass