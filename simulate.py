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
import cortico_cereb_connectivity.model as model
import pandas as pd


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
    # if its two dimensional, add a dimension
    if len(U_true.shape) == 2:
        U_true = U_true.unsqueeze(0)
    if len(U_pred.shape) == 2:
        U_pred = U_pred.unsqueeze(0)

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
    if len(U_true.shape) == 2:
        U_true = U_true.unsqueeze(0)
    if len(U_pred.shape) == 2:
        U_pred = U_pred.unsqueeze(0)

    intersection = (U_true * U_pred).sum(dim=2)
    size_true = U_true.sum(dim=2)
    size_pred = U_pred.sum(dim=2)

    dice_scores = (2 * intersection ) / (size_true + size_pred )
    mean_dice = dice_scores.mean().item()
    return mean_dice

def get_weighted_noise_std(n_task, max_n_task, noise):
    """Compute the noise level based on the number of tasks in the battery.

    Args:
        n_task: Number of tasks in the battery
        max_n_task: Maximum battery size
        noise: Base noise std

    Returns:
        weighted_noise: Noise std based on the number of tasks
    """
    return noise * np.sqrt((n_task / max_n_task))

def find_single_contrast(Vs, regionA, regionB):
    """ Find the task that maximizes the difference between regionA and regionB """
    difference = Vs[:, regionA -1] - Vs[:, regionB-1]
    sorted_idx = pt.argsort(difference)

    min_idx = sorted_idx[0].item()
    max_idx = sorted_idx[-1].item()

    return [max_idx, min_idx]

def make_thresholded_contrast(task1, task2, threshold):
    """gets the contrast between two tasks and thresholds it"""
    contrast_data = task1 - task2
    thresholded_data = pt.zeros_like(contrast_data)
    percentile = pt.quantile(contrast_data, threshold)
    thresholded_data[contrast_data >= percentile] = 1

    # make one hot
    thresholded_data = pt.nn.functional.one_hot(thresholded_data.long(), num_classes=2).T
    return thresholded_data

def sim_single_contrast(num_task_lib = 100,
                        n_parcels = 5,
                        U_true = None,
                        base_noise = 5,
                        max_battery_size = 28,
                        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5],
                        U_true_collapsed = None,
                        seed = None):
    """ Single simulation for the single contrast evaluation."""

     # Make new task battery
    if seed is not None:
        rng= np.random.default_rng(seed=seed)
    else:
        rng= np.random.default_rng()
    V_lib = rng.normal(0,1,(num_task_lib, n_parcels))
    V_lib = V_lib - V_lib.mean(axis=0,keepdims=True)
    V_lib = pt.tensor(V_lib, device=device, dtype=pt.float64)

    # get the single contrast
    max_idx, min_idx = find_single_contrast(V_lib, 5, 4)
    combination = [max_idx, min_idx]

    # get the V localizer
    V_localizer = V_lib[combination,:]
    V_localizer = ut.center_matrix(V_localizer,axis=0)
    V_localizer = ut.normalize_matrix(V_localizer,axis=0)

    # get the data for the parcellation estimation and add noise
    Y_localizer = V_localizer @ U_true
    weighted_noise_std = get_weighted_noise_std(2, max_battery_size, base_noise)
    noise = rng.normal(0,weighted_noise_std,Y_localizer.shape)
    noise = pt.tensor(noise, dtype=pt.float64, device=Y_localizer.device)
    Y_localizer = Y_localizer + noise
    # is cenering and normalizing necessary?
    # Y_localizer = ut.center_matrix(Y_localizer,axis=0)
    # Y_localizer = ut.normalize_matrix(Y_localizer,axis=0)

    results_df = pd.DataFrame()
    for threshold in thresholds:
        # get the thresholded contrast
        thresholded_contrast = make_thresholded_contrast(Y_localizer[0,:], Y_localizer[1,:], threshold)

        # Evaluate the contrast
        accuracy = get_dice_coefficient(U_true_collapsed, thresholded_contrast)

        D_ev = pd.DataFrame()
        D_ev['threshold'] = [threshold]
        D_ev['accuracy'] = accuracy
        results_df = pd.concat([results_df,D_ev],axis=0)

    return results_df


def sim_parcellation(num_task_lib = 100,
                     n_parcels = 5,
                     U_true = None,
                     battery_sizes = [3,4,6,8,10,14,18,24,28],
                     n_batteries = 50000,
                     base_noise = 2,
                     collapsed_U_true = None,
                     seed = None):
    # Make new task battery
    if seed is not None:
        rng= np.random.default_rng(seed=seed)
    else:
        rng= np.random.default_rng()
    V_lib = rng.normal(0,1,(num_task_lib, n_parcels))
    V_lib = V_lib - V_lib.mean(axis=0,keepdims=True)
    G_lib = V_lib @ V_lib.T
    # ensure tensor
    V_lib = pt.tensor(V_lib, device=device, dtype=pt.float64)

    # constants
    metrics = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc']
    max_battery_size = max(battery_sizes)

    results_df =pd.DataFrame()
    for n_task in battery_sizes:
        print(f"Processing battery size: {n_task}")
        # Generate possible battery combinations for current battery size and calculate eigenmetrics
        D = ct.build_combinations(G_lib=G_lib, strategy='random',n_batteries=n_batteries,n_tasks=n_task,replacement=False,rest_idx=None,seed=seed)
        for metric in metrics:
            # Find the best battery for the metric
            D_best = ct.choose_combination(D,metric)
            top_comb = D_best['combination'].values[0]

            # get the V battery
            V_battery = V_lib[top_comb,:]
            V_battery = ut.center_matrix(V_battery,axis=0)
            V_battery = ut.normalize_matrix(V_battery,axis=0)

            # get the data for the parcellation estimation and add noise
            Y_battery = V_battery @ U_true
            weighted_noise_std = get_weighted_noise_std(n_task, max_battery_size, base_noise)
            noise = rng.normal(0,weighted_noise_std,Y_battery.shape)
            noise = pt.tensor(noise, dtype=pt.float64, device=Y_battery.device)
            Y_battery = Y_battery + noise
            Y_battery = ut.center_matrix(Y_battery,axis=0)
            Y_battery = ut.normalize_matrix(Y_battery,axis=0)


            # Build the parcellation
            U_hats = et.estimate_Us(Y_battery, V_battery, method='correlation', hard=True)

            # This is for the single region analysis (optional argument to collapsee the parcellation into two regions)
            if collapsed_U_true is not None:
                # get the first 4 parcels and sum them
                everything_else = U_hats[:, :4, :].sum(dim=1, keepdim=True)
                # get the last parcel (target parcel)
                parcel_of_interest = U_hats[:, 4:, :]
                U_hats = pt.cat([everything_else, parcel_of_interest], dim=1)


            # Evaluate the parcellation
            if collapsed_U_true is not None:
                accuracy = get_dice_coefficient(collapsed_U_true, U_hats)
            else:
                accuracy = get_dice_coefficient(U_true, U_hats)

            D_ev = pd.DataFrame()
            D_ev['n_task'] = [n_task]
            D_ev['metric'] = [metric]
            D_ev['accuracy'] = accuracy
            results_df = pd.concat([results_df,D_ev],axis=0)

    return results_df

def sim_connectivity(num_task_lib = 100,
                     n_parcels = 5,
                     n_voxels_y = 100,
                     n_sim = 50,
                     battery_sizes = [3,4,6,8,10,14,18,24,28],
                     n_batteries = 100,
                     base_noise = 5,
                     ridge_alpha = 0.1,
                     seed = None):
    """ Single simulation for the connectivity estimation.
    """

    # Make new task battery
    if seed is not None:
        rng= np.random.default_rng(seed=seed)
    else:
        rng= np.random.default_rng()

    results_df = pd.DataFrame()

    for n_task in battery_sizes:
        print(f"Processing battery size: {n_task}")

        for n in range(n_sim):
            V_lib = rng.normal(0,1,(num_task_lib, n_parcels))
            V_lib = V_lib - V_lib.mean(axis=0,keepdims=True)
            G_lib = V_lib @ V_lib.T

            W_true = rng.normal(0,1,(n_parcels, n_voxels_y))

            metrics = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc']
            max_n_task = max(battery_sizes)


            # Generate possible battery combinations for current battery size and calculate eigenmetrics
            D = ct.build_combinations(G_lib, strategy='random',n_batteries=n_batteries,n_tasks=n_task,replacement=False)

            for metric in metrics:
                # Find the best battery for the metric
                D_best = ct.choose_combination(D,metric)
                top_comb = D_best['combination'].values[0]

                # get the x for the connectivity estimation
                data_x = V_lib[top_comb,:]
                data_x = ut.center_matrix(data_x,axis=0)

                # get the y for the connectivity estimation (add weighted noise)
                weighted_noise_std = get_weighted_noise_std(n_task, max_n_task, base_noise)
                data_y = data_x @ W_true
                data_y = data_y + rng.normal(0,weighted_noise_std,data_y.shape)
                data_y = ut.center_matrix(data_y,axis=0)

                # fit the model
                conn_model = getattr(model, 'L2regression')(ridge_alpha)
                conn_model.fit(data_x, data_y)

                # get the estimated W and correlate with W_true
                coef= conn_model.coef_.T # transpose to get the right shape

                corrcoef_matrix = np.corrcoef(coef.flatten(), W_true.flatten())
                pearson_corr = corrcoef_matrix[0, 1]

                D_ev = pd.DataFrame()
                D_ev['n_task'] = [n_task]
                D_ev['metric'] = [metric]
                D_ev['correlation'] = pearson_corr
                results_df = pd.concat([results_df,D_ev],axis=0)

    return results_df


if __name__=='__main__':
    D = sim_connectivity()
    pass