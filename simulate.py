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

def get_dice_single(U_true, U_pred, roi_index=0):
    """
    Compute Dice coefficient for a binary ROI (single class only).
    Assumes U_true and U_pred are one-hot tensors of shape (1, 2, P) or (2, P).

    Args:
        U_true (Tensor): Ground truth parcellation
        U_pred (Tensor): Predicted parcellation
        roi_index (int): Index of ROI class to evaluate (default 0)

    Returns:
        float: Dice score for ROI
    """
    if len(U_true.shape) == 3:
        U_true = U_true[0]
    if len(U_pred.shape) == 3:
        U_pred = U_pred[0]

    TP = (U_true[roi_index] * U_pred[roi_index]).sum()
    size_true = U_true[roi_index].sum()
    size_pred = U_pred[roi_index].sum()
    dice = 2 * TP / (size_true + size_pred)
    return dice.item()

def get_dice_multiclass(U_true, U_pred):
    """
    Compute average Dice coefficient across all classes.

    Args:
        U_true (Tensor): Ground truth (K, P) or (1, K, P)
        U_pred (Tensor): Predicted (K, P) or (1, K, P)

    Returns:
        float: Mean Dice over all classes
    """
    if len(U_true.shape) == 2:
        U_true = U_true.unsqueeze(0)
    if len(U_pred.shape) == 2:
        U_pred = U_pred.unsqueeze(0)

    intersection = (U_true * U_pred).sum(dim=2)
    size_true = U_true.sum(dim=2)
    size_pred = U_pred.sum(dim=2)
    dice_scores = 2 * intersection / (size_true + size_pred)
    return dice_scores.mean().item()

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

def find_max_contrast_against_all(Vs, region_idx):
    """
    Find the task that maximizes and minimizes the contrast between a region of interest (ROI)
    and the average of all other regions.

    Args:
        Vs: Task library (n_tasks, n_parcels)
        region_idx: Index of the region of interest (0-based)

    Returns:
        max_idx: Index of the task with highest contrast (ROI >> others)
        min_idx: Index of the task with lowest contrast (ROI << others)
    """
    roi = Vs[:, region_idx]
    
    # Exclude the ROI column to get all other regions
    others = pt.cat([Vs[:, :region_idx], Vs[:, region_idx + 1:]], dim=1)
    others_mean = pt.mean(others, dim=1)

    # Contrast: ROI - mean(other regions)
    difference = roi - others_mean
    sorted_idx = pt.argsort(difference)

    min_idx = sorted_idx[0].item()
    max_idx = sorted_idx[-1].item()

    return [max_idx, min_idx]


def make_thresholded_contrast(task1, task2, threshold):
    """gets the contrast between two tasks and thresholds it
    Args:
        task1: Task 1 data (n_voxels)
        task2: Task 2 data (n_voxels)
        threshold: Threshold for the contrast
    Returns:
        contrast_one_hot: One-hot encoded contrast (2, n_voxels)
    """

    contrast_data = task1 - task2
    mask = (contrast_data >= pt.quantile(contrast_data, threshold)).long()

    # make one hot
    contrast_one_hot = pt.stack([
        (mask == 1).float(),  # positive/ region A
        (mask == 0).float()   # everything else 
    ], dim=0)
    return contrast_one_hot

def collapse_U(U, target_parcels_indices = None):
    """
    Collapse the U matrix into two parcels: one for the target parcel and one for everything else.

    Args:
        U (Tensor): Shape (n_sub, K, P) or (K, P):
            - n_sub: number of subjects
            - K: number of parcels
            - P: number of voxels
        target_parcel (int): Index of the parcel to isolate

    Returns:
        Tensor: Collapsed U of shape (n_sub, 2, P) or (2, P)
    """
    # if 2d make 3d
    added_batch_dim = False
    if U.dim() == 2:
        U = U.unsqueeze(0)
        added_batch_dim = True

    all_indices = np.arange(U.shape[1])
    other_parcels_indices = np.setdiff1d(all_indices, target_parcels_indices)
    # select the target and non-target parcels
    target = U[:, target_parcels_indices, :]
    rest = U[:, other_parcels_indices, :]
    target_sum = target.sum(dim=1, keepdim=True)
    rest_sum = rest.sum(dim=1, keepdim=True)

    # combine
    U_collapsed = pt.cat([ target_sum,rest_sum], dim=1)

    # Remove batch dim if original input was 2D
    if added_batch_dim:
        U_collapsed = U_collapsed.squeeze(0)

    return U_collapsed

def sim_single_contrast(num_task_lib = 100,
                        n_parcels = 5,
                        U_true = None,
                        base_noise = 5,
                        max_battery_size = 28,
                        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5],
                        U_true_collapsed = None,
                        n_sim = 50,
                        seed = None):
    """ Single simulation for the single contrast parcellation estimation
    Args:
        num_task_lib: Number of tasks in the library
        n_parcels: Number of parcels in the U_true
        U_true: ground truth parcellation
        base_noise: Base noise level
        max_battery_size: Maximum battery size (from the list of battery sizes in the multi-task simulation)
        thresholds: List of thresholds to test
        U_true_collapsed: Collapsed U_true for the single region analysis
        n_sim: Number of simulations to run
        seed: Random seed for reproducibility
    returns:
    """

     # Make new task battery
    if seed is not None:
        rng= np.random.default_rng(seed=seed)
    else:
        rng= np.random.default_rng()
    
    results_df = pd.DataFrame()
    for n in range(n_sim):

        V_lib = rng.normal(0,1,(num_task_lib, n_parcels))
        V_lib = V_lib - V_lib.mean(axis=0,keepdims=True)
        V_lib = pt.tensor(V_lib, device=device, dtype=pt.float64)

        # get the single contrast
        max_idx, min_idx = find_max_contrast_against_all(V_lib, 4)
        combination = [max_idx, min_idx]

        # get the V localizer
        V_localizer = V_lib[combination,:]

        # get the data for the parcellation estimation and add noise
        Y_localizer = V_localizer @ U_true
        weighted_noise_std = get_weighted_noise_std(2, max_battery_size, base_noise)
        rng = np.random.default_rng(seed)
        noise = rng.normal(0,weighted_noise_std,Y_localizer.shape)
        noise = pt.tensor(noise, dtype=pt.float64, device=Y_localizer.device)
        Y_localizer = Y_localizer + noise
        # center but no normalization?
        Y_localizer = ut.center_matrix(Y_localizer,axis=0)
        # Y_localizer = ut.normalize_matrix(Y_localizer,axis=0)

        for threshold in thresholds:
            # get the thresholded contrast
            thresholded_contrast = make_thresholded_contrast(Y_localizer[0,:], Y_localizer[1,:], threshold)

            # Evaluate the contrast
            accuracy = get_dice_single(U_true_collapsed, thresholded_contrast)

            D_ev = pd.DataFrame()
            D_ev['threshold'] = [threshold]
            D_ev['accuracy'] = accuracy
            results_df = pd.concat([results_df,D_ev],axis=0)

    return results_df


def sim_parcellation(num_task_lib = 100,
                     n_parcels = 5,
                     U_true = None,
                     metrics = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc'],
                     battery_sizes = [3,4,6,8,10,14,18,24,28],
                     n_batteries = 100,
                     base_noise = 2,
                     collapsed_U_true = None,
                     n_sim = 50,
                     seed = None):
    """ Single simulation for the parcellation estimation
    Args:
        num_task_lib: Number of tasks in the library
        n_parcels: Number of parcels in the U_true
        U_true: ground truth parcellation
        battery_sizes: List of battery sizes to test
        n_batteries: Number of batteries to sample for each battery size
        base_noise: Base noise level
        collapsed_U_true: Collapsed U_true for the single region analysis
        n_sim: Number of simulations to run
        seed: Random seed for reproducibility
    returns:
        results_df: DataFrame with the results of the simulations
    """
    # Make new task battery
    if seed is not None:
        rng= np.random.default_rng(seed=seed)
    else:
        rng= np.random.default_rng()

    # constants
    max_battery_size = max(battery_sizes)

    results_df =pd.DataFrame()
    for n_task in battery_sizes:
        print(f"Processing battery size: {n_task}")
        for n in range(n_sim):
            V_lib = rng.normal(0,1,(num_task_lib, n_parcels))
            G_lib = V_lib @ V_lib.T
            G_lib = G_lib - G_lib.mean(axis=0,keepdims=True)
            # ensure tensor
            V_lib = pt.tensor(V_lib, device=device, dtype=pt.float64)

            # Generate possible battery combinations for current battery size and calculate eigenmetrics
            D = ct.build_combinations(G_lib=G_lib, strategy='random',n_batteries=n_batteries,n_tasks=n_task,replacement=False,rest_idx=None,seed=seed)
            for metric in metrics:
                # Find the best battery for the metric
                D_best = ct.choose_combination(D,metric)
                top_comb = D_best['combination'].values[0]

                if n_task == 2:
                    top_comb = find_max_contrast_against_all(Vs=V_lib,region_idx=4)

                # get the V battery
                V_battery = V_lib[top_comb,:]

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
                    U_hats = collapse_U(U_hats, target_parcels_indices=[4])

                # Evaluate the parcellation
                if collapsed_U_true is not None:
                    accuracy = get_dice_single(collapsed_U_true, U_hats)
                else:
                    accuracy = get_dice_multiclass(U_true, U_hats)

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
                     ridge_alpha = 1000,
                     seed = None):
    """ Single simulation for the connectivity estimation.
    """

    # Make new task battery
    if seed is not None:
        rng= np.random.default_rng(seed=seed)
    else:
        rng= np.random.default_rng()

    # constants
    metrics = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc']
    max_n_task = max(battery_sizes)

    results_df = pd.DataFrame()
    for n_task in battery_sizes:
        print(f"Processing battery size: {n_task}")

        for n in range(n_sim):
            V_lib = rng.normal(0,1,(num_task_lib, n_parcels))
            V_lib = V_lib - V_lib.mean(axis=0,keepdims=True)
            G_lib = V_lib @ V_lib.T

            # sample the connectivity weights from a normal
            W_true = rng.normal(0,1,(n_parcels, n_voxels_y))

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
                coef= conn_model.coef_.T

                corrcoef_matrix = np.corrcoef(coef.flatten(), W_true.flatten())
                pearson_corr = corrcoef_matrix[0, 1]

                D_ev = pd.DataFrame()
                D_ev['n_task'] = [n_task]
                D_ev['metric'] = [metric]
                D_ev['correlation'] = pearson_corr
                results_df = pd.concat([results_df,D_ev],axis=0)

    return results_df


if __name__=='__main__':
    D = sim_parcellation()
    pass