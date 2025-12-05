"""
Module for functions used in the simulation of the task-battery construction problem.
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
from scipy.ndimage import binary_dilation, binary_erosion, shift
from scipy.stats import gamma
from scipy.optimize import brentq
from scipy.ndimage import distance_transform_edt



device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

def make_U_spatial(height=100, width=100, K_main=5):
    """Like make_U_simple but enforces equal parcel size per centroid."""
    centroids = np.array([
        [0, 0],
        [width - 1, 0],
        [0, height - 1],
        [width - 1, height - 1],
        [(width - 1) / 2, (height - 1) / 2],
    ])

    X, Y = np.meshgrid(np.arange(width), np.arange(height), indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    dists = np.sum((coords[:, None, :] - centroids[None, :, :])**2, axis=-1)

    P = height * width
    desired = P // K_main
    labels = np.full(P, -1, dtype=int)
    remaining = set(range(P))

    for k in range(K_main - 1):
        rem = np.array(list(remaining))
        order = np.argsort(dists[rem, k])
        chosen = rem[order[:desired]]
        labels[chosen] = k
        remaining -= set(chosen)

    labels[list(remaining)] = K_main - 1

    U_true = np.zeros((K_main, P), dtype=np.float64)
    U_true[labels, np.arange(P)] = 1.0
    return U_true

def make_U_individuals_old(U_true,grid_width,grid_height, n_individuals=8,
                               shift_range=3, size_jitter=2,
                               seed=None, device=None):
    """
    Make individual parcellations where only the last parcel moves or changes size.

    Args:
    U_true : Ground truth parcellation (K, P)
    grid_width : width of the spatial grid
    grid_height : height of the spatial grid
    n_individuals : number of individuals to simulate
    shift_range : how many pixels to shift parcel 5 (random x/y)
    size_jitter : how much to grow/shrink parcel 5
    seed : random seed for reproducibility
    device : torch device to place the output tensors on
    Returns:
    individuals : list of individual parcellations (each is a tensor of shape (K, P))
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(U_true, pt.Tensor):
        U_true = U_true.detach().cpu().numpy()

    K, P = U_true.shape
    width, height = grid_width, grid_height
    base_labels = np.argmax(U_true, axis=0).reshape(width, height)

    individuals = []
    for _ in range(n_individuals):
        new_labels = base_labels.copy()

        # extract parcel 5 mask (last parcel)
        k = K - 1
        mask = (base_labels == k).astype(float)

        # random small spatial shift
        dx = np.random.randint(-shift_range, shift_range + 1)
        dy = np.random.randint(-shift_range, shift_range + 1)
        mask_shifted = shift(mask, shift=(dx, dy), order=0, mode='nearest')

        # random growth or shrinkage
        if np.random.rand() < 0.5:
            mask_shifted = binary_dilation(mask_shifted, iterations=np.random.randint(1, size_jitter + 1))
        else:
            mask_shifted = binary_erosion(mask_shifted, iterations=np.random.randint(1, size_jitter + 1))

        # update the parcel in the label map
        new_labels[mask_shifted > 0] = k

        # rebuild binary membership matrix
        U_ind = np.zeros((K, P))
        for kk in range(K):
            U_ind[kk, new_labels.flatten() == kk] = 1

        if device is not None:
            U_ind = pt.from_numpy(U_ind).to(device=device, dtype=pt.float64)

        individuals.append(U_ind)

    return individuals

def get_boundary(region, grid_width=30, grid_height=30):
    """Return coordinates of pixels on the outer boundary of 'region' from a 2d mask of 1s and 0s.
    
    Args:
        region : 2D boolean numpy array
    Returns:
        boundary : list of (x, y) tuples of boundary pixel coordinates
    """
    boundary = []
    for x in range(1, grid_width - 1):
        for y in range(1, grid_height - 1):
            if region[x, y]:
                # Boundary if any of 4-neighbors is False
                if not (region[x-1, y] and region[x+1, y] and region[x, y-1] and region[x, y+1]):
                    boundary.append((x, y))
    return np.array(boundary)

def adjust_last_parcel(U_true, grid_width, grid_height, target_size, seed=None):
    """
    Generate an individual variant of the base parcellation by adjusting the size of the last parcel

    Args:
        U_true        : (K, P) binary parcellation matrix (one-hot format)
        grid_width    : width of spatial grid
        grid_height   : height of spatial grid
        target_size   : desired number of voxels for the last parcel
        seed          : random seed for reproducibility

    Returns:
        U_adj : adjusted (K, P) parcellation matrix
    """
    if seed is not None:
        np.random.seed(seed)

    # Setup and extract index of last parcel
    K, P = U_true.shape
    labels = np.argmax(U_true, axis=0).reshape(grid_width, grid_height)
    k = K - 1
    mask = (labels == k)
    

    # get how much needs to be changed (added or removed)
    current_size = mask.sum()
    diff = target_size - current_size

    # if its already the right size, return copy
    if diff == 0:
        return U_true.copy()

    mask_adj = mask.copy()

    # Shrink region (remove voxels)
    if diff < 0:
        to_remove = -diff
        while to_remove > 0:
            # gets x y coordinates of boundary voxels
            boundary = get_boundary(mask_adj,grid_width=grid_width,grid_height=grid_height)
            if len(boundary) == 0:
                break
            n = min(to_remove, len(boundary))
            # randomly select n boundary voxels to remove
            chosen = boundary[np.random.choice(len(boundary), n, replace=False)]
            for (x, y) in chosen:
                mask_adj[x, y] = False
            to_remove -= n

        # Reassign removed voxels to nearest non-k region
        non_k_mask = (labels != k)
        # this func finds nearest voxel not belonging to target mask, and the loop reassigns labels
        dist, nearest_idx = distance_transform_edt(~non_k_mask, return_indices=True)
        for (x, y) in np.argwhere(mask & ~mask_adj):
            nx, ny = nearest_idx[0, x, y], nearest_idx[1, x, y]
            labels[x, y] = labels[nx, ny]

    # --- Grow region (add voxels)
    else:
        to_add = diff
        while to_add > 0:
            boundary = get_boundary(mask_adj,grid_width=grid_width,grid_height=grid_height)
            candidates = []
            for (x, y) in boundary:
                # Collect 4-neighbor candidates that aren’t part of region
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    xx, yy = x + dx, y + dy
                    if 0 <= xx < grid_width and 0 <= yy < grid_height:
                        if not mask_adj[xx, yy]:
                            candidates.append((xx, yy))
            if len(candidates) == 0:
                break
            # some would be duplicates, remove them
            candidates = np.unique(candidates, axis=0)
            n = min(to_add, len(candidates))
            chosen = candidates[np.random.choice(len(candidates), n, replace=False)]
            for (x, y) in chosen:
                mask_adj[x, y] = True
                labels[x, y] = k
            to_add -= n

    # Convert back to binary (K, P)
    U_adj = np.zeros_like(U_true)
    flat_labels = labels.flatten()
    for kk in range(K):
        U_adj[kk, flat_labels == kk] = 1

    return U_adj

def make_U_individuals(U_true, grid_width, grid_height,
                       n_individuals=8, size_range=(120, 260),
                       seed=None, device=None):
    """
    Generate multiple individual parcellations by randomly varying
    the size of the last parcel within a specified range.

    Args:
        U_true        : (K, P) base parcellation (numpy or torch tensor)
        grid_width    : spatial grid width
        grid_height   : spatial grid height
        n_individuals : number of individuals to simulate
        size_range    : (min, max) range for target size of last parcel
        seed          : random seed for reproducibility
        device        : optional torch device for output
    Returns:
        individuals : list of individual parcellations (each (K, P))
    """
    if seed is not None:
        np.random.seed(seed)

    # Convert torch tensor to numpy if needed
    if isinstance(U_true, pt.Tensor):
        U_true = U_true.detach().cpu().numpy()

    individuals = []
    for i in range(n_individuals):
        # Randomly choose a target size for parcel 5
        target_size = np.random.randint(size_range[0], size_range[1] + 1)

        # Create adjusted individual
        U_ind = adjust_last_parcel(U_true, grid_width, grid_height,
                                   target_size=target_size, seed=seed+i)

        if device is not None:
            U_ind = pt.from_numpy(U_ind).to(device=device, dtype=pt.float64)

        individuals.append(U_ind)

    return individuals





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

def get_jaccard_single(U_true, U_pred, roi_index=0):
    """
    Compute Jaccard index (Intersection over Union) for a binary ROI (single class only).
    Assumes U_true and U_pred are one-hot tensors of shape (1, 2, P) or (2, P).

    Args:
        U_true (Tensor): Ground truth parcellation
        U_pred (Tensor): Predicted parcellation
        roi_index (int): Index of ROI class to evaluate (default 0)

    Returns:
        float: Jaccard index for ROI
    """
    if len(U_true.shape) == 3:
        U_true = U_true[0]
    if len(U_pred.shape) == 3:
        U_pred = U_pred[0]

    TP = (U_true[roi_index] * U_pred[roi_index]).sum()
    size_true = U_true[roi_index].sum()
    size_pred = U_pred[roi_index].sum()
    union = size_true + size_pred - TP
    jaccard = TP / union
    return jaccard.item()


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


def make_thresholded_contrast(task1, task2, threshold, mode='percentile'):
    """
    Get the contrast between two tasks and threshold it.

    Args:
        task1 (Tensor): Task 1 data (n_voxels)
        task2 (Tensor): Task 2 data (n_voxels)
        threshold (float): Threshold value
            - if mode='percentile', interpreted as quantile (0–1)
            - if mode='absolute', interpreted as raw value
        mode (str): 'percentile' or 'absolute'
    Returns:
        contrast_one_hot (Tensor): One-hot encoded contrast (2, n_voxels)
    """

    contrast_data = task1 - task2

    if mode == 'percentile':
        thresh_value = pt.quantile(contrast_data, threshold)
    elif mode == 'absolute':
        thresh_value = threshold
    else:
        raise ValueError("mode must be either 'percentile' or 'absolute'")

    mask = (contrast_data >= thresh_value).long()

    # one-hot encoding 
    contrast_one_hot = pt.stack([
        (mask == 1).float(),  # region A
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
            V_lib = V_lib - V_lib.mean(axis=0,keepdims=True)
            G_lib = V_lib @ V_lib.T
            
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

                V_battery = ut.center_matrix(V_battery,axis=0)
                V_battery = ut.normalize_matrix(V_battery,axis=0)

                # Build the parcellation
                U_hats = et.estimate_Us(Y_battery, V_battery, method='cos_angle', hard=True)

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
    

def sim_single_vs_multi(U_individuals,U_individuals_collapsed,base_noise,snr_ratios,seed = 0):
    """ Single simulation comparing contrast localization vs multi-task localization
    args:
        U_individuals: list of individual parcellations (each is a tensor of shape (K, P))
        U_individuals_collapsed: list of individual collapsed parcellations (each is a tensor of shape (2, P))
        base_noise: base noise level
        snr_ratios: list of snr ratios to sample from gamma distribution (from empirical data mdtb1)
        seed: random seed for reproducibility
    returns:
        results_df: DataFrame with the results of the simulations
    """
    results_df = pd.DataFrame()
    max_n_task = 10
    types = ['contrast_fixed','multitask','contrast_adaptive']

    # fit gamma to snr ratios
    shape, loc, scale = gamma.fit(snr_ratios, floc=0)  

    # generate Vs that are orthonal on the column and row (has to be square matrix)
    rng= np.random.default_rng(seed)
    V = np.eye(5, 5)
    A = rng.normal(size=(5, 5))
    Q, _ = np.linalg.qr(A)
    V_lib = Q @ V
    V_lib = pt.tensor(V_lib, dtype=pt.float64, device=device)

    parcellation_contrast_T = []
    parcellation_contrast_percentage = []
    parcellation_multi = []


    for type in types:
        # get the single contrast
        if  type == 'contrast_fixed' or type == 'contrast_adaptive':
            max_idx, min_idx = find_max_contrast_against_all(V_lib, 4)
            combination = [max_idx, min_idx]
        elif type == 'multitask':
            combination = [0,1,2,3,4]

        # get the V localizer
        V_battery = V_lib[combination,:]
        n_task = V_battery.shape[0]

        # Battery-level noise (same for all subs)
        weighted_noise_std = get_weighted_noise_std(n_task=n_task, max_n_task=max_n_task, noise=base_noise)
        battery_noise = rng.normal(0, weighted_noise_std, (V_battery.shape[0], U_individuals[0].shape[1]))
        battery_noise = pt.tensor(battery_noise, dtype=pt.float64, device=device)

         # precompute subject data
        contrasts, true_sizes = [], []
        for i, (U, Uc) in enumerate(zip(U_individuals, U_individuals_collapsed)):
            rng_sub = np.random.default_rng(seed+i)
            snr_factor = rng_sub.gamma(shape, scale=scale)
            Y = V_battery @ U 
            Y = Y * np.sqrt(snr_factor)
            Y = Y + battery_noise
            contrasts.append((Y[0,:], Y[1,:]))
            true_sizes.append(Uc[0,:].sum().item())
        avg_true = np.mean(true_sizes)

        # calibrate threshold optimizing function that finds value that minimizes the difference between predicted and true size on avg
        if type == 'contrast_fixed':
            def f(th):
                pred_sizes = [make_thresholded_contrast(y1,y2,threshold=th,mode='absolute')[0,:].sum().item()
                              for y1,y2 in contrasts]
                return np.mean(pred_sizes) - avg_true
            best_th = brentq(f, 0.01, 50.0)
            print(f"Best threshold (matched to actual data): {best_th:.3f}")
        if type == 'contrast_adaptive':
            def f(th):
                pred_sizes = [make_thresholded_contrast(y1,y2,threshold=th,mode='percentile')[0,:].sum().item()
                              for y1,y2 in contrasts]
                return np.mean(pred_sizes) - avg_true
            best_th = brentq(f, 0.01, 0.99)
            print(f"Best percentile threshold (matched to actual data): {best_th:.3f}")

        for individual in range(len(U_individuals)):
            # get the data for the parcellation estimation and add noise
            Y_battery = V_battery @ U_individuals[individual] 

            # subject-specific SNR variation
            rng_sub = np.random.default_rng(seed= seed + individual)
            snr_factor = rng_sub.gamma(shape, scale=scale)
            Y_battery = Y_battery * np.sqrt(snr_factor)

            # add battery-level noise
            Y_battery = Y_battery + battery_noise

            if type == 'multitask':
                # Y_battery = ut.center_matrix(Y_battery,axis=0)
                Y_battery = ut.normalize_matrix(Y_battery,axis=0)

                # V_battery = ut.center_matrix(V_battery,axis=0)
                V_battery = ut.normalize_matrix(V_battery,axis=0)

                U_hat = et.estimate_Us(Y_battery, V_battery, method='cos_angle', hard=True)
                U_hat= collapse_U(U_hat, target_parcels_indices=[4])[0]
                parcellation_multi.append(U_hat.cpu().numpy())

            elif type == 'contrast_fixed':
                U_hat = make_thresholded_contrast(Y_battery[0,:], Y_battery[1,:],threshold= best_th,mode='absolute')
                parcellation_contrast_T.append(U_hat.cpu().numpy())

            elif type == 'contrast_adaptive':
                U_hat = make_thresholded_contrast(Y_battery[0,:], Y_battery[1,:],threshold= best_th,mode='percentile')
                parcellation_contrast_percentage.append(U_hat.cpu().numpy())
                

            predicted_size = U_hat[0, :].sum().item()

            # Evaluate the contrast
            accuracy = get_dice_single(U_individuals_collapsed[individual], U_hat)
            D_ev = pd.DataFrame()
            D_ev['type'] = [type]
            D_ev['n_tasks'] = [n_task]
            D_ev['snr_factor'] = [snr_factor]
            D_ev['individual'] = [individual]
            D_ev['accuracy'] = accuracy
            D_ev['predicted_size'] = predicted_size
            D_ev['true_size'] = U_individuals_collapsed[individual][0,:].sum().item()
            D_ev['true_everything_size'] = U_individuals_collapsed[individual][1,:].sum().item()
            D_ev['predicted_everything_size'] = U_hat[1,:].sum().item()
            D_ev['threshold'] = [best_th if type in ['contrast_fixed','contrast_adaptive'] else np.nan]
            results_df = pd.concat([results_df,D_ev],axis=0)

    return results_df,parcellation_contrast_T,parcellation_contrast_percentage,parcellation_multi




if __name__=='__main__':
    D = sim_parcellation()
    pass