def align_conditions(Ya, Yb, info_a, info_b):
    """
    Align two datasets based on shared conditions, align all conditions to the mean of shared conditions,
    then average shared conditions and append unique conditions.

    Args:
    Ya (numpy array): Dataset A (subjects x conditions x voxels) or (conditions x voxels)
    Yb (numpy array): Dataset B (subjects x conditions x voxels) or (conditions x voxels)
    info_a (pandas.DataFrame): Info file for Dataset A
    info_b (pandas.DataFrame): Info file for Dataset B

    Returns:
    combined_data (numpy array): Combined dataset
    combined_info (pandas.DataFrame): Combined info
    """

    shared_conditions = np.intersect1d(info_a['cond_code'], info_b['cond_code'])
    if len(shared_conditions) == 0:
        raise ValueError("No shared conditions between datasets.")

    # Standardize input dimensions to 3D if needed
    if len(Ya.shape) == 2:
        Ya = Ya[None, :, :]
    if len(Yb.shape) == 2:
        Yb = Yb[None, :, :]

    # Sort shared conditions and get indices
    shared_sorted = np.sort(shared_conditions)
    order_a = []  # Indices for shared conditions in Dataset A
    order_b = []  # Indices for shared conditions in Dataset B

    # Loop through each condition in the sorted shared conditions array
    for cond in shared_sorted:
        # Find the index of the condition in info_a that matches the current condition code
        idx_a = info_a[info_a['cond_code'] == cond].index[0]
        # Find the index of the condition in info_b that matches the current condition code
        idx_b = info_b[info_b['cond_code'] == cond].index[0]

        # Append the indices to the respective lists
        order_a.append(idx_a)
        order_b.append(idx_b)

    # Align shared conditions
    Ya_shared = Ya[:, order_a, :]
    Yb_shared = Yb[:, order_b, :]
    Ya_mean, Yb_mean = Ya_shared.mean(1, keepdims=True), Yb_shared.mean(1, keepdims=True)
    Ya_aligned, Yb_aligned = Ya - Ya_mean, Yb - Yb_mean

    # Average aligned shared conditions
    shared_avg = (Ya_aligned[:, order_a, :] + Yb_aligned[:, order_b, :]) / 2.0

    # Combine shared and unique conditions
    unique_a = np.setdiff1d(info_a['cond_code'], shared_sorted)
    unique_a_indices = info_a['cond_code'].isin(unique_a)
    unique_b = np.setdiff1d(info_b['cond_code'], shared_sorted)
    unique_b_indices = info_b['cond_code'].isin(unique_b)

    Ya_aligned_unique = Ya_aligned[:, unique_a_indices, :]
    Yb_aligned_unique = Yb_aligned[:, unique_b_indices, :]
    combined_data = np.concatenate([shared_avg, Ya_aligned_unique,
                                    Yb_aligned_unique], axis=1)

    # Create combined info file
    shared_info = info_a.loc[order_a, ['cond_name', 'cond_code']].copy()
    shared_info['source'] = 'averaged'
    unique_info = pd.concat([info_a[unique_a_indices], info_b[unique_b_indices]])
    unique_info['source'] = 'Novel'
    combined_info = pd.concat([shared_info, unique_info[['cond_name', 'cond_code', 'source']]], ignore_index=True)

    if Ya.shape[0] == 1:
        combined_data = combined_data[0]
    else:
        combined_data = combined_data


    return combined_data, combined_info

def make_dataset(data, info, battery, n_repeats, random_seed=1):
    """
    Creates a dataset with multiple betas per task, updating condition and partition vectors,
    while handling repeated tasks with different betas for each occurrence.
    
    Parameters:
    - data: numpy array of shape [voxels, conditions, subjects]
    - info: pandas DataFrame with 'cond_name' column
    - battery: list of task names (can include repeats)
    - n_repeats: number of betas to select per task
    - random_seed: int, optional random seed for reproducibility
    
    Returns:
    - dataset: numpy array of shape [voxels, selected_conditions, subjects]
    - cond_v_train: numpy array of condition labels
    - part_v_train: numpy array of partition labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Convert battery from indices to names if necessary
    if isinstance(battery[0], (int, np.integer)):
        battery = info['cond_name'].iloc[battery].tolist()
    
    indices, cond_v_train, part_v_train = [], [], []
    task_selected_betas = {}
    partition_numbers = list(range(1, n_repeats + 1))
    condition_counter = 1

    for task in battery:
        task_indices_all = info[info['cond_name'] == task].index.tolist()
        task_selected_betas.setdefault(task, set())

        # Filter available indices not yet selected for this task
        available_indices = list(set(task_indices_all) - task_selected_betas[task])

        # Ensure enough available betas for the current repetition
        if len(available_indices) < n_repeats:
            print(f"Insufficient betas for task '{task}'. Available: {len(available_indices)}")
            continue

        # Randomly select 'n_repeats' betas and update tracking
        selected_indices = np.random.choice(available_indices, size=n_repeats, replace=False).tolist()
        task_selected_betas[task].update(selected_indices)
        indices.extend(selected_indices)

        # Update condition and partition vectors for each selected beta
        cond_v_train.extend([condition_counter] * n_repeats)
        part_v_train.extend(partition_numbers)
        
        condition_counter += 1

    # Convert vectors to numpy arrays and create the dataset
    cond_v_train = np.array(cond_v_train)
    part_v_train = np.array(part_v_train)
    dataset = data[:, indices, :]

    return dataset, cond_v_train, part_v_train  


def exhuastive_traditional_batteries(Vs, isolate_parcels, length=8):
    isolated_parcels = Vs[:, isolate_parcels]
    isolated_sums = isolated_parcels.sum(axis=1)
    task_max_isolated = np.argmax(isolated_sums)

    other_task_indices = [i for i in range(Vs.shape[0]) if i != task_max_isolated]
    
    task_batteries = []
    for task in other_task_indices:
        task_list = [task_max_isolated, task] * (length // 2)
        task_batteries.append(tuple(task_list))
        
    return task_batteries
    

def traditional_battery(Vs, isolate_parcels, length=8):
    n_tasks, _ = Vs.shape
    contrast_results = []

    for task1 in range(n_tasks):
        for task2 in range(n_tasks):
            if task1 != task2:
                Vs_contrast = Vs[task1, :] - Vs[task2, :]

                # Find the highest activation parcel and its parcel index
                max_activation = np.max(Vs_contrast)
                max_parcel_index = np.argmax(Vs_contrast)

                # Find the second-highest activation parcel
                sorted_activations = np.sort(Vs_contrast)
                second_highest_activation = sorted_activations[-2]

                # Check if max_parcel_index is the parcel of interest and get the difference with 2nd highest activation
                if max_parcel_index in isolate_parcels:
                    difference = max_activation - second_highest_activation

                    contrast_results.append({
                        'task1': task1,
                        'task2': task2,
                        'difference': difference,
                        'max_activation': max_activation,
                        'second_highest_activation': second_highest_activation
                    })

    # Find the pairwise contrast with the maximum difference 
    if contrast_results:
        best_contrast = max(contrast_results, key=lambda x: x['difference'])
        best_task1 = best_contrast['task1']
        best_task2 = best_contrast['task2']

        # Make the battery
        tasks_list = [best_task1, best_task2] * (length // 2)
        return tuple(tasks_list)
    

    def max_value_distribution_analysis(df, num_batteries, num_iterations, eval_metric):
        metrics = ['variance', 'variance_mc', 'inverse_trace', 'inverse_trace_mc', 'log_det', 'log_det_mc']
        results = []

        for _ in range(num_iterations):
            # Randomly sample task batteries without replacement
            sampled_df = df.sample(n=num_batteries, replace=False)

            iteration_results = {}
            for metric in metrics:
                # Find the row with the highest value for the current metric
                max_metric_row = sampled_df.loc[sampled_df[metric].idxmax()]
                # Record the evaluation metric (e.g., 'cos') value
                iteration_results[metric] = max_metric_row[eval_metric]

            results.append(iteration_results)
            
        result_df = pd.DataFrame(results)

        return result_df
    


def custom_G(n_tasks=16, n_groups=4, group_size=4, target_corr=0.0004, variance_factors=[1.0, 0.75, 0.5, 0.25]):
    " makes a task covariance matrix with a target correlation between tasks"
    G = np.zeros((n_tasks, n_tasks))
    task_index = 0

    for group in range(n_groups):
        variances = variance_factors[group]
    
        # Compute covariances based on desired correlation
        covariances = target_corr * np.outer(variance_factors[group], variance_factors[group])
        np.fill_diagonal(covariances, variances)

        # Place the block into G
        start, end = task_index, task_index + group_size
        G[start:end, start:end] = covariances

        task_index += group_size

    return G

def custom_R(K_total, group_size, base_parcel_correlation, sub_parcel_extra_correlation):
    """
    makes a parcel covariance matrix with a base correlation between main parcels and extra correlation within subparcels
    """
    R = np.full((K_total, K_total), base_parcel_correlation)
    for i in range(0, K_total, group_size):
        R[i:i+group_size, i:i+group_size] = base_parcel_correlation + sub_parcel_extra_correlation

    np.fill_diagonal(R, 1)
    return R