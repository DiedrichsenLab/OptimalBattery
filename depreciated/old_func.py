def estimate_Us_ols(Y,V):
    """
    get U_hat using OLS regression
    Args:
        Y: Individual fMRI data (n_subjects, n_tasks, n_voxels)
        V: Functional Profile (n_tasks, n_components)
    Returns:
        U_hat: Individual parcellations (n_subjects, n_components, n_voxels)
    """
    # Uhat =  (V^T V)^-1 V^T Yyy


    U_hats = np.linalg.inv(V.T @ V) @ V.T @ Y
    return U_hats # s, k, p


def find_optimal_battery(task_matrix, task_names, num_tasks=4, function='trace', top_n=1, sample_size=1000, average_across_subjects=True,offset=0.0001):

    """
    Finds the top N combinations of tasks based on a specified function, either on the group-averaged second moment matrix or by averaging matrices across subjects.
    
    Args:
        task_matrix (torch.Tensor): The task data matrix of shape (num_subjects, num_tasks, num_voxels)
        task_names (list): List of task names corresponding to the tasks in the task_matrix.
        num_tasks (int): The number of tasks required for the battery
        function (str): The function to optimize for. Can be 'trace' (for total variance) or 'inverse_trace' (for total precision).
        top_n (int): The number of top combinations to return. Default is 1.
        sample_size (int or None): If specified, randomly sample this many combinations from all possible combinations. If None, use all combinations.
        average_across_subjects (bool): If True, perform the search on the group second moment matrix based on group-averaged data. 
                                        If False, perform the search on a group second moment matrix based on averaging individual second moment matrices.
    
    Returns:
        list of tuples: A list of the top N combinations. Each tuple contains:
            - function_result (float): The result of the specified function for this combination.
            - combination (numpy.array): The indices of the tasks in the combination.
    """

    # if task matrix has nan values, replace them with zeros
    task_matrix[np.isnan(task_matrix)] = 0

    total_tasks = len(np.unique(task_names))
    num_runs = task_matrix.shape[1] // total_tasks
    
    # Generate all task indices
    task_indices = np.arange(total_tasks)

    if sample_size is not None:
        sampled_combinations = np.random.randint(0, len(task_indices), (sample_size, num_tasks))

    else:
        # Generate all possible combinations if sample_size is None
        all_combinations = list(combinations_with_replacement(task_indices, num_tasks))
        sampled_combinations = all_combinations

    # create condition_v and partition vector
    cond_vec = np.tile(np.arange(1, total_tasks+1), num_runs)

    # make a vector of 1 repated 16 times then 2 repeated 16 times and so on
    part_vec = np.repeat(np.arange(1, num_runs+1), total_tasks)
    
    # If we are averaging across subjects, average task_matrix across subjects (dim=0)
    if average_across_subjects:
        avg_task_matrix = np.nanmean(task_matrix, axis=0)  # Averaged across subjects
        G_group,E = est_G_crossval(avg_task_matrix,cond_vec,part_vec)

    else:
        # Compute the covariance matrix for each subject individually
        G_matrices = []
        for subj in range(task_matrix.shape[0]): 
            G_s,E_s = est_G_crossval(task_matrix[subj], cond_vec, part_vec)
            G_matrices.append(G_s)
        G_matrices_stacked = np.stack(G_matrices, 0)
        G_group = np.nanmean(G_matrices_stacked, axis=0)  # Averaged across subjects

    eye_matrix = offset * np.eye(num_tasks)
    ones_vector = np.ones((num_tasks, num_tasks))
    centering_matrix = np.eye(num_tasks) - ones_vector / num_tasks

    # Initialize top results based on function
    if function in ['trace', 'determinant','maximize_lowest_eigenvalue']:
        top_results = [(-float('inf'), None)] * top_n
    elif function == 'inverse_trace':
        top_results = [(float('inf'), None)] * top_n


    for i, comb in enumerate(sampled_combinations):
        if i % 100000 == 0:
            print(f"Processing sample {i+1}/{len(sampled_combinations)}")

        # Extract subset covariance for the averaged data
        subset_varcov = G_group[comb, :][:, comb]
        centered_varcov = centering_matrix @ subset_varcov @ centering_matrix.T
        centered_varcov = centered_varcov + eye_matrix

        eigenvalues, _ = np.linalg.eigh(centered_varcov)

        # sort the eigenvalues descendingly
        eigenvalues = np.sort(eigenvalues)[::-1]

        # exclude the last
        eigenvalues = eigenvalues[:-1]

        # Compute trace or inverse trace
        if function == 'trace':
            function_result = np.sum(eigenvalues)
        elif function == 'inverse_trace':
            inverse_eigenvalues = 1.0 / eigenvalues
            function_result = np.sum(inverse_eigenvalues)
        elif function == 'determinant':
            function_result = np.prod(eigenvalues)
        elif function == 'maximize_lowest_eigenvalue':
            function_result = eigenvalues[-1]
        else:
            raise ValueError("Invalid function argument")

        function_result_value = function_result.item()


        # After initialization, update only if the new result is better
        if function == 'inverse_trace':
            if function_result_value < top_results[-1][0]:  # We want the smallest values for 'inverse_trace'
                top_results[-1] = (function_result_value,comb)
        else:  # 'trace'
            if function_result_value > top_results[-1][0]:  # We want the largest values for 'trace'
                top_results[-1] = (function_result_value,comb)

        # Sort only after an update
        top_results.sort(reverse=(function in ['trace', 'determinant','maximize_lowest_eigenvalue']))

    return top_results

def combination_vectors(data, info, battery, n_repeats, random_seed=1):
    """
    Create a dataset with multiple betas per task, updating condition and partition vectors.
    Handles repeated tasks in the battery, ensuring different betas are selected for each occurrence.

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

    # if battery is indices instead of names, convert to names, it could be numpy int
    if isinstance(battery[0], (int,np.int32)):
        battery = info['cond_name'].iloc[battery].tolist()

    indices = []
    cond_v_train = []
    part_v_train = []
    
    # Keep track of the number of times each task has appeared
    task_occurrence = {}
    
    # Keep track of betas already selected for each task
    task_selected_betas = {}
    
    # For consistent partition numbers across tasks, create partition_numbers list
    partition_numbers = list(range(1, n_repeats+1))
    
    # Initialize condition counter
    condition_counter = 1
    
    # For each task in the battery
    for task in battery:
        # Update the occurrence count for the task
        occurrence = task_occurrence.get(task, 0) + 1
        task_occurrence[task] = occurrence
        
        # Get all indices for the current task
        task_indices_all = info[info['cond_name'] == task].index.tolist()
        
        # Initialize selected betas for this task if not already
        if task not in task_selected_betas:
            task_selected_betas[task] = []
        
        # Exclude betas already selected for this task
        available_indices = list(set(task_indices_all) - set(task_selected_betas[task]))
        
        # Check if there are enough betas for the task occurrence
        if len(available_indices) < n_repeats:
            print(f"Not enough betas for task '{task}' occurrence {occurrence}. Available betas: {len(available_indices)}")
            continue  # Skip if not enough betas
        
        # Randomly select 'n_repeats' betas without replacement
        selected_indices = np.random.choice(available_indices, size=n_repeats, replace=False).tolist()
        
        # Update the selected betas for this task and occurrence
        task_selected_betas[task].extend(selected_indices)
        
        # Extend the indices list with the selected indices
        indices.extend(selected_indices)
        
        # For each selected beta, update the condition and partition vectors
        for i in range(n_repeats):
            # Condition number: assign a unique number per task occurrence
            cond_v_train.append(condition_counter)
            # Partition number: cycle through 1 to n_repeats for each beta
            part_v_train.append(partition_numbers[i])
        
        # Increment condition counter after each task occurrence
        condition_counter += 1
    
    # Convert condition and partition vectors to numpy arrays
    cond_v_train = np.array(cond_v_train)
    part_v_train = np.array(part_v_train)
    
    # Create the dataset with the selected indices
    dataset = data[:, indices, :]
    
    return dataset, cond_v_train, part_v_train


def gram_schmidt(V):
    """ Apply Gram-Schmidt process to matrix V for orthogonalization. """
    Q = np.zeros_like(V)
    for i in range(V.shape[0]):
        q = V[i, :]
        for j in range(0, i):
            q = q - np.dot(Q[j, :], V[i, :]) * Q[j, :]
        Q[i, :] = q / np.linalg.norm(q)
    return Q


def generate_Vs(n_tasks, n_parcel, Vs_type='random', noise_std=0.01):
    """
    Generate Vs of different types: 'normal', 'orthogonal', 'correlated'.

    Args:
    - n_tasks: Number of tasks (rows).
    - n_parcel: Number of parcels (columns).
    - Vs_type: Type of Vs to generate ('normal', 'orthogonal', 'correlated').
    - noise_std: Standard deviation for noise added to Vs.

    Returns:
    - Vs: Generated Vs matrix of shape (n_tasks, n_parcel).
    """
    if Vs_type == 'random':
        # Generate Vs from normal distribution
        Vs = np.random.normal(0, 1, (n_tasks, n_parcel))
        Vs += noise_std * np.random.randn(n_tasks, n_parcel) # add noise
        # Vs = Vs - np.mean(Vs, axis=0, keepdims=True)  # Subtract row mean

    elif Vs_type == 'orthogonal':
        V_random = np.random.randn(n_tasks, n_parcel)
        # Center the data
        # V_random -= np.mean(V_random, axis=1, keepdims=True)
        # Apply Gram-Schmidt
        Vs = gram_schmidt(V_random)
        # Optionally add noise
        Vs += np.random.normal(0, noise_std, (n_tasks, n_parcel))

    elif Vs_type == 'correlated':
        # Generate Vs with high correlation between tasks
        base_pattern = np.random.randn(1, n_parcel) * .001  # Strong base pattern
        Vs = np.tile(base_pattern, (n_tasks, 1))  # Repeat pattern for all tasks
        Vs += noise_std * np.random.randn(n_tasks, n_parcel)  # Add small noise
        # Vs = Vs - np.mean(Vs, axis=0, keepdims=True)
        
    else:
        raise ValueError(f"Unknown Vs_type '{Vs_type}'. Choose 'normal', 'orthogonal', or 'correlated'.")
    
    return Vs


def estimate_Us_NNLS(Y, V):
    """
    Estimate the U matrices using NNLS regression.

    Args:
        Y: Individual fMRI data (n_subjects, n_tasks, n_voxels)
        V: Functional Profile (n_tasks, n_components)

    Returns:
        U_hat: Individual parcellations (n_subjects, n_components, n_voxels)
    """
    n_subjects, n_tasks, n_voxels = Y.shape
    _, n_components = V.shape

    U_hat = np.zeros((n_subjects, n_components, n_voxels))

    for subj in range(n_subjects):
        for voxel in range(n_voxels):
            y_voxel = Y[subj, :, voxel]  # Shape: (n_tasks,)
            u_voxel, error = nnls(V, y_voxel)
            U_hat[subj, :, voxel] = u_voxel

    return U_hat


def estimate_Us_NNLS_lasso(Y, V, alpha=0.005, max_iter=50000):
    """
    Estimate the U matrices using NNLS regression with Lasso regularization.

    Args:
        Y: Individual fMRI data (n_subjects, n_tasks, n_voxels)
        V: Functional Profile (n_tasks, n_components)
        alpha: Regularization strength
        max_iter: Maximum number of iterations

    Returns:
        U_hat: Individual parcellations (n_subjects, n_components, n_voxels)
    """

    n_subjects, n_tasks, n_voxels = Y.shape
    _, n_components = V.shape

    U_hat = np.zeros((n_subjects, n_components, n_voxels))

    for subj in range(n_subjects):
        for voxel in range(n_voxels):
            y_voxel = Y[subj, :, voxel]  # Shape: (n_tasks,)

            # Create and fit the Lasso model for each voxel
            lasso = Lasso(alpha=alpha, positive=True)
            lasso.fit(V, y_voxel) 

            # Store weigts
            U_hat[subj, :, voxel] = lasso.coef_
    return U_hat

def estimate_Us_NNLS_ridge(Y, V, alpha=0.1, max_iter=50000):
    """
    Estimate the U matrices using NNLS regression with Ridge regularization (L2).

    Args:
        Y: Individual fMRI data (n_subjects, n_tasks, n_voxels)
        V: Functional Profile (n_tasks, n_components)
        alpha: Regularization strength for Ridge
        max_iter: Maximum number of iterations (currently not used in Ridge)

    Returns:
        U_hat: Individual parcellations (n_subjects, n_components, n_voxels)
    """
    n_subjects, n_tasks, n_voxels = Y.shape
    _, n_components = V.shape

    U_hat = np.zeros((n_subjects, n_components, n_voxels))

    # Loop over each subject and each voxel
    for subj in range(n_subjects):
        for voxel in range(n_voxels):
            y_voxel = Y[subj, :, voxel]  # Shape: (n_tasks,)

            # Create and fit the Ridge model for each voxel
            ridge = Ridge(alpha=alpha)
            ridge.fit(V, y_voxel)

            # Store the weights
            U_hat[subj, :, voxel] = ridge.coef_

    return U_hat