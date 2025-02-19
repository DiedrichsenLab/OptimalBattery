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


def make_U_basic(s=24, k=16, p=40, type='hard', seed=1):
    """
    Generate the true U matrices for simulation using a normal distribution.
    
    Parameters:
    s: number of subjects
    k: number of parcels
    p: number of voxels
    type: type of Us
    seed: random seed for reproducibility

    Returns:
    Us: ndarray, shape (s, k, p)

    """
    np.random.seed(seed)

    if type == 'hard':
        values = np.random.normal(0, 1, (s, k, p))  # Shape: (s, k, p)
        
        # Find the max indices for each voxel
        max_indices = np.argmax(values, axis=1)  # Shape: (s, p)

        Us = np.zeros((s, k, p))

        for subj in range(s):
            Us[subj][max_indices[subj], np.arange(p)] = 1

    elif type == 'prob':
        # Generate soft assignments using a normal distribution
        Us = np.random.normal(0, 1, (s, k, p))  # Shape: (s, k, p)
        Us = np.exp(Us)  # Apply exponential function
        Us_sum = np.sum(Us, axis=1, keepdims=True)  # Calculate sum along parcel dimension
        Us = Us / Us_sum  # Normalize to get probabilities 

    return Us



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


def center_normalize(X, axis=0):
    """Center and normalize the data along the specified axis, ignoring NaNs."""
    mean = np.nanmean(X, axis=axis, keepdims=True)
    X = X - mean
    norm = np.sqrt(np.nansum(X**2, axis=axis, keepdims=True))
    norm = np.where(norm == 0, 1.0, norm) # needs review
    X = X / norm
    return X

def percentage_correct_real_parcellation(U_true, U_pred):
    """Compute the percentage of correctly classified voxels."""
    percentages = []
    for i in range(U_pred.shape[0]):
        correct_voxels = np.sum(U_true * U_pred[i])
        total_voxels = U_pred.shape[2]
        percentage = (correct_voxels / total_voxels) * 100
        percentages.append(percentage)


    return np.mean(percentages)

def percentage_correct_localization(U_true, U_pred):
    hits = np.sum(U_true * U_pred)
    false_positives = np.sum(U_pred * (1 - U_true))
    percentage = (hits / (hits + false_positives)) * 100
    if np.isnan(percentage):
        percentage = 0
    return percentage



def evaluate_combination_simulation_singleregion(combination,
                                            Ytrue,Vr, Ur,
                                            n_iter=100,
                                            sig_e=0.04,
                                            parcel_to_evaluate = None):
    """Evaluate the localization performance for a single combination of tasks.
    Args:
        combination: The combination of tasks to evaluate
        Ytrue: True tuning functions of all voxels across all tasks (generated using the fine 25 region parcellation)
        Vr: The reduced task matrix for the regions you want to discover
        Ur: The reduced parcellation (correct answer)
        n_iter: Number of iterations to run
        sig_e: Standard deviation of the noise to add to the data
        parcel_to_evaluate: The parcel to evaluate the localization performance for
    """
    Ur = Ur[np.newaxis,:,:]
    # Get the task subset indices and corresponding data
    task_subset_indices = list(combination)

    V_subset = Vr[task_subset_indices,:]
    V_subset = center_normalize(V_subset,axis=0)
    y_subset = Ytrue[task_subset_indices,:]


    perc = np.zeros((n_iter,))
    for i in range(n_iter):
        y = y_subset + np.random.normal(0, sig_e, y_subset.shape)
        y_norm = center_normalize(y,axis=0)

        U_hat = et.estimate_Us_projection(y_norm, V_subset)
        U_hat_one_hot = get_U_hat_one_hot(U_hat)

        Ur_eval = Ur[:,parcel_to_evaluate,:]
        U_hat_one_hot_eval = U_hat_one_hot[:,parcel_to_evaluate,:]
    
        #eval
        perc[i] = percentage_correct_localization(Ur_eval, U_hat_one_hot_eval)

    

    return perc.mean()
    

def evaluate_dataframe_simulation_singleregion(D,
                                                Ytrue,Vr, Ur,
                                                sig_e=1,
                                                parcel_to_evaluate = None):
    """ Evaluate the localization performance for each combination in the DataFrame D.

        Args:
            D: DataFrame containing the combinations to evaluate
            Ytrue: True tuning functions of all voxels across all tasks (generated using the fine 25 region parcellation)
            Vr: The reduced task matrix for the regions you want to discover
            Ur: The reduced parcellation (correct answer)  
            estimation_method: The method to estimate the parcellation
    """

    # Create a new column with combinations as tuples to make them hashable
    D['combination_tuple'] = D['combination'].apply(lambda x: tuple(x)) 
    # Get unique combinations
    unique_combinations = D['combination_tuple'].unique()

    perc_dict= {}
    # Loop over each unique combination
    for i, comb_tuple in enumerate(unique_combinations):
        if i % 1000 == 0:
            print(f"Processing combination: {i}")
        perc = evaluate_combination_simulation_singleregion(comb_tuple,Ytrue,Vr, Ur,sig_e=sig_e,parcel_to_evaluate = parcel_to_evaluate)
        perc_dict[comb_tuple] = perc
    
    # Map the computed cos_HBP values back to the DataFrame
    D['perc'] = D['combination_tuple'].map(perc_dict)    
    return D


def build_combinations(G_lib, strategy='random',n_iter=1000,n_tasks=4,seed=1,balanced_sampling_unique = None): 
    """ Builds a set of task-batteries and evalates them 
    G_lib: second moment matrices of task-library
    strategy: 'random' or 'exhaustive'
    n_iter: number of iterations for random strategy
    """
    np.random.seed(seed)
    D_list = []
    n_lib_task = G_lib.shape[0]

    if strategy == 'random':
        comb = np.array([np.random.choice(n_lib_task, size=n_tasks, replace=True) for _ in range(n_iter)])
    elif strategy == 'exhaustive':
        pass 
    elif strategy == 'balanced':
        comb = set()  
        total_unique = len(balanced_sampling_unique)
        iter_per_unique = n_iter // total_unique
        for n_unique in balanced_sampling_unique:
            for _ in range(iter_per_unique):
                    unique_tasks = np.random.choice(n_lib_task, size=n_unique, replace=False)
                    remaining = n_tasks - n_unique
                    remaining_comb = np.random.choice(unique_tasks, size=remaining, replace=False)
                    full_comb = np.concatenate([unique_tasks, remaining_comb])
                    # sort the combination to avoid duplicates
                    full_comb = np.sort(full_comb)
                    comb.add(tuple(full_comb))
        comb = list(comb)
    else:
        raise ValueError('Invalid strategy')
    for i in range(len(comb)):
        if i % 10000 == 0:
            print(f'building{i}')
        n_unique = len(set(comb[i]))
        d = eigenval_crit(G_lib[comb[i],:][:,comb[i]],center=True)
        d['n_tasks'] = [len(comb[i])]
        d['combination'] = [comb[i]]
        d['n_unique'] = [n_unique]
        D_list.append(pd.DataFrame(d))
    D = pd.concat(D_list)
    return D 


def get_prediction_error_numpy(ytest,vtest,U_hat,indices = None): # old implemntation but was a clear bottle neck in the evaluation framework
    """Compute the prediction error for simulated data.
    Args:
        ytest: Test data
        vtest: Test v vectors
        U_hat: Estimated Us
        indices: The indices of the voxels to evaluate
    return:
        avg_cos: Mean prediction error across subjects
        cos_std: Standard deviation of the prediction error across subjects
    """
    if U_hat.ndim == 2:
        U_hat = U_hat[np.newaxis,:,:]
    if ytest.ndim == 2:
        ytest = ytest[np.newaxis,:,:]

    #normalize vtest
    vtest_normalized = normalize_matrix(vtest,axis = 0)
    # center and normalize ytest
    ytest_centered = center_matrix(ytest,axis = 1)
    ytest_normalized = normalize_matrix(ytest_centered,axis = 1)

    cos_err = np.zeros((U_hat.shape[0],))
    for i in range(U_hat.shape[0]):
        # get reconstructed y
        yhat = np.matmul(vtest_normalized,U_hat[i])
        if indices is not None:
            cosine_error_vox = 1 - np.nansum(ytest_normalized[i][:, indices] * yhat[:, indices], axis=0)
        else:
            cosine_error_vox = 1 - np.nansum(ytest_normalized[i] * yhat, axis=0)
        # mean across voxels within a subject
        cos_err[i] = np.nanmean(cosine_error_vox)
    
    #avg across subjects
    avg_cos = np.nanmean(cos_err)
    # std across subjects
    cos_std = np.nanstd(cos_err)

    return avg_cos, cos_std

def build_battery_dataset(YLib, info, combination, n_repeats=1):
    """
    Constructs a dataset based on a task battery combination.
    
    Parameters:
        YLib (numpy.ndarray): The full data array of shape (subjects, regressors, voxels).
        info (pandas.DataFrame): Information about regressors.
        combination (list): List of indices to include in battery dataset
        n_repeats (int): how much data you want for that combination, default is 1 meaning just make one artificial run of the combination
    
    Returns:
        final_dataset (numpy.ndarray): The constructed task battery dataset of shape (subjects, regressors, voxels).
    """
    # list to include n_repeats of the combination dataset (artificial runs)
    Y_subset_list = []
    total_runs = info['run'].nunique()

    task_groups = info.groupby(['task_num_uni', 'cond_num'])
    task_only_groups = info.groupby(['task_num_uni'])
    
    # to make sure no regressor is chosen twice across n_repeats
    selected_indices = []
    for _ in range(n_repeats):
        Y_subset = []
        
        for idx in combination:
            task_num = info.loc[idx, 'task_num_uni']
            cond_num = info.loc[idx, 'cond_num']
            
           # Get regressor list for the 'task'
            task_group = task_only_groups.get_group(task_num)
            task_indices = task_group.index.tolist()
            
            # Get regressor list for the 'condition of interest'
            condition_group = task_groups.get_group((task_num, cond_num))
            condition_indices = condition_group.index.tolist()
            
            # Determine if it's a task or condition
            num_task_regressors = len(task_indices) // total_runs  # if result is > 1, it's a task with multiple conditions
            
            # Never choose the same regressor twice
            chosen_indices = []
            while len(chosen_indices) < num_task_regressors:
                chosen_idx = condition_indices[pt.randint(len(condition_indices), (1,)).item()]
                if chosen_idx not in selected_indices and chosen_idx not in chosen_indices: #hasn't been chosen for the preview repeat and hasn't been chosen if it's a condition
                    chosen_indices.append(chosen_idx)
                    selected_indices.append(chosen_idx)

            # if it's a 10s condition then average 3x10s to get 30s for example
            averaged_vector = pt.mean(YLib[:,chosen_indices, :], axis=1)
            Y_subset.append(averaged_vector)
        
        # make dataset for current repeat and append to the list
        Y_subset = pt.stack(Y_subset, axis=1)  
        Y_subset_list.append(Y_subset)
    
    # Average repeats
    stacked_Y = pt.stack(Y_subset_list, axis=0)
    final_dataset = pt.mean(stacked_Y, axis=0)
    return final_dataset