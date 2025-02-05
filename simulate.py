"""
Module for function used in the simulation of the task-battery construction problem.
Author: Bassel Arafat
"""


import numpy as np
import matplotlib.pyplot as plt

def random_matrix_normal(G, R, make_exact=False, rng=None):
    n_tasks = G.shape[0]
    n_parcels = R.shape[0]

    if rng is None:
        rng = np.random.default_rng()
    else:
        rng = rng
    V = rng.standard_normal((n_tasks, n_parcels))

    if make_exact:
        P_row = np.linalg.inv(V @ V.T)
        L_row = np.linalg.cholesky(P_row)
        Vs = L_row.T @ V  
    else:
        Vs = V

    lam, eV = np.linalg.eigh(G)
    lam[lam < 1e-15] = 0
    lam = np.sqrt(lam)
    chol_G = eV * lam.reshape((1, eV.shape[1]))

    lam, eV = np.linalg.eigh(R)
    lam[lam < 1e-15] = 0
    lam = np.sqrt(lam)
    chol_R = eV * lam.reshape((1, eV.shape[1]))
    V = chol_G @ Vs @ chol_R.T
    # V =Vs @ chol_R.T

    return V


def find_best_V(G, R, num_iter=1000,rng=None):
    """
    Finds the best V matrix that minimizes the deviation from the desired 
    row and column covariance matrices.
    
    Parameters:
        G (np.ndarray): Desired row covariance matrix.
        R (np.ndarray): Desired column covariance matrix.
        num_iter (int): Number of iterations to try generating V.

    Returns:
        np.ndarray: The V matrix with the lowest deviation from the desired covariances.
    """
    min_deviation = float('inf')
    best_V = None

    for i in range(num_iter):
        # Generate a random V matrix with desired properties
        V = random_matrix_normal(G, R, make_exact=True,rng = rng)
        
        # Compute the row and column covariance matrices of V
        Rs = V @ V.T
        Cs = V.T @ V
        
        # Calculate deviations using nested summations
        dev_R = np.sqrt(np.sum(np.sum((Rs - G) ** 2, axis=1), axis=0))
        dev_C = np.sqrt(np.sum(np.sum((Cs - R) ** 2, axis=1), axis=0))
        
        # Calculate the total deviation
        total_deviation = dev_R + dev_C
        
        # Update best_V if the current total deviation is the lowest found
        if total_deviation < min_deviation:
            min_deviation = total_deviation
            best_V = V


    return best_V


def test_produce_V(): 
    """ Simple test whether matrix normal production works on average. 
    """
    N = 5
    num_iter = 1000
    R = np.random.normal(0,1,(N,N))
    C = np.random.normal(0,1,(N,N))
    cov_R = R @ R.T / N
    cov_C = C @ C.T / N 

    V = np.zeros((num_iter, N, N)) 
    for i in range(num_iter):
        V[i] = random_matrix_normal(cov_R, cov_C, make_exact=True)

    Rs = V@V.transpose([0,2,1])
    Cs = V.transpose([0,2,1])@V
    fig = plt.figure()

    # Plot mean covariance matrices 
    plt.subplot(3,2,1)
    plt.imshow(cov_R)
    plt.title('Row desired')
    plt.colorbar()
    plt.subplot(3,2,2)
    plt.imshow(Rs.mean(axis=0)/N)
    plt.title('Row produced')
    plt.colorbar()
    plt.subplot(3,2,3)
    plt.imshow(cov_C)
    plt.title('Col desired')
    plt.colorbar()
    plt.subplot(3,2,4)
    plt.imshow(Cs.mean(axis=0)/N)
    plt.title('Col produced')
    plt.colorbar()
    # Plot deviation from desired covariance structure
    dev_R = np.sqrt(np.sum(np.sum((Rs - cov_R)**2,axis=2),axis=1))
    dev_C = np.sqrt(np.sum(np.sum((Cs - cov_C)**2,axis=2),axis=1))
    plt.subplot(3,2,5)
    plt.scatter(dev_R,dev_C)
    plt.xlabel('Row deviation')
    plt.ylabel('Col deviation')
    plt.show()
    pass 

def make_U_spatial(grid, centroids, K_main, K_subparcels):
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


if __name__=='__main__':
    test_produce_V()
    pass