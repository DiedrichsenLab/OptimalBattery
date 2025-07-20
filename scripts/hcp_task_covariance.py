
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Functional_Fusion.dataset as ds
import Functional_Fusion.atlas_map as am
import os
import random
from scipy.stats import ttest_ind


def get_covariance_matrices(space = 'fs32k',tasks=None,base_dir = None):
    """ Get tge covariance matrices for the HCP task data.
    Args:
        space (str): The atlas space to use, e.g., 'fs32k'.
        tasks (list): List of tasks to include, if None all tasks are included.
        base_dir (str): Base directory for the dataset, if None uses the default path.
    Returns:
        COV (list): List of covariance matrices for each task.
    """
    atlas,_= am.get_atlas(atlas_str=space)
    HCP_dataset = ds.DataSetHcpTask(base_dir + '/HCP_tfMRI')
    data_hcp , info_hcp= HCP_dataset.get_data(space=space,ses_id='ses-task',type = 'CondHalf')
    data_hcp[np.isnan(data_hcp)] = 0

    num_subj = data_hcp.shape[0]

    COV = []
    infos = []
    if tasks is None:
        tasks = info_hcp['task_name'].unique()
    for task in tasks:
        task_info = info_hcp[info_hcp['task_name'] == task]
        task_data = data_hcp[:,info_hcp['task_name'] == task,:]
        task_cov = np.zeros((num_subj,task_data.shape[1],task_data.shape[1]))
        for i in range(num_subj):
            task_cov[i] = np.cov(task_data[i],rowvar=True)
        COV.append(task_cov)
        infos.append(task_info)
    return COV,tasks,infos

def plot_all_covariances(COV,tasks,infos):
    for i,task in enumerate(tasks):
        plt.subplot(3, 3, i+1)
        cov = np.mean(COV[i],axis=0)
        scale= np.max(cov)
        plot_covariance(cov,task,infos[i],scale)

def plot_covariance(cov,task,info,scale):
    ax = plt.gca()
    plt.imshow(cov,cmap='viridis',vmin=0,vmax=scale)
    plt.title(task)
    N = len(info)
    ax.set_xticks([])
    ax.set_yticks(np.arange(0, N, 1))
    ax.set_yticklabels(info.cond_name)
    # Draw horizontal and vertical lines in middle
    a = np.array([0, N])-0.5
    b = np.array([N, N])/2-0.5
    plt.plot(a,b, color='black', linewidth=1)
    plt.plot(b,a, color='black', linewidth=1)
    pass

def estimate_components(cov,info):
    """ Uses the two halves to estimate the the estimation variance components
    Args:
        cov (np.ndarray): Covariance matrix of shape (N, N)
        info (pd.DataFrame): DataFrame containing information about the conditions
    Returns:
        n (float): Estimated noise component (IID for each condition)
        nb (float): Estimated noise block component
        s (float): Estimated signal component
        c (float): Estimated signal covariance component
    """
    i1 = info.half==1
    i2 = info.half==2
    within_run = (cov[i1,:][:,i1] + cov[i2,:][:,i2])/2
    between_run = (cov[i1,:][:,i2] + cov[i2,:][:,i1])/2
    N = np.sum(i1)

    ondiag = np.where(np.eye(N))
    offdiag = np.where(1-np.eye(N))

    # Average the within-partition and across-partition cross-products for each subject and separate split
    v1 = np.nanmean(within_run[ondiag[0],ondiag[1]])
    v2 = np.nanmean(within_run[offdiag[0],offdiag[1]])
    v3 = np.nanmean(between_run[ondiag[0],ondiag[1]])
    v4 = np.nanmean(between_run[offdiag[0],offdiag[1]])
    n=(v1-v3-(v2-v4))/v1
    nb=(v2-v4)/v1
    s= (v3)/v1
    c= v4/v1
    return n,nb,s,c

def estimate_all_components(COV,tasks,infos):
    df= pd.DataFrame()
    for i,task in enumerate(tasks):
        num_subj = COV[i].shape[0]
        for s in range(num_subj):
            n,nb,s,c = estimate_components(COV[i][s],infos[i])
            d = {'subj':[i],'task':task,'noise':n,'noise_block':nb,'signal':s,'sig_cov':c}
            df = pd.concat([df,pd.DataFrame(data=d)])
    return df

if __name__=='__main__':
    COV, tasks, infos = get_covariance_matrices(space = 'fs32k')
    plot_all_covariances(COV,tasks,infos)
    D = estimate_all_components(COV,tasks,infos)

    # get_covariance_matrices(space = 'fs32k')
    pass