import numpy as np
from numpy.linalg import inv,eig,eigh
import PcmPy as pcm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from numpy import sqrt

instr_code = 90 # Instruction condition code

def var_contrasts(X,reg_ind):
    """ Calculate the variance you would get for contrasts on beta-estimates
    from a design matrix X the regressors indicating different conditions
    We do this seperately for all contrast against rest
    and all the pairwise contrasts between conditions.
    """
    conv_beta = inv(X.T@X)
    reg_ind[reg_ind==instr_code] = 0
    # CI are the contrast against rest
    CI = pcm.matrix.indicator(reg_ind,positive=True).T
    CI = CI / CI.sum(axis=1,keepdims=True)
    # CP are the pairwise contrast
    CP = pcm.matrix.pairwise_contrast(reg_ind,positive=True)
    var_i = np.diag(CI@conv_beta@CI.T)
    var_p = np.diag(CP@conv_beta@CP.T)
    return var_i,var_p

def make_design_matrix(reg_id=[[1, 2], [3, 4], [5, 6]],
                       p_rest=0.2,
                       T=100,
                       instruction_TR=0):  # Default: no instruction time
    """Make a design matrix with the conditions indicated in reg_id,
    rest phases with probability p_rest, and optional instruction periods."""
    
    num_part = len(reg_id)  # Number of runs
    part_v = np.kron(np.arange(num_part), np.ones((T,)))  # Run indices
     # Add instruction condition to each run 
    if instruction_TR > 0:
        reg_id = [r + [instr_code] for r in reg_id] 
    reg_ind = np.concatenate(reg_id)

    # Get the partition index for the columns of the design matrix
    part_ind = [np.ones(len(r)) * i for i, r in enumerate(reg_id)]
    part_ind = np.concatenate(part_ind)

    X = np.zeros((T * num_part, reg_ind.shape[0]))
    cv = []
    for i in range(num_part):
        num_cond = len(reg_id[i])
        if instruction_TR > 0:
            num_reg = len(reg_id[i])
            num_cond = num_reg - 1

        # Deduct instruction TRs if instruction_time > 0
        instruction_TRs = instruction_TR * num_cond
        lc = int((T - T * p_rest - instruction_TRs) // num_reg)
        lr = T - lc * num_cond - instruction_TRs

        # Build the condition vector
        if instruction_TR > 0:
            ccvv = np.array(reg_id[i][:-1])  # Task conditions
        else:
            ccvv = np.array(reg_id[i])
        cond_v = np.kron(ccvv, np.ones((lc,)))  # Task periods
        cond_v = np.concatenate((cond_v, np.ones((instruction_TRs,)) * instr_code))  # Instruction periods
        cond_v = np.concatenate((cond_v, np.zeros((lr,))))  # Rest periods
        cv.append(cond_v)

        # Fill the design matrix
        row = np.where(part_v == i)[0]
        col = np.where(part_ind == i)[0]
        X[np.ix_(row, col)] = pcm.matrix.indicator(cond_v, positive=True)

    cond_v = np.concatenate(cv)
    X = np.concatenate([X, pcm.indicator(part_v)], axis=1)
    reg_ind = np.concatenate([reg_ind, np.zeros(num_part)])

    # Use the first condition to get a measure of TR per condition
    lc = (cond_v > 0).sum()  # Count task time
    lr = (cond_v == 0).sum()  # Count rest time
    return X, cond_v, part_v, reg_ind, part_ind, lc, lr




# THis is the covariance matrix for the regressors
# For an interspersed design
design_i=[[1,2,3,4,5,6]]*3 # Intersperse design
X2,_,_,reg2,_,_,_ = make_design_matrix(design_i,instruction_TR=5)
conv_beta = inv(X2.T@X2)
var_i,var_p = var_contrasts(X2,reg2)
