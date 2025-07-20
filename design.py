import numpy as np
from numpy.linalg import inv,eig,eigh
import PcmPy as pcm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from numpy import sqrt
import os
from matplotlib import gridspec

instr_code = 90 # Instruction condition code

def make_design_matrix_simple(reg_id=[[1, 2], [3, 4], [5, 6]],
                       p_rest=0.2,
                       T=100,
                       instruction_TR=0):  # Default: no instruction time
    """Make a design matrix with the conditions indicated in reg_id,
    rest phases with probability p_rest, and optional instruction periods.
    Args:
        reg_id (list): List of lists, where each sublist contains the condition codes for each run.
        p_rest (float): Proportion of run dedicated to rest periods.
        T (int): Total number of time points (s) in each run
        instruction_TR (int): Number of time points for the instruction period, default is 0 (no instruction).
    Returns:
        X (np.ndarray): Design matrix of shape (T * num_part, num_reg)
        cond_v (np.ndarray): Condition vector of shape (T * num_part,)
        part_v (np.ndarray): Partition vector indicating the run for each time point.
        reg_ind (np.ndarray): Regressor index for the columns of the design matrix.
        part_ind (np.ndarray): Partition index for the columns of the design matrix.
        lc (int): Length of task periods.
        lr (int): Length of rest periods."""

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
        lc = int((T - T * p_rest - instruction_TRs) // num_cond)
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

def make_design_matrix(reg_id=[1, 2, 3, 4, 5, 6] * 3,
                       p_rest=0.2,
                       num_rest = 1,
                       T=100,
                       instruction_TR=0,
                       randomize_order=False):  # Default: no instruction time
    """Make a design matrix with the conditions indicated in reg_id,
    rest phases with probability p_rest, and optional instruction periods.
    This function generates temporally more realistic design matrices, so we can look at
    hrf and temporal noise properties.
    """

    num_part = len(reg_id)  # Number of runs
    part_v = np.kron(np.arange(num_part), np.ones((T,)))  # Run indices

    # Make the regressor index for the columns of the design matrix
    if instruction_TR > 0:
        reg_ind_list = [np.unique(r + [instr_code])  for r in reg_id]
    else:
        reg_ind_list = [np.unique(r) for r in reg_id]
    reg_ind = np.concatenate(reg_ind_list)

    # Make the partition index for the columns of the design matrix
    part_ind = [np.ones(len(r)) * i for i, r in enumerate(reg_ind_list)]
    part_ind = np.concatenate(part_ind)

    # Build the design matrix
    X = np.zeros((T * num_part, reg_ind.shape[0]))
    cond_v = []
    for i in range(num_part):
        num_cond = len(reg_id[i])
        condv = np.array(reg_id[i]+[0]*num_rest)  # These are the conditions for this run
        if randomize_order:
            np.random.shuffle(condv)
        instruction_TRs = instruction_TR * len(condv)  # Number of instruction TRs

        # Deduct instruction TRs if instruction_time > 0
        len_task = int((T - T * p_rest - instruction_TRs) // num_cond) # Length of a task period
        len_rest = int(T - len_task * num_cond - instruction_TRs)//num_rest # Length of a rest period

        # Build the condition vector:
        cv = np.zeros((T,))  # Initialize condition vector
        start = 0  # Start index for the first condition
        for j,c in enumerate(condv):  # Loop over conditions
            cv[start:start+instruction_TR] = instr_code  # Task conditions
            if c == 0: # If this is a rest condition
                start += instruction_TR + len_rest
            else:
                cv[start+instruction_TR:start+instruction_TR+len_task] = c
                start += instruction_TR + len_task
        cond_v.append(cv)

        # Fill the design matrix
        row = np.where(part_v == i)[0]
        col = np.where(part_ind == i)[0]
        X[np.ix_(row, col)] = pcm.matrix.indicator(cv, positive=True)

    cond_v = np.concatenate(cond_v)
    # Concetenate the design matrix and add intercept
    X = np.concatenate([X, pcm.indicator(part_v)], axis=1)
    reg_ind = np.concatenate([reg_ind, np.zeros(num_part)])

    # Use the first condition to get a measure of TR per condition
    lc = (cond_v > 0).sum()  # Count task time
    lr = (cond_v == 0).sum()  # Count rest time
    return X, cond_v, part_v, reg_ind, part_ind, lc, lr



def var_contrasts(X,reg_ind):
    """ Calculate the variance you would get for contrasts on beta-estimates
    from a design matrix X the regressors indicating different conditions
    We do this seperately for all contrast against rest
    and all the pairwise contrasts between conditions.
    Args:
        X (np.ndarray): Design matrix of shape (T * num_part, num_reg)
        reg_ind (np.ndarray): Regressor index for the columns of the design matrix.
    Returns:
        var_i (np.ndarray): Variance of the contrasts against rest.
        var_p (np.ndarray): Variance of the pairwise contrasts.
        sig_e: Estimated noise variance unique to conditions
        sig_b: Estimated noise variance shared between conditions within run (baseline)
    """
    cov_beta = inv(X.T@X)
    reg_ind[reg_ind==instr_code] = 0
    # CI are the contrast against rest
    CI = pcm.matrix.indicator(reg_ind,positive=True).T
    CI = CI / CI.sum(axis=1,keepdims=True)
    # CP are the pairwise contrast
    CP = pcm.matrix.pairwise_contrast(reg_ind,positive=True)
    # Variance of the different contrasts
    var_i = np.diag(CI@cov_beta@CI.T)
    var_p = np.diag(CP@cov_beta@CP.T)
    # Noise variance and covariance in the run can just be read
    # off, as no signal is present, and cross-run-covariance is zero
    sig_b = cov_beta[0,1]
    sig_e = cov_beta[0,0] - sig_b  # Noise variance unique
    return var_i,var_p,sig_e,sig_b

def compare_designs_1(design=None,
                    T=300,
                    p_rest=[0.7,0.6,1/2,1/2.5,1/3,1/4,1/5,1/6,1/7,1/8,1/9,1/11,1/15],
                    contrast_in=None,
                    instruction_time=5):
    """Compare blocked and interspersed designs with optional instruction times.
    Args:
        design (list): List of designs () to compare, if None uses default designs."""
    if design is None:
        design = [[[1, 2,1,2], [3, 4,3,4]],  # Blocked design
                  [[1, 2, 3, 4]] * 2]  # Interspersed design
    # get unique indices from one of the designs
    n_unique = len(np.unique(np.concatenate(design[0])))
    n_contrasts = n_unique*(n_unique-1)//2

    ci = np.zeros(n_contrasts,dtype=bool)
    ci[[0,5]]=True # THese are the within-run contrast

    DF = pd.DataFrame()
    for rp in p_rest:
        for i, d in enumerate(design):

            # Build design matrix
            instr = instruction_time if i == 1 else 0  # Only add instructions to interspersed
            X, _, _, reg_ind, part_ind, lc, lr = make_design_matrix(d, rp, num_rest=1, T=T, instruction_TR=instr, randomize_order=True)

            # If not given, determine the within / between contrasts from the first design
            if contrast_in is None:
                CP = pcm.matrix.pairwise_contrast(reg_ind,positive=True)
                contrast_in = np.ones(CP.shape[0],dtype=bool)
                for c in range(CP.shape[0]):
                    for p in np.unique(part_ind):
                        if np.sum(CP[c,np.where(part_ind==p)[0]]) != 0:
                            contrast_in[c] = False
                            break
            # Computer variance of contrasts
            vari, varp, sig_e, sig_b = var_contrasts(X, reg_ind)
            if i == 0:
                design_name = 'Blocked'
            else:
                design_name = 'Interspersed'
            df = {'Design':design_name,
                  'length_rest':[lr],
                  'p_rest':[lr/(T*len(d))],
                  'length_cond':[lc],
                  'num_cond':n_unique,
                  'Task_Vs_Rest':sqrt(vari.mean()),
                  'std_pairwise':sqrt(varp.mean()),
                  '2Task_Within':sqrt(varp[contrast_in].mean()),
                  '2Task_Between':sqrt(varp[~contrast_in].mean()),
                  'sigma2_b':sig_b,
                  'sigma2_e':sig_e}
            DF = pd.concat([DF,pd.DataFrame(df)],ignore_index=True)
    return DF

def draw_reference_lines(dir,part_v,color='k',lw=1):
    """Draw vertical lines at the start of each run."""
    lines = np.where(part_v[1:]- part_v[:-1])[0]  # Get the difference between runs
    lines = np.append(lines, part_v.shape[0]-1)  # Add the last line
    for l in lines:
        if dir == 'hor':
            plt.axhline(l+0.5, color=color, lw=lw)
        elif dir == 'ver':
            plt.axvline(l+0.5, color=color, lw=lw)

def plot_contrast_variance(DF):
    """ Plots the variance of contrasts for different designs, as function of the proportion of rest."""
    contrast_styles = {
        'Task_Vs_Rest': '-',
        '2Task_Within': '--',
        '2Task_Between': ':'
    }

    # diff colors for diff designs
    design_colors = sb.color_palette("tab10", len(DF['Design'].unique()))

    for i, design_name in enumerate(DF['Design'].unique()):
        subset = DF[DF['Design'] == design_name]
        for j, (contrast, style) in enumerate(contrast_styles.items()):
            sb.lineplot(
                data=subset,
                x='p_rest',
                y=subset[contrast] + (i * 0.003) + (j * 0.001), # Offset for visibility
                label=f"{design_name} - {contrast}",
                linestyle=style,
                color=design_colors[i]  )

    # add vertical lines for rest prob = task prob
    plt.axvline(1/(DF.num_cond[0]+1), color='black', linestyle=':')
    # plt.axvline(1/3, color='black', linestyle=':')

    plt.xlabel('Proportion of Rest', fontsize=14)
    plt.ylabel('Standard Deviation', fontsize=14)
    plt.title('Design Comparison', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)

    #adjust spacing on the x to be less
    plt.xticks(np.arange(0.1, 0.71, step=0.1))

    sb.despine()
    plt.tight_layout()

def plot_baseline_noise(DF):
    """ Plots a line for each design for the baseline / (baseline + cond)"""

    # diff colors for diff designs
    design_colors = sb.color_palette("tab10", len(DF['Design'].unique()))

    for i, design_name in enumerate(DF['Design'].unique()):
        subset = DF[DF['Design'] == design_name]
        sb.lineplot(
            data=subset,
            x='p_rest',
            y=subset.sigma2_b / (subset.sigma2_b+subset.sigma2_e), # Offset for visibility
            color=design_colors[i]  )

    # add vertical lines for rest prob = task prob
    plt.axvline(1/(DF.num_cond[0]+1), color='black', linestyle=':')

    plt.xlabel('Proportion of Rest', fontsize=14)
    plt.ylabel('baseline', fontsize=14)

    #adjust spacing on the x to be less
    plt.xticks(np.arange(0.1, 0.71, step=0.1))
    sb.despine()
    plt.tight_layout()


def example_designs():
    design_b = [[1, 2,1,2,1,2], [3,4,3,4,3,4], [5, 6,5,6,5,6]]  # Blocked design
    design_i = [[1, 2, 3, 4, 5, 6]] * 3  # Interspersed design
    design = [design_b, design_i]
    title = ['Blocked','Interspersed']
    plt.figure(figsize=(10, 5))
    for i in range(2):
        X, cond_v, part_v, reg_ind, part_ind, lc, lr = make_design_matrix(design[i],
                                                                        p_rest=0.2,
                                                                        T=300,
                                                                        num_rest=2,
                                                                        instruction_TR=5,
                                                                        randomize_order=True)
        plt.subplot(1, 2, i + 1)
        plt.title('Design %d' % (i + 1))
        plt.imshow(X,aspect='auto')
        plt.imshow(X,aspect='auto',interpolation='none')
        draw_reference_lines('hor', part_v, color='k', lw=1)
        draw_reference_lines('ver', part_ind, color='k', lw=1)

        pass

def generate_figure():
    design_b=[[1,2],[3,4]] # Blocked design
    design_i=[[1,2,3,4]]*2 # Intersperse design
    T=300
    numpart = len(design_b)
    DF = compare_designs_1([design_b,design_i],T=T, instruction_time=0)
    plt.figure(figsize=(6,7))
    gs = gridspec.GridSpec(4, 1)
    ax0 = plt.subplot(gs[0:3,0])
    plot_contrast_variance(DF)
    ax1 = plt.subplot(gs[3,0])
    plot_baseline_noise(DF)

    pass

if __name__ == '__main__':
    generate_figure()
    plt.show()