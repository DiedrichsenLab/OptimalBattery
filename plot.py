import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

def plot_correlations(D, x_vars=['max_var'], y_vars=['log_det'], show_p_values=False):
    """
    Plots correlation between multiple x and y variable pairs at each unique offset.
    If there are multiple y_vars, it creates subplots in a row, one per y_var, with all x_vars on the same subplot.
    
    Parameters:
    D : DataFrame
        Data containing the columns for offsets, x variables, and y variables.
    x_vars : list of str
        List of column names to be used as x variables.
    y_vars : list of str
        List of column names to be used as y variables.
    show_p_values : bool
        If True, displays the p-value of each correlation above the plotted point.
    """
    offsets = D['offset'].unique()
    n_y_vars = len(y_vars)
    
    # Create a row of subplots for each y_var
    fig, axs = plt.subplots(1, n_y_vars, figsize=(6 * n_y_vars, 5), sharey=True)
    
    # Ensure axs is iterable even if there's only one y_var
    if n_y_vars == 1:
        axs = [axs]
    
    # Define different line styles for each x_var
    line_styles = ['-', '--', '-.', ':']
    style_map = {x: line_styles[i % len(line_styles)] for i, x in enumerate(x_vars)}
    
    for i, y in enumerate(y_vars):
        ax = axs[i]
        
        for x in x_vars:
            corrs = []
            p_values = []
            for offset in offsets:
                data = D[D['offset'] == offset]
                x_data = data[x]
                y_data = data[y]
                corr, p_value = pearsonr(x_data, y_data)
                corrs.append(corr)
                p_values.append(p_value)
            
            # Plot correlations
            ax.plot(offsets, corrs, label=f'{x} vs {y}', linestyle=style_map[x])
            
            # Optionally add p-value annotations
            if show_p_values:
                for j, (offset, p_val) in enumerate(zip(offsets, p_values)):
                    ax.annotate(f'{p_val:.2e}', (offset, corrs[j]), 
                                textcoords="offset points", xytext=(0, 5), ha='center', 
                                fontsize=8, rotation=90)  # Set rotation to 90 degrees for vertical text
        
        ax.set_xscale('log')
        ax.set_xlabel('offset')
        ax.set_title(f'Correlations with {y}')
    
    axs[0].set_ylabel('correlation')
    fig.suptitle('Correlations at Different Offsets')
    fig.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.tight_layout()
    plt.show()
