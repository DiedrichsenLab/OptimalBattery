import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import to_rgb, ListedColormap


def plot_correlations(D, x_vars=['max_var'], y_vars=['log_det'],title = None):
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
            for offset in offsets:
                data = D[D['offset'] == offset]
                x_data = data[x]
                y_data = data[y]
                
                # Calculate correlation if possible
                if len(x_data) > 1 and len(y_data) > 1:
                    corr, _ = pearsonr(x_data, y_data)
                    corrs.append(corr)
                else:
                    corrs.append(np.nan)
            
            # Plot correlations
            ax.plot(offsets, corrs, label=f'{x} vs {y}', linestyle=style_map[x])
        
        ax.set_xscale('log')
        ax.set_xlabel('offset')
        ax.set_title(f'Correlations with {y}')
        
        # Set custom y-axis formatting
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.2f}'))
    
    axs[0].set_ylabel('correlation')
    fig.suptitle('Correlations at Different Offsets')
    fig.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()

    return

def create_custom_colormap(base_colors, K_subparcels):
    """
    Creates a custom colormap by generating shades of base colors.
    
    Parameters:
    base_colors : list of str
        List of color names as base colors (e.g., ['red', 'green']).
    K_subparcels : int
        Number of subparcels used to create gradient shades of each base color.
    
    Returns:
    ListedColormap
        A custom colormap generated from the input base colors.
    """
    cmap_list = []
    for color in base_colors:
        base_rgb = np.array(to_rgb(color))
        for i in range(K_subparcels):
            factor = 0.6 + 0.4 * (i / (K_subparcels - 1))
            shade_rgb = base_rgb * factor + (1 - factor) * np.ones(3)
            cmap_list.append(shade_rgb)

    return ListedColormap(cmap_list)