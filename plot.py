import numpy as np
from matplotlib.colors import to_rgb, ListedColormap

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