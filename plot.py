"""
Module for plotting 
Author: Bassel Arafat
"""

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb, ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns


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


def average_per_subject(df, average_column='correlation'):
    """"
    Averages the specified column per subject and groups by task size and metric."""
    # group by task size and metric
    grouped = df.groupby(['n_task', 'metric'])[average_column]

    result = []
    for (n_task, metric), group in grouped:
        subject_corr_lists = group.tolist()
        corr_array = np.array(subject_corr_lists)
        
        avg_corr_per_subject = np.mean(corr_array, axis=0)
        result.append({
            'n_task': n_task,
            'metric': metric,
            f'avg_{average_column}_per_subject': avg_corr_per_subject.tolist()
        })
    
    return pd.DataFrame(result)