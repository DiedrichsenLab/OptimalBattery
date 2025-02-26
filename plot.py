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


def expand_subject_results(iter_df):
    """Expands iter_df by separating prediction errors per subject."""
    expanded_results = []
    for _, row in iter_df.iterrows():
        cos_subjects = eval(row["cos_subjects"]) if isinstance(row["cos_subjects"], str) else row["cos_subjects"]
        for i, cos_value in enumerate(cos_subjects):
            expanded_results.append({
                "n_iter": row["n_iter"],
                "n_parcel": row["n_parcel"],
                "n_task": row["n_task"],
                "metric": row["metric"],
                "subject": i + 1,
                "cos_value": cos_value
            })

    return pd.DataFrame(expanded_results)

def resample_df(results_df, metrics, n_proposal_sets):
    """Performs resampling and expands the results for analysis."""
    unique_n_parcel = results_df["n_parcel"].unique()
    unique_n_task = results_df["n_task"].unique()
    proposals_list = []
    
    for n_parcel in unique_n_parcel:
        for n_task in unique_n_task:
            subset_df = results_df[(results_df["n_parcel"] == n_parcel) & (results_df["n_task"] == n_task)]
            for iter in range(n_proposal_sets):
                sampled_df = subset_df.sample(len(subset_df), replace=True)
                for metric in metrics:
                    best_row = sampled_df.loc[sampled_df[metric].idxmax()]
                    if isinstance(best_row, pd.DataFrame):
                        best_row = best_row.iloc[0]
                    proposals_list.append({
                        "n_iter": iter,
                        "n_parcel": n_parcel,
                        "n_task": n_task,
                        "metric": metric,
                        "cos_subjects": best_row["cos_subjects"],
                    })
    
    iter_df = pd.DataFrame(proposals_list)
    return iter_df


def compute_aggregated_results(expanded_results_df):
    """Computes the mean and SEM for each metric."""
    averaged_results_df = expanded_results_df.groupby(["n_parcel", "n_task", "metric", "subject"], as_index=False).agg({"cos_value": "mean"})
    df= averaged_results_df.groupby(["n_parcel", "n_task", "metric"], as_index=False).agg(
        cos_mean=("cos_value", "mean"),
        cos_sem=("cos_value", lambda x: x.std() / np.sqrt(len(x)))
    )
    return df
def compute_baseline(results_df):
    """Computes the baseline average across subjects using precomputed cos_mean."""
    df =  results_df.groupby(["n_parcel", "n_task"], as_index=False).agg(
        cos_mean=("cos_mean", "mean")
    )
    return df


def plot_results(aggregated_results_df, baseline_aggregated_df):
    """Generates line plots comparing metrics and baseline across n_task and n_parcel."""
    unique_n_parcel = aggregated_results_df["n_parcel"].unique()
    num_plots = len(unique_n_parcel)
    n_cols = int(np.ceil(np.sqrt(num_plots))) 
    n_rows = int(np.ceil(num_plots / n_cols)) 
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex=False, sharey=True)
    axes = np.array(axes).flatten()
    
    for ax, n_parcel in zip(axes, unique_n_parcel):
        df_plot = aggregated_results_df[aggregated_results_df["n_parcel"] == n_parcel]
        sns.lineplot(data=df_plot, x="n_task", y="cos_mean", hue="metric", ax=ax, marker="o")
        
        for metric in df_plot["metric"].unique():
            metric_data = df_plot[df_plot["metric"] == metric]
            ax.fill_between(metric_data["n_task"],
                            metric_data["cos_mean"] - metric_data["cos_sem"],
                            metric_data["cos_mean"] + metric_data["cos_sem"],
                            alpha=0.2)
        
        baseline_data = baseline_aggregated_df[baseline_aggregated_df["n_parcel"] == n_parcel]
        sns.lineplot(data=baseline_data, x="n_task", y="cos_mean", ax=ax, color="black", linestyle="dashed", label="Baseline")
        ax.set_title(f"n_parcel = {n_parcel}")
        ax.set_xlabel("Number of Tasks")
        ax.set_ylabel("Average Cos error")
        ax.legend(title="Metric")
    
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()