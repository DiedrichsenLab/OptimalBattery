"""
Module for plotting 
Author: Bassel Arafat
"""

import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb, ListedColormap
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import torch as pt
import math
import Functional_Fusion.atlas_map as am
import SUITPy as suit
import fitz  # PyMuPDF
import os
from PIL import Image


def average_per_subject(df, average_column='correlation'):
    """"
    Averages the specified column per subject and groups by task size and metric.
    Args:
        df (pd.DataFrame): DataFrame containing the data to be averaged.
        average_column (str): The column to average. Default is 'correlation'.
    Returns:
        pd.DataFrame: A DataFrame with the average values per subject, grouped by task size and metric.
    """
    # group by task size and metric
    grouped = df.groupby(['n_task', 'metric','roi'])[average_column]

    result = []
    for (n_task, metric, roi), group in grouped:
        if type(group.iloc[0]) == str:
            # Convert string representation of list to actual list
            subject_corr_lists = [ast.literal_eval(item) for item in group.tolist()]
        else:
            # If the group is already a list, no conversion needed
            subject_corr_lists = group.tolist()

        corr_array = np.array(subject_corr_lists)        
        avg_corr_per_subject = np.mean(corr_array, axis=0)
        result.append({
            'n_task': n_task,
            'metric': metric,
            'roi': roi,
            f'avg_{average_column}_per_subject': avg_corr_per_subject.tolist()
        })
    
    return pd.DataFrame(result)

def plot_U_simulation(U,cmap = None,height = None,width = None,title = None):
    """
    plot ground truth parcellation for simulation
    args:
        U: (n_parcels, height*width pixels) tensor of parcellation labels
        cmap: colormap for the plot
        height: height of the plot
        width: width of the plot
        title: title of the plot
    """
    if type(U) == np.ndarray:
        U = pt.tensor(U)
    parcel_labels_plot = U.argmax(dim=0).numpy()
    parcel_labels_plot = parcel_labels_plot.reshape((height, width))
    plt.imshow(parcel_labels_plot, cmap=cmap)
    if title is not None:
        plt.title(title)
    else:
        plt.title('figure')
    return 



def plot_multi_flat(data,overlay_type='label',cscale = None,cmap='gray',colorbar=True,stats='mode',showfigure=True,save=False,single_fig = False):
    """
    Plot multiple flatmaps in a grid layout for multiple subjects.
    Args:
        data (np.ndarray): 2D array of shape (n_subjects, n_vertices) containing the parcellation data for each subject.
        overlay_type (str): Type of overlay to use ('label' or 'func'). label for parcellation, func for functional data.
        cscale (tuple): Color scale for the flatmap. Default is None.
        cmap (str): Colormap to use. Default is 'gray'.
        colorbar (bool): Whether to show colorbar. Default is True. only shows it once for the first plot.
        stats (str): Statistics to use for the flatmap. Default is 'mode'.
        showfigure (bool): Whether to show the figure. Default is True.
        save (bool): Whether to save the figure. Default is False.
    Returns:
        fig (matplotlib.figure.Figure): The figure object containing the flatmaps.
    """
    space = 'SUIT3'
    atlas,_= am.get_atlas(atlas_str=space)

    n_subs = data.shape[0]
    if single_fig:
        ncols = 1
    else:
        ncols = 4
    nrows = math.ceil(n_subs / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    else:
        axes = [axes]

    for i in range(n_subs):
        subject_parcellation = data[i]
        subject_parcellation = atlas.data_to_nifti(subject_parcellation)
        img = suit.flatmap.vol_to_surf(subject_parcellation, space='SUIT', stats=stats, ignore_zeros=False)

        plt.sca(axes[i])
        show_cb = colorbar if i == 0 else False  # Show colorbar only for first
        suit.flatmap.plot(img, overlay_type=overlay_type, cscale=cscale,cmap=cmap, colorbar=show_cb, new_figure=False)
        axes[i].axis('off')  

        # Turn off unused axes
        for j in range(n_subs, len(axes)):
            axes[j].axis('off')

    if save:
        plt.savefig('flatmaps.png', dpi=300, bbox_inches='tight')
    if showfigure:
        plt.show()
    return fig


def save_flatmap_to_pdf(fig, title, pdf):
    """
    Save a flatmap figure to a PDF file with a title.
    Args:
        fig (matplotlib.figure.Figure): The figure object to save.
        title (str): The title for the figure.
        pdf (matplotlib.backends.backend_pdf.PdfPages): The PDF file to save the figure to.
    """
    fig.set_constrained_layout(True)
    fig.suptitle(title, fontsize=14)
    for ax in fig.get_axes():
        ax.set_rasterization_zorder(1)
    pdf.savefig(fig)
    plt.close(fig)


def compress_pdf(pdf_path, dpi=150):
    """
    Compress a PDF file by rendering each page as an image and saving it back to PDF.
    Args:
        pdf_path (str): Path to the PDF file to compress.
        dpi (int): DPI for rendering images. Default is 150.
    """
    doc = fitz.open(pdf_path)
    temp_images = []
    temp_dir = os.path.join(os.path.dirname(pdf_path), "temp_flatten")
    os.makedirs(temp_dir, exist_ok=True)

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)

    # Render each page to an image
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat)
        img_path = os.path.join(temp_dir, f"page_{i}.png")
        pix.save(img_path)
        temp_images.append(img_path)
    doc.close()

    # Recombine images into a single flattened PDF
    image_objs = [Image.open(p).convert("RGB") for p in temp_images]
    if image_objs:
        image_objs[0].save(
            pdf_path,
            save_all=True,
            append_images=image_objs[1:],
        )
    for p in temp_images:
        os.remove(p)
    os.rmdir(temp_dir)

def plot_sim_subject_parcellation(
    i,
    U_individuals_collapsed,
    parcellations_dict,
    results_df,
    grid_width,
    grid_height,
    methods=('contrast_T', 'contrast_percentage', 'multi')
):
    # --- Extract subject info ---
    snr = results_df.loc[results_df["individual"] == i, "snr_factor"].values[0]
    true_map = np.argmax(U_individuals_collapsed[i].cpu().numpy(), axis=0).reshape(grid_width, grid_height)
    true_size = results_df.loc[results_df["individual"] == i, "true_size"].values[0]

    n_methods = len(methods)

    # --- Figure setup ---
    fig = plt.figure(figsize=(8, 6))  
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1.7])  # bottom row taller
    plt.subplots_adjust(hspace=0.45)

    cmap = ListedColormap(["#FFD300","#8c8c8c"])  # grey background, yellow ROI

    # Top row: True ROI centered 
    ax_true = fig.add_subplot(gs[0, 1])
    ax_true.imshow(true_map.T, cmap=cmap, interpolation="nearest")
    ax_true.set_title(
        f"Simulated individual {i}\nTrue ROI  |  SNR = {snr:.3f}  |  Size = {true_size:.0f}",
        fontsize=11, weight='bold'
    )
    ax_true.axis("off")

    # Determine centered columns for methods 
    if n_methods == 1:
        cols = [1]
    elif n_methods == 2:
        cols = [0, 2]
    elif n_methods == 3:
        cols = [0, 1, 2]


    # Bottom row: Predicted maps
    for method, col in zip(methods, cols):
        ax = fig.add_subplot(gs[1, col])
        pred_map = np.argmax(parcellations_dict[method][i], axis=0).reshape(grid_width, grid_height)

        acc = results_df.query("individual == @i and type == @method")["accuracy"].values[0]
        pred_size = results_df.query("individual == @i and type == @method")["predicted_size"].values[0]

        ax.imshow(pred_map.T, cmap=cmap, interpolation="nearest")
        ax.set_title(
            f"{method}\nAcc = {acc:.2f}  |  Size = {pred_size:.0f}",
            fontsize=9
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return fig