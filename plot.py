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



def plot_multi_flat(data,overlay_type='label',cscale = None,cmap='gray',colorbar=True,stats='mode',showfigure=True,save=False):
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
    ncols = 4
    nrows = math.ceil(n_subs / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 5))
    axes = axes.flatten()

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
