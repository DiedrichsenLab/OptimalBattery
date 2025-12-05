from IndividualParcellation.global_config import *
import numpy as np
import torch as pt
import nitools as nt
import seaborn as sns
import matplotlib.pyplot as plt
import nilearn.plotting as plotting
from Functional_Fusion.dataset import DataSetMDTB
import Functional_Fusion.atlas_map as am
import PcmPy as pcm
import OptimalBattery.construct as ct
from OptimalBattery.global_config import data_dir,save_dir

# Constants
device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
space = 'fs32k'
atlas,_= am.get_atlas(atlas_str=space)
base_dir = f'{data_dir}/FunctionalFusion_new'
cort_dir = f'{data_dir}/Atlas_templates/fs_LR_32'

#Load data
MDTB_dataset = DataSetMDTB(f'{base_dir}/MDTB')
data_mdtb_s1,info_mdtb_1  =MDTB_dataset.get_data(space=space,ses_id='ses-s1',type='CondRun')
data_mdtb_s1[np.isnan(data_mdtb_s1)] = 0
task_names = info_mdtb_1.cond_name.values
nconds = info_mdtb_1['cond_name'].nunique()
nparts = info_mdtb_1['run'].nunique()
cond_names = info_mdtb_1[:29].cond_name.values

# Load surface files for left and right hemispheres
surfs = [f"{base_dir}/Atlases/tpl-fs32k/tpl-fs32k_hemi-{h}_inflated.surf.gii" for h in ['L', 'R']]
def plot_cortex(data, threshold=0.0, cmap='binary', figsize=(12, 6),title = 'figure'):  
    # Convert data to CIFTI format
    cifti = atlas.data_to_cifti(data)
    
    # Extract data for the cortical surfaces
    all_img = nt.surf_from_cifti(cifti)
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=figsize)
    for h, hemi in enumerate(['left', 'right']):
        plotting.plot_surf_stat_map(
            surfs[h], all_img[h], hemi=hemi,
            colorbar=False,
            cmap=cmap,
            axes=axes[h],
            threshold=threshold,
            title=f'{title} {hemi}',
        )
    
    return fig

# First covariance structure across human lobes
atlas_dir = f'{base_dir}/Atlases/tpl-fs32k'
model_name_L = f'{atlas_dir}/HumanLobes.L.label.gii'
model_name_R = f'{atlas_dir}/HumanLobes.R.label.gii'
parcels = atlas.read_data([model_name_L,model_name_R])

ROI_frontal = np.isin(parcels, [1]).astype(int)[np.newaxis, :]
ROI_parietal = np.isin(parcels, [2]).astype(int)[np.newaxis, :]
ROI_occipital = np.isin(parcels, [5]).astype(int)[np.newaxis, :]
ROI_temporal = np.isin(parcels, [4]).astype(int)[np.newaxis, :]



data_mdtb_s1_frontal = data_mdtb_s1[:,:,:] * ROI_frontal
G_frontal = ct.get_G(data_mdtb_s1_frontal,nconds,nparts)

data_mdtb_s1_parietal = data_mdtb_s1[:,:,:] * ROI_parietal
G_parietal = ct.get_G(data_mdtb_s1_parietal,nconds,nparts)

data_mdtb_s1_occipital = data_mdtb_s1[:,:,:] * ROI_occipital
G_occipital = ct.get_G(data_mdtb_s1_occipital,nconds,nparts)

data_mdtb_s1_temporal = data_mdtb_s1[:,:,:] * ROI_temporal
G_temporal = ct.get_G(data_mdtb_s1_temporal,nconds,nparts)


# Motor vs prefrontal cortex
atlas_dir = f'{base_dir}/Atlases/tpl-fs32k'
model_name_L = f'{atlas_dir}/glasser.L.label.gii'
model_name_R = f'{atlas_dir}/glasser.R.label.gii'
parcels = atlas.read_data([model_name_L,model_name_R])

ROI_motor = np.isin(parcels, [53,9,8,39,36,37,96,54,44,55,78,56,42]).astype(int)[np.newaxis, :]
ROI_PFC = np.isin(parcels, [74,81,82,83,84,86]).astype(int)[np.newaxis, :]

data_mdtb_s1_motor = data_mdtb_s1[:,:,:] * ROI_motor
G_motor = ct.get_G(data_mdtb_s1_motor,nconds,nparts)

data_mdtb_s1_PFC = data_mdtb_s1[:,:,:] * ROI_PFC
G_PFC = ct.get_G(data_mdtb_s1_PFC,nconds,nparts)


cov_mats = [
    [G_frontal, G_parietal, G_occipital, G_temporal], 
    [G_motor,   G_PFC] ]                                 

titles = [
    ["Frontal", "Parietal", "Occipital", "Temporal"],
    ["Motor Cortex", "Prefrontal Cortex"]
]

nrows = 2
ncols = 4

fig, axes = plt.subplots(
    nrows, ncols, 
    figsize=(4*ncols, 4*nrows),
    squeeze=False
)

for r in range(nrows):
    for c in range(ncols):
        ax = axes[r, c]

        if c >= len(cov_mats[r]):
            ax.axis("off")
            continue

        G = cov_mats[r][c]
        title = titles[r][c]

        im = ax.imshow(G, cmap="viridis")
        ax.set_title(title, fontsize=14)

        # Tick labels (task names)
        ax.set_xticks(np.arange(len(cond_names)))
        ax.set_yticks(np.arange(len(cond_names)))
        ax.set_xticklabels(cond_names, rotation=90, fontsize=6)
        ax.set_yticklabels(cond_names, fontsize=6)

        # Colorbar
        plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig(f"{save_dir}/supp/region_covs.pdf")
plt.show()