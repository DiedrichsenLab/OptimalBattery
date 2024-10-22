import HierarchBayesParcel.arrangements as ar
import numpy as np
import matplotlib.pyplot as plt
from HierarchBayesParcel.util import indicator
import Functional_Fusion.atlas_map as am
import HierarchBayesParcel.emissions as em
import HierarchBayesParcel.full_model as fm
import os
import nitools as nt
from Functional_Fusion.dataset import DataSetMDTB


# Add your functional fusion dir here
BASE_DIR = BASE_DIR = '/cifs/diedrichsen/data/FunctionalFusion'
if not os.path.exists(BASE_DIR):
    BASE_DIR = '/Volumes/diedrichsen_data$/data/FunctionalFusion'

# Load atlas
atlas,_= am.get_atlas(atlas_str='MNISymC2')


# Load group prior
model_name = f'/Atlases/tpl-MNI152NLin2009cSymC/atl-NettekovenSym32_space-MNI152NLin2009cSymC_probseg.nii'
U = atlas.read_data(BASE_DIR + model_name)
U = U.T

# Make an arrangement model (from group prior)
ar_model = ar.build_arrangement_model(U, prior_type='prob', atlas=atlas,
                                        sym_type='sym')

# Load data
MDTB_dataset = DataSetMDTB(BASE_DIR + '/MDTB')
data_mdtb_s1,info_mdtb_1  =MDTB_dataset.get_data(space='MNISymC2',ses_id='ses-s1',type='CondRun')

# record kappas
kappas = []
max_n_tasks = 26 # max number of tasks to include 

for n_tasks in range(3,max_n_tasks+1):
    # Find condition names for the first n_tasks
    unique_conditions = info_mdtb_1['cond_name'].unique()[:n_tasks]

    # Find the indices of the first n_tasks (regressors of these tasks)
    condition_indices = info_mdtb_1[info_mdtb_1['cond_name'].isin(unique_conditions)].index.tolist()

    # index the data
    data_t = data_mdtb_s1[:,condition_indices,:]

    # Create part_vec and cond_vec
    cond_vec = np.tile(np.arange(1, n_tasks + 1), 16)
    part_vec = np.repeat(np.arange(1, 16 + 1), n_tasks)
    x_matrix = indicator(cond_vec)

    # Get n_parcels
    K = ar_model.K_full
    
    # Build emission model and then full model
    em_model = em.MixVMF(K=K, P=atlas.P, X=x_matrix, part_vec=part_vec,
                            subject_specific_kappa=False, parcel_specific_kappa=False, 
                            subjects_equal_weight=True)

    M_1 = fm.FullMultiModel(arrange=ar_model, emission=[em_model])
    M_1.initialize([data_t])

    # Fit the model
    M_1, ll,_,U_individual = M_1.fit_em(iter=200, tol=0.01,
                                        fit_arrangement=False,
                                        fit_emission= True,
                                        first_evidence=False)
    
    # Record kappa
    kappas.append(M_1.emissions[0].kappa)

    # Print Kappa
    print(f'Kappa for {n_tasks} tasks: {M_1.emissions[0].kappa}')
