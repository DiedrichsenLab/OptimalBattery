import OptimalBattery.simulate as sim
import HierarchBayesParcel.spatial as spatial
import HierarchBayesParcel.arrangements as ar
import torch as pt
import os
import OptimalBattery.plot as plot
import matplotlib.pyplot as plt
import pandas as pd


device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')


# start with some U_true, in this simulation its 5 parcels
height = 30
width = 30
K_main = 5
K_subparcels = 1
K_total = 5

grid = spatial.SpatialGrid(height, width)
arrangeT = ar.ArrangeIndependent(K=5, P=grid.P)
# define centroids more systematically
center_1 = (0, 0)
center_5 = (int((height-1)/2), int((width-1)/2))
center_2 = (width-1,0 )
center_4 = (height-1, width-1)
center_3 = (0, height-1)
centroids = [center_1, center_2, center_3, center_4, center_5]

   
U_true_5 = sim.make_U_spatial(grid, centroids, K_main, K_subparcels)
U_true_5 = pt.from_numpy(U_true_5).to(device=device, dtype=pt.float64)

# make a collapsed U_true where one region is 1 and the rest are 0
collapsed_U_true = sim.collapse_U(U_true_5, target_parcels_indices=[4])



# constants
battery_sizes = [2,4,18] # only for multi
metrics = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc'] # only for multi
thresholds = [0.1,0.5,0.99] # only for single
n_batteries = 1000 # only for multi
num_task_lib = 100 # shared
n_parcels = 5 # shared
base_noise_list = [2,10]  # shared
n_sim = 4  # shared


all_dfs = []
for base_noise in base_noise_list:
    print(f'base noise: {base_noise}')
    # Run multitask simulation
    D_multi = sim.sim_parcellation(
        num_task_lib=num_task_lib,
        n_parcels=n_parcels,
        U_true=U_true_5,
        battery_sizes=battery_sizes,
        n_batteries=n_batteries,
        base_noise=base_noise,
        n_sim=n_sim,
        collapsed_U_true=collapsed_U_true
    )

    D_multi['simulation_type'] = 'multi'
    D_multi['base_noise'] = base_noise
    D_multi['threshold'] = None  # add missing columns
    all_dfs.append(D_multi)

    # Run single contrast simulation
#     D_single = sim.sim_single_contrast(
#         num_task_lib=num_task_lib,
#         n_parcels=n_parcels,
#         U_true=U_true_5,
#         base_noise=base_noise,
#         max_battery_size=max(battery_sizes),
#         thresholds=thresholds,
#         U_true_collapsed=collapsed_U_true,
#         n_sim=n_sim
#     )

#     D_single['simulation_type'] = 'single'
#     D_single['base_noise'] = base_noise
#     D_single['n_task'] = None  # add missing columns
#     D_single['metric'] = None
#     all_dfs.append(D_single)

# # Combine everything
# final_df = pd.concat(all_dfs, ignore_index=True)

# # Save
# save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
# save_path = os.path.join(save_dir, 'sim_localization.tsv')
# final_df.to_csv(save_path, sep='\t', index=False)