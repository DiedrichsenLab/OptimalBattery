import OptimalBattery.simulate as sim
import HierarchBayesParcel.spatial as spatial
import HierarchBayesParcel.arrangements as ar
import torch as pt
import os



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
collapsed_U_true = sim.collapse_U(U_true_5, target_parcel_idx=4)


# constants
battery_sizes = [3,4,5,6,7,8,9,10,11,12,13,14,15,16] # only for multi
metrics = ['random','variance','variance_mc','log_det_mc','inverse_trace_mc'] # only for multi
n_batteries = 1000 # only for multi
num_task_lib = 100 # shared
n_parcels = 5 # shared
base_noise = 2  # shared
n_sim = 50  # shared

# Run the multitask simulation
D_multi = sim.sim_parcellation(num_task_lib = num_task_lib,
                    n_parcels = n_parcels,
                    U_true = U_true_5,
                    battery_sizes = battery_sizes,
                    n_batteries = n_batteries,
                    base_noise = base_noise,
                    n_sim = n_sim,
                    collapsed_U_true=collapsed_U_true)

save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
save_path = os.path.join(save_dir, 'sim_localization_multi.tsv')
D_multi.to_csv(save_path, sep='\t', index=False)



# run the single contrast simulations
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95,0.99]
D_single = sim.sim_single_contrast(num_task_lib = num_task_lib,
                            n_parcels = n_parcels,
                            U_true = U_true_5,
                            base_noise = base_noise,
                            max_battery_size = max(battery_sizes),
                            thresholds = thresholds,
                            U_true_collapsed = collapsed_U_true,
                            n_sim = n_sim)

save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
save_path = os.path.join(save_dir, 'sim_localization_single.tsv')
D_single.to_csv(save_path, sep='\t', index=False)

