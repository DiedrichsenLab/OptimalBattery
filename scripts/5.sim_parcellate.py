import OptimalBattery.simulate as sim
import OptimalBattery.plot as plot
import HierarchBayesParcel.spatial as spatial
import HierarchBayesParcel.arrangements as ar
import torch as pt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
base_dir = 'Y:/data'

# start with some U_true, in this simulation 5 parcels
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
U_true_5 = pt.tensor(U_true_5,device=device, dtype=pt.float64)


# run analysis
D = sim.sim_parcellation(num_task_lib = 100,
                     n_parcels = 5,
                     U_true = U_true_5,
                     battery_sizes = [3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                     n_batteries = 1000,
                     base_noise = 2,
                     collapsed_U_true = None,
                     n_sim = 100,
                     seed = None)

save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
save_path = os.path.join(save_dir, 'sim_parcellation.tsv')
D.to_csv(save_path, sep='\t', index=False)