import OptimalBattery.simulate as sim
import torch as pt
import os

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')
# start with some U_true, in this simulation 5 parcels
height = 30
width = 30
K_main = 5

U_true_5 = sim.make_U_spatial(height, width, K_main)
U_true_5 = pt.tensor(U_true_5,device=device, dtype=pt.float64)


# run analysis
D = sim.sim_parcellation(num_task_lib = 100,
                     n_parcels = 5,
                     U_true = U_true_5,
                     battery_sizes = [3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                     n_batteries = 30000,
                     base_noise = 2,
                     collapsed_U_true = None,
                     n_sim = 100,
                     seed = None)

save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
save_path = os.path.join(save_dir, 'parcellation_sim.tsv')
D.to_csv(save_path, sep='\t', index=False)