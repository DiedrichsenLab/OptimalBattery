import OptimalBattery.simulate as sim
import os


# Make a random task battery with 100 task
D = sim.sim_connectivity(num_task_lib = 100,
                     n_parcels = 5,
                     n_voxels_y = 100,
                     n_sim = 100,
                     battery_sizes = [3,4,5,6,7,8,9,10,11,12,13,14,15,16],
                     n_batteries = 1000,
                     base_noise = 2,
                     ridge_alpha = 1,
                     seed = None)

save_dir = os.path.abspath(os.path.join(os.getcwd(),'eval_tsvs'))
save_path = os.path.join(save_dir, 'connectivity_sim.tsv')
D.to_csv(save_path, sep='\t', index=False)