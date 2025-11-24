# script for the first part of figure 1 . Single contrast vs multi in simulations.

import OptimalBattery.simulate as sim
import torch as pt
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import OptimalBattery.plot as plot
from scipy.stats import pearsonr
from OptimalBattery.global_config import save_dir,repo_dir

# Start with a base U 
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

grid_height = 30
grid_width = 30

U_true_5 = sim.make_U_spatial(height=grid_height, width=grid_width, K_main= 5)
U_true_5 = pt.from_numpy(U_true_5).to(device)

labels = pt.argmax(U_true_5, axis=0)          
label_map = labels.reshape(grid_height, grid_width)           
plt.imshow(label_map.cpu().numpy())
plt.savefig(f"{save_dir}/single_vs_multi/U_true.pdf")

# make individual Us that vary in about 2 SD in region size
U_individuals = sim.make_U_individuals(U_true_5,grid_width,grid_height, n_individuals=5000,
                            size_range = (120,240),seed=1,device=device)


# make collpased versions of U_individuals (Region of interest and everything else)
U_individuals_collapsed = []
target_indices = [4]
for U_ind in U_individuals:
    U_collapsed = sim.collapse_U(U_ind, target_parcels_indices=target_indices)
    U_individuals_collapsed.append(U_collapsed)

# Generate data-drvien SNR list (variance of the signal from mdtb-1)
snr_list = [0.0196991187488852, 0.0291119097017132, 0.009868315372056239, 0.02666920728512233, 0.012530722513964174, 0.023969897215134296, 0.026436919129805385, 0.019008834666825788, 0.014048736255142842, 0.018222607813836222, 0.03655696258923394, 0.00945854991438873, 0.017826522554411507, 0.016721016465301754, 0.01134745499645742, 0.027609353003180823, 0.019002938368900588, 0.014779159602009994, 0.019332472019233885, 0.012547128382596882, 0.015449975415808167, 0.017610806336780898, 0.009659853650544005, 0.007693769241799343]


results_df, parcellations_single_threshold,parcellations_single_percentage, parcellations_multi = sim.sim_single_vs_multi(
    U_individuals,
    U_individuals_collapsed,
    base_noise=0.055,
    snr_ratios=snr_list,seed=47
)

results_df.to_csv(f"{repo_dir}/eval_tsvs/localization_sim_results.tsv", sep="\t", index=False)


# plot individuals with high/low SNR and small/large ROIs
# Selected individuals:
# lowSNR_normSize: subj=1164, SNR=0.0074, size=180.0      
# highSNR_normSize: subj=2296, SNR=0.0334, size=180.0     
# normSNR_smallROI: subj=1414, SNR=0.0170, size=122.0     
# normSNR_largeROI: subj=3122, SNR=0.0171, size=238.0

# Pack parcellations into a dict
# parcellations = {
#     "contrast_T": parcellations_single_threshold,
#     "contrast_percentage": parcellations_single_percentage,
#     "multi": parcellations_multi
# }

# fig = plot.plot_sim_subject_parcellation(
#     i=1414,
#     U_individuals_collapsed=U_individuals_collapsed,
#     parcellations_dict=parcellations,
#     results_df=results_df,
#     grid_width=grid_width,
#     grid_height=grid_height,
#     methods=('contrast_percentage', 'multi')
# )
# # save as pdf
# fig.savefig(f"{save_dir}/supp/single_vs_multi_sim_plots/simulated_subject_normSNR_smallsize.pdf", bbox_inches='tight')


# fig = plot.plot_sim_subject_parcellation(
#     i=3122,
#     U_individuals_collapsed=U_individuals_collapsed,
#     parcellations_dict=parcellations,
#     results_df=results_df,
#     grid_width=grid_width,
#     grid_height=grid_height,
#     methods=('contrast_percentage', 'multi')
# )
# fig.savefig(f"{save_dir}/supp/single_vs_multi_sim_plots/simulated_subject_normSNR_highsize.pdf", bbox_inches='tight')

# fig = plot.plot_sim_subject_parcellation(
#     i=1164,
#     U_individuals_collapsed=U_individuals_collapsed,
#     parcellations_dict=parcellations,
#     results_df=results_df,
#     grid_width=grid_width,
#     grid_height=grid_height,
#     methods=('contrast_T', 'multi')
# )
# fig.savefig(f"{save_dir}/supp/single_vs_multi_sim_plots/simulated_subject_lowSNR_normsize.pdf", bbox_inches='tight')


# fig = plot.plot_sim_subject_parcellation(
#     i=2296,
#     U_individuals_collapsed=U_individuals_collapsed,
#     parcellations_dict=parcellations,
#     results_df=results_df,
#     grid_width=grid_width,
#     grid_height=grid_height,
#     methods=('contrast_T', 'multi')
# )
# fig.savefig(f"{save_dir}/supp/single_vs_multi_sim_plots/simulated_subject_highSNR_normsize.pdf", bbox_inches='tight')