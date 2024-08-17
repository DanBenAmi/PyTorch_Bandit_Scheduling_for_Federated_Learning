import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# Set font size
fontsize = 13

fig, ax = plt.subplots(1, 4, figsize=(40, 8))

# for j, alpha in enumerate([0, 0.1, 1, 100]):
for j, alpha in enumerate([0.1, 1, 2, 10]):
    with open(
            # f"../results/param_compare/2024-07-30_alpha=[0, 0.1, 0.5, 1, 2, 10, 50, 100]/track_observations_alpha={alpha}.pkl",
            f"../results/param_compare/2024-07-30_beta=[0, 0.1, 0.5, 1, 2, 10, 100]/track_observations_beta={alpha}.pkl",
            'rb') as f:
        data = pickle.load(f)
    data = np.array(data)[::5]

    means = np.linspace(0.2, 0.9, 10)

    # Calculate the mean for each consecutive 50 clients
    mean_data = np.mean(data.reshape(-1, 10, 50), axis=2)  # Shape will be (20, 10)

    # Plot the bar chart
    num_time_steps, num_groups = mean_data.shape

    bar_width = 0.8  # Width of each bar
    opacity = 0.8  # Transparency of each bar

    # Define colors for each time step
    colors = plt.cm.viridis(np.linspace(0, 1, num_time_steps))

    # Plot each time step as a separate bar with overlapping
    for t in range(num_time_steps - 1, -1, -1):
        bars = ax[j].bar(np.arange(num_groups),
                         mean_data[t, :],
                         bar_width,
                         alpha=opacity,
                         color=colors[t],
                         label=f'iteration {(t + 1) * 500}',
                         zorder=num_time_steps - t)

    # Add labels, title, and legend with specified fontsize
    # ax[j].set_xlabel(f'Group mean iter time (alpha={alpha})', fontsize=fontsize)
    ax[j].set_xlabel(f'Group mean iter time (beta={alpha})', fontsize=fontsize)
    ax[j].set_ylabel('Mean Observations Counter', fontsize=fontsize)
    ax[j].set_xticks(np.arange(num_groups))
    ax[j].set_xticklabels([f'{np.arange(0.1, 1.1, 0.1)[i]:.1f} iter\ntime' for i in range(num_groups)],
                          fontsize=fontsize - 1)
    ax[j].tick_params(axis='y', labelsize=fontsize - 1)

    # if j == 0:
    if j == 3:
        ax[j].legend(fontsize=fontsize - 1)

plt.tight_layout()
# plt.savefig("alphas")
plt.savefig("betas")
plt.show(block=True)
