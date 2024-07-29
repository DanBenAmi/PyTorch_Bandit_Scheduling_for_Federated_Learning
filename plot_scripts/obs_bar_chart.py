import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

with open("../results/param_compare/2024-07-29_alpha=[0, 0.5, 1, 2, 10]/track_observations_alpha=0.pkl", 'rb') as f:
    data = pickle.load(f)
data = np.array(data)

means = np.linspace(0.2,0.9,10)

# Step 2: Calculate the mean for each consecutive 50 clients
mean_data = np.mean(data.reshape(-1, 10, 50), axis=2)  # Shape will be (20, 10)

# Step 3: Plot the bar chart
num_time_steps, num_groups = mean_data.shape
fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.8  # Width of each bar
opacity = 0.8    # Transparency of each bar

# Define colors for each time step
colors = plt.cm.viridis(np.linspace(0, 1, num_time_steps))

# Plot each time step as a separate bar with overlapping
for t in range(num_time_steps-1,-1,-1):
    bars = ax.bar(np.arange(num_groups),
                  mean_data[t, :],
                  bar_width,
                  alpha=opacity,
                  color=colors[t],
                  label=f'iteration {(t+1)*500}',
                  zorder=num_time_steps-t)

# Add labels, title, and legend
ax.set_xlabel('Groups of 50 iid Clients')
ax.set_ylabel('Mean Observations Counter')
# ax.set_title('Mean Observations Counters of 50 Clients for Each 500 iterations')
ax.set_xticks(np.arange(num_groups))
ax.set_xticklabels([f'Group {i+1} - {means[i]} iter mean time' for i in range(num_groups)])
ax.legend()

plt.show()
