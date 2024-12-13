import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

FONTSIZE = 16

def read_pkl_files(dir_path):
    data_dicts = {}
    for filename in os.listdir(dir_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'rb') as f:
                data_dicts[os.path.splitext(filename)[0]] = (pickle.load(f))
    return data_dicts

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_data(data_dicts, window_size=1, save_path=None):
    plt.figure(figsize=(16, 6))

    # Plot time vs. accuracy
    plt.subplot(1, 2, 1)
    markers = ['o', '+', '*', 'x', 'v']
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        smoothed_acc = np.concatenate((data['accuracy'][0:1],moving_average(data['accuracy'][1:], window_size)))
        plt.plot(data['time'][:len(smoothed_acc)], smoothed_acc, marker=markers[i], label=f"{cs_name}")
    plt.xlabel('Time', fontsize=FONTSIZE)
    plt.ylabel('Accuracy', fontsize=FONTSIZE)
    # plt.title('Time vs. Accuracy', fontsize=16)
    plt.legend(fontsize=FONTSIZE)

    # Plot time vs. loss
    plt.subplot(1, 2, 2)
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        smoothed_loss = moving_average(data['loss'], window_size)
        plt.plot(data['time'][:len(smoothed_loss)], smoothed_loss, marker=markers[i], label=f"{cs_name}")
    plt.xlabel('Time', fontsize=FONTSIZE)
    plt.ylabel('Loss', fontsize=FONTSIZE)
    # plt.title('Time vs. Loss', fontsize=16)
    plt.legend(fontsize=FONTSIZE)

    plt.tight_layout()
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    plt.show()


def plot_data_zoom(data_dicts, window_size=1, zoom=False, zoom_xlim_loss=(370, 400), zoom_ylim_loss=(0.3, 0.45), zoom_xlim_acc=(370, 400), zoom_ylim_acc=(0.84, 0.91), zoom_position_acc=[0.8,0.7,1,1.5], zoom_position_loss=[0.8,0.5,1,1.5], save_path=None):
    """
    Plots data with options for zoomed-in sections and saving the plot.

    Parameters:
    - data_dicts: Dictionary containing datasets to plot
    - window_size: Smoothing window size (default: 1, no smoothing)
    - zoom: Whether to include a zoomed-in section (default: False)
    - zoom_xlim, zoom_ylim: Limits for the zoomed-in section (default: None, uses full plot limits)
    - zoom_position: [x, y, width, height] for zoomed-in position
    - save_path: Path to save the plot (default: None, doesn't save)



    """
    global FONTSIZE
    plt.figure(figsize=(16, 6))

    # Plot time vs. accuracy
    plt.subplot(1, 2, 1)
    markers = ['o', '+', '*', 'x', 'v']
    ax1 = plt.gca()  # Main axis for accuracy
    # ax1.set_xticklabels(ax1.get_xticks(), fontsize=FONTSIZE-4)
    # ax1.set_yticklabels(ax1.get_yticks(), fontsize=FONTSIZE-4)
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        smoothed_acc = moving_average(data['accuracy'], window_size)
        plt.plot(data['time'][:len(smoothed_acc)], smoothed_acc, marker=markers[i], label=f"{cs_name}")
    plt.xlabel('Time', fontsize=FONTSIZE)
    plt.ylabel('Accuracy', fontsize=FONTSIZE)
    # plt.title('Time vs. Accuracy', fontsize=16)
    plt.legend(fontsize=FONTSIZE)

    if zoom:
        # Add zoomed-in plot for accuracy
        axins1 = inset_axes(ax1, width=zoom_position_acc[2], height=zoom_position_acc[3],
                            loc='upper right', bbox_to_anchor=zoom_position_acc[:2],
                            bbox_transform=ax1.transAxes)
        for i, (cs_name, data) in enumerate(data_dicts.items()):
            smoothed_acc = moving_average(data['accuracy'], window_size)
            axins1.plot(data['time'][:len(smoothed_acc)], smoothed_acc, marker=markers[i], markersize=5)
        axins1.set_xlim(zoom_xlim_acc or ax1.get_xlim())
        axins1.set_ylim(zoom_ylim_acc or ax1.get_ylim())
        axins1.set_xticks([])
        axins1.set_yticks([])
        if zoom_xlim_acc and zoom_ylim_acc:
            mark_inset(ax1, axins1, loc1=2, loc2=4, ec="0.5")

    # Plot time vs. loss
    plt.subplot(1, 2, 2)
    ax2 = plt.gca()  # Main axis for loss
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        smoothed_loss = moving_average(data['loss'], window_size)
        plt.plot(data['time'][:len(smoothed_loss)], smoothed_loss, marker=markers[i], label=f"{cs_name}")
    plt.xlabel('Time', fontsize=FONTSIZE)
    plt.ylabel('Loss', fontsize=FONTSIZE)
    # plt.title('Time vs. Loss', fontsize=16)
    plt.legend(fontsize=FONTSIZE)

    if zoom:
        # Add zoomed-in plot for loss
        axins2 = inset_axes(ax2, width=zoom_position_loss[2], height=zoom_position_loss[3],
                            loc='upper right', bbox_to_anchor=zoom_position_loss[:2],
                            bbox_transform=ax2.transAxes)
        for i, (cs_name, data) in enumerate(data_dicts.items()):
            smoothed_loss = moving_average(data['loss'], window_size)
            axins2.plot(data['time'][:len(smoothed_loss)], smoothed_loss, marker=markers[i], markersize=5)
        axins2.set_xlim(zoom_xlim_loss or ax2.get_xlim())
        axins2.set_ylim(zoom_ylim_loss or ax2.get_ylim())
        axins2.set_xticks([])
        axins2.set_yticks([])
        if zoom_xlim_loss and zoom_ylim_loss:
            mark_inset(ax2, axins2, loc1=1, loc2=3, ec="0.5")

    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")



if __name__ == '__main__':
    # dir_path, param = '../results/param_compare/2024-08-03_21:21_beta=[0.1, 1, 2, 10]', 'beta'  # Replace with your directory path
    # dir_path = '../results/methods_compare/fashion_mnist/2024-08-09_19:07__iid__fashion_mnist__500_25__20t__lr4'  # Replace with your directory path
    # data_dicts = read_pkl_files(dir_path)
    # plot_data(data_dicts)
    # plt.show(block=True)

    # Define the root directory
    root_dir = '../results/methods_compare/selected_res'

    subdirs = [Path(root_dir) / d for d in os.listdir(root_dir) if os.path.isdir(Path(root_dir) / d)]
    # Sort the subdirectories by creation time
    subdirs_sorted = sorted(subdirs, key=os.path.getctime)[::-1]
    # Loop through each folder in the directory
    for dir_path in subdirs_sorted:

        if os.path.isfile(dir_path):
            continue

        # Ensure it's a directory
        if not os.path.isdir(dir_path):
            continue

        if 'lin_reg' in str(dir_path) or 'iid' not in str(dir_path):
            continue

        try:
            print(dir_path)
            data_dicts = read_pkl_files(dir_path)
            plot_data(data_dicts, save_path=r"../results/methods_compare/selected_res/plots")
            plt.show(block=True)
            tmp=2
        except:
            print("didn't managed to check dir. moving to next dir.")
