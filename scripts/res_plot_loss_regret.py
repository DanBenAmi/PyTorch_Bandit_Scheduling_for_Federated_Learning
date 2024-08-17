import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

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

def plot_data(data_dicts, window_size=1):
    plt.figure(figsize=(12, 6))

    # Plot time vs. accuracy
    plt.subplot(1, 2, 1)
    markers = ['o', '+', '*', 'x', 'v']
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        smoothed_reg = moving_average(np.cumsum(data['regret'][1:]), window_size)
        plt.plot(range(len(smoothed_reg)), smoothed_reg, marker=markers[i], label=f"{cs_name}")
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Regret', fontsize=14)
    plt.title('Time vs. Regret', fontsize=16)
    plt.legend(fontsize=12)

    # Plot time vs. loss
    plt.subplot(1, 2, 2)
    for i, (cs_name, data) in enumerate(data_dicts.items()):
        smoothed_loss = moving_average(data['loss'], window_size)
        plt.plot(data['time'][:len(smoothed_loss)], smoothed_loss, marker=markers[i], label=f"{cs_name}")
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Time vs. Loss', fontsize=16)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # # Get a list of all subdirectories in the root directory
    # root_dir = r'../results/methods_compare/lin_reg'
    # subdirs = [Path(root_dir) / d for d in os.listdir(root_dir) if os.path.isdir(Path(root_dir) / d)]
    # # Sort the subdirectories by creation time
    # subdirs_sorted = sorted(subdirs, key=os.path.getctime)
    # # Get the last subdirectory created
    # dir_path = subdirs_sorted[-4]
    #
    # # dir_path = r'../results/param_compare/lin_reg/'  # Replace with your directory path
    # print(dir_path)
    # data_dicts = read_pkl_files(dir_path)
    # plot_data(data_dicts)
    # plt.show(block=True)

    # # Define the root directory
    root_dir = r'../results/methods_compare/lin_reg'
    subdirs = [Path(root_dir) / d for d in os.listdir(root_dir) if os.path.isdir(Path(root_dir) / d)]
    # Sort the subdirectories by creation time
    subdirs_sorted = sorted(subdirs, key=os.path.getctime)[::-1]
    # Loop through each folder in the directory
    for dir_path in subdirs_sorted:
        if os.path.isfile(dir_path):
            continue

        # Ensure it's a directory
        if os.path.isdir(dir_path):
            # Extract the parameter (alpha or beta) from the folder name
            if '__iid' not in str(dir_path):
                continue

        try:
            print(dir_path)
            data_dicts = read_pkl_files(dir_path)
            plot_data(data_dicts)
            plt.show(block=False)
            tmp = 0
            plt.show(block=True)
        except:
            print("didn't managed to check dir. moving to next dir.")




    tmp = 3

