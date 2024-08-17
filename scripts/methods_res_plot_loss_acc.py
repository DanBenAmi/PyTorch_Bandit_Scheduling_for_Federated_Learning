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
        smoothed_acc = moving_average(data['accuracy'][1:], window_size)
        plt.plot(data['time'][:len(smoothed_acc)], smoothed_acc, marker=markers[i], label=f"{cs_name}")
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Time vs. Accuracy', fontsize=16)
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
    # dir_path, param = '../results/param_compare/2024-08-03_21:21_beta=[0.1, 1, 2, 10]', 'beta'  # Replace with your directory path
    # dir_path = '../results/methods_compare/fashion_mnist/2024-08-09_19:07__iid__fashion_mnist__500_25__20t__lr4'  # Replace with your directory path
    # data_dicts = read_pkl_files(dir_path)
    # plot_data(data_dicts)
    # plt.show(block=True)

    # Define the root directory
    root_dir = '../results/methods_compare/fashion_mnist'

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
            if '_iid' not in str(dir_path):
                continue

        try:
            print(dir_path)
            data_dicts = read_pkl_files(dir_path)
            plot_data(data_dicts)
            plt.show(block=True)
            tmp=2
        except:
            print("didn't managed to check dir. moving to next dir.")
