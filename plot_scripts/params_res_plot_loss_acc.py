import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def read_pkl_files(dir_path):
    data_dicts = []
    for filename in os.listdir(dir_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'rb') as f:
                data_dicts.append(pickle.load(f))
    return data_dicts

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

def plot_data(data_dicts, window_size=10, param='alpha'):
    plt.figure(figsize=(12, 6))

    # Plot time vs. accuracy
    plt.subplot(1, 2, 1)
    markers = ['o', '+', '*', 'x', 'v']
    for i, data in enumerate(data_dicts):
        smoothed_acc = moving_average(data['accuracy'], window_size)
        plt.plot(data['time'][:len(smoothed_acc)], smoothed_acc, marker=markers[i], label=f"{param}={data[param]}")
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Time vs. Accuracy', fontsize=16)
    plt.legend(fontsize=12)

    # Plot time vs. loss
    plt.subplot(1, 2, 2)
    for i, data in enumerate(data_dicts):
        smoothed_loss = moving_average(data['loss'], window_size)
        plt.plot(data['time'][:len(smoothed_loss)], smoothed_loss, marker=markers[i], label=f"{param}={data[param]}")
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Time vs. Loss', fontsize=16)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # dir_path, param = '../results/param_compare/2024-08-03_21:21_beta=[0.1, 1, 2, 10]', 'beta'  # Replace with your directory path
    dir_path, param = '../results/param_compare/2024-08-03_21:22_alpha=[0, 0.1, 1, 10, 100]', 'alpha'  # Replace with your directory path
    dir_path = '../results/param_compare/2024-08-03_21:22_alpha=[0, 0.1, 1, 10, 100]'  # Replace with your directory path
    data_dicts = read_pkl_files(dir_path)
    plot_data(data_dicts, param=param)
    plt.show(block=True)
