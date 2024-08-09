import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

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
    # dir_path, param = '../results/param_compare/2024-08-03_21:21_beta=[0.1, 1, 2, 10]', 'beta'  # Replace with your directory path
    dir_path = '../results/methods_compare/2024-08-08_18:08__iid__cifar10__500_25__20t__lr3'  # Replace with your directory path
    data_dicts = read_pkl_files(dir_path)
    plot_data(data_dicts)
    plt.show(block=True)
