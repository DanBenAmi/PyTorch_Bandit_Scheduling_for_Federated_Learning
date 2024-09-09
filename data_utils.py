import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import random
import copy
from tqdm import tqdm


def get_data(dataset="cifar10"):
    if dataset.lower()=="cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset.lower() == "fashion_mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize( (0.5,), (0.5,))])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset.lower() == "lin_reg":
        train_size = 60 #TODO change back the number of data points later
        test_size = 5000
        n_features = 10
        # Generate some example data
        x_train = torch.randn(train_size, n_features)  # 1000 samples, 10 features
        true_weights = torch.randn(n_features)
        bias = 0.5
        noise_level = 0.5
        y_train = torch.matmul(x_train, true_weights) + bias + noise_level * torch.randn(train_size)  # Linear relation with noise TODO change back to noise of 0.05* randn
        x_test = torch.randn(test_size, n_features) # 1000 samples, 10 features
        y_test = torch.matmul(x_test, true_weights) + bias
        # Create TensorDatasets
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

    return train_dataset, test_dataset

def add_noise(data_points, q):
    if data_points.ndim == 2: # lin_reg dataset
        noise = np.random.normal(loc=0, scale=(1 - q) / 2, size=data_points.shape)
    else:
        noise = np.random.normal(loc=0, scale=(1 - q) * 0.2, size=data_points.shape)
    noisy_data_points = data_points + noise
    if noisy_data_points.ndim > 2: # images and not vectors..
        noisy_data_points = np.clip(noisy_data_points, 0, 1)  # Ensure the values are within [0, 1]
    return torch.Tensor(noisy_data_points)


def distribute_datapoints(total_datapoints, n_clients, skewness):
    # Ensure skewness is between 0 and 1
    skewness = max(0, min(skewness, 1))

    # Create an array to hold the data points for each client
    datapoints = np.zeros(n_clients, dtype=int)

    if skewness == 0:
        # Uniform distribution
        datapoints[:] = total_datapoints // n_clients
        for i in range(total_datapoints % n_clients):
            datapoints[i] += 1
    else:
        # Skewed distribution
        # Create a random distribution skewed towards the beginning
        weights = np.exp(-skewness * np.arange(n_clients))
        weights /= weights.sum()

        # Assign data points based on weights
        assigned_points = 0
        for i in range(n_clients):
            if i == n_clients - 1:
                # Assign remaining points to the last client
                datapoints[i] = total_datapoints - assigned_points
            else:
                datapoints[i] = int(np.round(weights[i] * total_datapoints))
                assigned_points += datapoints[i]

    return datapoints.tolist()


def split_data(dataset, n_clients, iid=True, dataset_name='lin_reg', data_sizes=None, qs=None):
    if iid:
        client_datasets = random_split(dataset, [len(dataset) // n_clients] * n_clients)
        qs = [None]*n_clients

    if not iid:
        if not data_sizes:
            data_sizes = np.array(distribute_datapoints(len(dataset), n_clients, 2 / n_clients))
            np.random.shuffle(data_sizes)

        # quality levels
        if not qs:
            qs = [random.uniform(0.7, 1) for _ in range(n_clients)]  # Random noise levels between 0 and 1 for each client

        client_datasets = random_split(dataset, data_sizes)

        # change q
        for i, client_data in enumerate(client_datasets):
            data_points = torch.stack([data[0] for data in client_data])
            labels = torch.tensor([data[1] for data in client_data])
            noisy_images = add_noise(data_points.numpy(), qs[i])
            client_datasets[i] = TensorDataset(noisy_images, labels)

    return client_datasets, qs







