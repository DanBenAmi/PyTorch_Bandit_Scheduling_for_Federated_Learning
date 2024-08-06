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
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset.lower() == "lin_reg":
        train_size = 60000
        test_size = 10000
        # Generate some example data
        x_train = torch.randn(train_size, 10) * 0.5 + 0.5  # 1000 samples, 10 features
        true_weights = torch.arange(10).float() / 10
        bias = 0.5
        y_train = torch.matmul(x_train, true_weights) + bias + 0.05 * torch.randn(train_size)  # Linear relation with noise
        x_test = torch.randn(test_size, 10) * 0.5 + 0.5  # 1000 samples, 10 features
        y_test = torch.matmul(x_test, true_weights) + bias
        # Create TensorDatasets
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

    return train_dataset, test_dataset

def add_noise(image, q):
    noise = np.random.normal(loc=0, scale=(1 - q) / 8, size=image.shape)
    noisy_image = image + noise
    if noisy_image.ndim > 2: # images and not vectors..
        noisy_image = np.clip(noisy_image, 0, 1)  # Ensure the values are within [0, 1]
    return torch.Tensor(noisy_image)

def split_data(dataset, n_clients, iid=True, dataset_name='lin_reg'):
    client_datasets = random_split(dataset, [len(dataset) // n_clients] * n_clients)

    if iid:
        qs = [None]*n_clients

    if not iid:
        # quality levels
        qs = [random.uniform(0, 1) for _ in
                        range(n_clients)]  # Random noise levels between 0 and 1 for each client

        for i, client_data in enumerate(client_datasets):
            data_points = torch.stack([data[0] for data in client_data])
            labels = torch.tensor([data[1] for data in client_data])
            noisy_images = add_noise(data_points.numpy(), qs[i])
            client_datasets[i] = TensorDataset(noisy_images, labels)

    return client_datasets, qs







