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


def get_data(dataset="cifar10", iid=True):
    if dataset.lower()=="cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    elif dataset.lower() == "fashion_mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    elif dataset.lower()=="lin_reg":
        train_size = 60000
        test_size = 10000
        # Generate some example data
        x_train = torch.randn(train_size, 10)  # 1000 samples, 10 features
        true_weights = torch.arange(10).float() / 10
        y_train = torch.matmul(x_train, true_weights) + 0.5 + 0.05 * torch.randn(train_size)  # Linear relation with noise
        x_test = torch.randn(test_size, 10)  # 1000 samples, 10 features
        y_test = torch.matmul(x_test, true_weights) + 0.5
        # Create TensorDatasets
        train_dataset = TensorDataset(x_train, y_train)
        test_dataset = TensorDataset(x_test, y_test)

    return train_dataset, test_dataset

