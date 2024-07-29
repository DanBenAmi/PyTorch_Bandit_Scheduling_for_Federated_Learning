import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import random
import copy

# Define the Client class
class Client:
    def __init__(self, client_id, train_data, local_model, device):
        self.client_id = client_id
        self.train_data = train_data
        self.device = device
        self.local_model = local_model


    def train(self, optimizer, criterion, epochs):
        self.local_model.train()
        train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)
        for epoch in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Debug: Print loss values
                print(f"Client {self.client_id}, Epoch {epoch}, Loss: {loss.item()}")


# Define the Federated Learning Simulation class
class FederatedLearning:
    def __init__(self, global_model, clients, device):
        self.global_model = global_model
        self.clients = clients
        self.device = device

    def distribute_model(self):
        global_weights = copy.deepcopy(self.global_model.state_dict())
        return global_weights

    def aggregate_models(self, client_updates, num_selected_clients):
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = client_updates[k] / num_selected_clients
        self.global_model.load_state_dict(global_dict)

    def train(self, rounds, epochs_per_round, fraction=0.1):
        for round in range(rounds):
            selected_clients = random.sample(self.clients, int(fraction * len(self.clients)))
            initial_weights = self.distribute_model()
            num_selected_clients = len(selected_clients)

            # Initialize client_updates with zeros
            client_updates = {k: torch.zeros_like(v).to(self.device) for k, v in initial_weights.items()}
            criterion = nn.CrossEntropyLoss()
            print(f"global before training: {list(self.global_model.parameters())[0][0][0]}")#DEBUG!!

            for client in selected_clients:
                client.local_model.load_state_dict(initial_weights)
                local_optimizer = optim.SGD(client.local_model.parameters(), lr=0.01, momentum=0.9)
                print(f"parameters before training: {list(client.local_model.parameters())[0][0][0]}")#DEBUG!!

                client.train(local_optimizer, criterion, epochs_per_round)

                print(f"parameters after training: {list(client.local_model.parameters())[0][0][0]}")#DEBUG!!
                # Accumulate the updates
                with torch.no_grad():
                    for k, v in client.local_model.state_dict().items():
                        client_updates[k] += v
                time.sleep(0.5) #DEBUG!!
            # Aggregate updates
            self.aggregate_models(client_updates, num_selected_clients)
            print(f"global after training: {list(self.global_model.parameters())[0][0][0]}")#DEBUG!!



# Load and preprocess the dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Split the dataset into smaller datasets for each client
num_clients = 100
client_datasets = random_split(train_dataset, [len(train_dataset) // num_clients] * num_clients)

# Define the global model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
global_model = models.resnet18(pretrained=False, num_classes=10)
local_model = models.resnet18(pretrained=False, num_classes=10).to(device)

# Create clients
clients = [Client(client_id=i, local_model=local_model, train_data=client_datasets[i], device=device) for i in range(num_clients)]

# Create Federated Learning simulation
fl_simulation = FederatedLearning(global_model=global_model, clients=clients, device=device)

# Train the global model using Federated Learning
fl_simulation.train(rounds=10, epochs_per_round=1, fraction=0.03)

# Test the global model
global_model.eval()
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = global_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the global model on the test images: {100 * correct / total}%')
