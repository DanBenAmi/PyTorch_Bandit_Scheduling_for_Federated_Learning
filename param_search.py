import pickle
import time
from datetime import datetime
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

from Linear_regrression.LinearRegressionModel import LinearRegressionModel
from CNN.CNN_Model import CNNModel
from FL import FederatedLearning
from Client_Selection import *
from data_utils import *

Debug = True

def param_search(time_bulks, n_clients, selection_size, cs_inp, param_name, param_vals, dataset="Cifar10"):
    total_time = time_bulks * n_clients // selection_size

    # Load and preprocess the dataset
    train_dataset, test_dataset = get_data(dataset)

    # Split the dataset into smaller datasets for each client
    client_datasets = random_split(train_dataset, [len(train_dataset) // n_clients] * n_clients)

    # Define the global model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if Debug:
        device = "cpu"

    # res dir
    res_dir = os.path.join(f"results/param_compare", f'{datetime.now().strftime("%Y-%m-%d_%H:%M")}_{param_name}={param_vals}')
    os.makedirs(res_dir, exist_ok=True)

    if dataset == 'lin_reg':
        global_model = LinearRegressionModel(input_dim=train_dataset[0][0].size()[0], output_dim=1)
        local_model = LinearRegressionModel(input_dim=train_dataset[0][0].size()[0], output_dim=1).to(device)
    else:
        global_model = CNNModel(input_shape=tuple(train_dataset[0][0].size()), num_classes=10)
        local_model = CNNModel(input_shape=tuple(train_dataset[0][0].size()), num_classes=10).to(device)

    global_weights = copy.deepcopy(global_model.state_dict())

    # Create clients iid
    fast_clients_relation = 0.9  # 0.9
    all_clients_dists = np.concatenate((np.random.uniform(low=[0.7, 0], high=[0.9, 0.05], size=(round(
        n_clients * (1-fast_clients_relation)), 2)), np.random.uniform(low=[0.1, 0], high=[0.2, 0.05], size=(round(
        n_clients * fast_clients_relation), 2))))
    # all_clients_dists = np.stack(
    #     (np.repeat(np.linspace(0.2, 0.9, 10), n_clients // 10), np.ones(n_clients) * 0.1)).transpose(1, 0)
    all_clients = [Client(id=i, local_model=local_model, data=client_datasets[i], mean_std_time=all_clients_dists[i],
                          device=device) for i in range(n_clients)]

    warmup_iters = cs_inp.pop('warmup_iters')
    for param in param_vals:
        res = cs_inp.copy()
        res.update({'total_time': total_time, 'dataset':dataset, 'n_clientsn':n_clients, 'selection_size':selection_size})
        global_model.load_state_dict(global_weights)

        # Create Federated Learning simulation
        fl_simulation = FederatedLearning(global_model=global_model, all_clients=all_clients, test_data=test_dataset, device=device, track_observations=True)

        # client selection method
        cs_inp[param_name] = param
        bsfl = BSFL(all_clients, total_time, n_clients, selection_size, **cs_inp)
        fl_simulation.selection_alg_warmup(bsfl, warmup_iters)

        # Train the global model using Federated Learning
        res.update(fl_simulation.train(selection_size, bsfl, total_time))

        with open(os.path.join(res_dir, f"results_{param_name}={param}.pkl"), 'wb') as f:
            pickle.dump(res, f)

        # with open(os.path.join(res_dir, f"track_observations_{param_name}={param}.pkl"), 'wb') as f:
        #     pickle.dump(fl_simulation.track_observations, f)


if __name__ == "__main__":
    cs_inp = {"alpha":0.1, 'beta':1, 'tau_min':0.1, 'iid':True, 'warmup_iters': 3500}
    param_search(time_bulks=20, n_clients=500, selection_size=25, cs_inp=cs_inp, param_name='alpha', param_vals=[0, 0.1, 1, 100], dataset='cifar10')



