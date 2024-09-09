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
from torch.utils.tensorboard import SummaryWriter

from Linear_regrression.LinearRegressionModel import LinearRegressionModel
from CNN.CNN_Model import CNNModel
from CNN.FlexibleCNN import FlexibleCNN
from FL import FederatedLearning
from Client_Selection import *
from data_utils import *

Debug = False

def selection_methods_compare(cs_methods, css_args, time_bulks, n_clients, selection_size, dataset_name='cifar10', iid=True, calc_regret=False, lr=0.001):
    total_time = time_bulks * n_clients // selection_size

    # Load and preprocess the dataset
    train_dataset, test_dataset = get_data(dataset_name)

    # Split the dataset into smaller datasets for each client
    client_datasets, qs = split_data(train_dataset, n_clients, iid=iid, dataset_name=dataset_name)


    # Define the global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Debug:
        device = "cpu"

    # res dir
    res_dir = os.path.join(f"results/methods_compare/{dataset_name}", f'{datetime.now().strftime("%Y-%m-%d_%H:%M")}'
        f'_{"" if iid else "non"}_iid__{dataset_name}__{n_clients}_{selection_size}__{time_bulks}t__'
        f'lr{int(-1*np.log10(lr))}')
    os.makedirs(res_dir, exist_ok=True)

    if dataset_name == 'lin_reg':
        global_model = LinearRegressionModel(input_dim=train_dataset[0][0].size()[0], output_dim=1)
        local_model = LinearRegressionModel(input_dim=train_dataset[0][0].size()[0], output_dim=1).to(device)
    else:
        # global_model = CNNModel(input_shape=tuple(train_dataset[0][0].size()), num_classes=10)
        global_model = FlexibleCNN(input_shape=tuple(train_dataset[0][0].size()), num_classes=10)
        # local_model = CNNModel(input_shape=tuple(train_dataset[0][0].size()), num_classes=10).to(device)global_model = CNNModel(input_shape=tuple(train_dataset[0][0].size()), num_classes=10)
        local_model = FlexibleCNN(input_shape=tuple(train_dataset[0][0].size()), num_classes=10).to(device)

    global_weights = copy.deepcopy(global_model.state_dict())

    # Create clients iid
    fast_clients_relation = 0.05  # iid: 0.03      non: 0.1
    slow_clients_relation = 0.2  # iid: 0.2     non: 0.2
    all_clients_dists = np.concatenate((
        np.random.uniform(low=[0.93, 0], high=[0.97, 0.03], size=(round(n_clients * fast_clients_relation), 2)),
        np.random.uniform(low=[0.1, 0], high=[0.12, 0.03], size=(round(n_clients * slow_clients_relation), 2)),
        np.random.uniform(low=[0.7, 0], high=[0.8, 0.03],
                          size=(round(n_clients * (1 - slow_clients_relation - fast_clients_relation)), 2))
    ))
    # all_clients_dists = np.concatenate((
    #     np.random.uniform(low=[0.93, 0.3], high=[0.97, 0.7], size=(round(n_clients * fast_clients_relation), 2)),
    #     np.random.uniform(low=[0.13, 0.5], high=[0.15, 0.7], size=(round(n_clients * slow_clients_relation), 2)),
    #     np.random.uniform(low=[0.51, 0], high=[0.54, 0.03],
    #                       size=(round(n_clients * (1 - slow_clients_relation - fast_clients_relation)), 2))
    # ))
    # all_clients_dists = np.stack(
    #     (np.repeat(np.linspace(0.2, 0.9, 10), n_clients // 10), np.ones(n_clients) * 0.1)).transpose(1, 0)
    all_clients = [Client(id=i, local_model=local_model, data=client_datasets[i], mean_std_rate=all_clients_dists[i],
                          device=device, q=qs[i]) for i in range(n_clients)]

    # save initialization
    # with open(os.path.join(res_dir, f"compare_init.pkl"), 'wb') as f:
    #     pickle.dump({"all_clients": all_clients, "global_weights": global_weights}, f)

    run_dict = {"lr": lr, "calc_regret": calc_regret, "iid": iid, 'total_time': total_time, 'dataset':dataset_name, 'n_clientsn':n_clients, 'selection_size':selection_size, "fast_clients_relation": fast_clients_relation, "slow_clients_relation": slow_clients_relation, **css_args[0]}
    print(datetime.now().strftime("%Y-%m-%d_%H:%M"), "\n", {"lr": lr, "calc_regret": calc_regret, "iid": iid, 'total_time': total_time, 'dataset':dataset_name, 'n_clientsn':n_clients, 'selection_size':selection_size, "fast_clients_relation": fast_clients_relation, "slow_clients_relation": slow_clients_relation}, "\n", css_args[0])
    # bsfl_loss, bsfl_acc = None, None
    alpha, beta = css_args[0]['alpha'], css_args[0]['beta']
    for cs_method, args in zip(cs_methods, css_args):
        tb_dir = os.path.join(res_dir, f"tb_{cs_method}")
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_hparams(run_dict, {})

        res = args.copy()
        res.update({'total_time': total_time, 'dataset':dataset_name, 'n_clientsn':n_clients, 'selection_size':selection_size, "fast_clients_relation": fast_clients_relation, "slow_clients_relation": slow_clients_relation})
        global_model.load_state_dict(global_weights)

        # Create Federated Learning simulation
        fl_simulation = FederatedLearning(global_model=global_model, all_clients=all_clients, test_data=test_dataset,
                                          device=device, track_observations=False, iid=iid, alpha=alpha, beta=beta, tb_writer=writer)
        res.update({"data_sizes": fl_simulation.data_size, "data_quality": fl_simulation.data_quality, "rate_dists": all_clients_dists, "lr": lr})

        # client selection method
        warmup_iters = args.pop('warmup_iters')
        selection_method = cs_method(all_clients, total_time, n_clients, selection_size, iid, **args)
        fl_simulation.selection_alg_warmup(selection_method, iters=warmup_iters)
        warmup_n_observations = selection_method.n_observations.copy()

        # Train the global model using Federated Learning
        res.update(fl_simulation.train(selection_size, selection_method, total_time, calc_regret=calc_regret, warmup_iters=warmup_iters, lr=lr))
        res.update({'n_observations_no_warmup': res['n_observations'] - warmup_n_observations, "warmup_iters": warmup_iters})

        with open(os.path.join(res_dir, f"{selection_method}.pkl"), 'wb') as f:
            pickle.dump(res, f)

        # Early stop
        # if isinstance(selection_method, BSFL):
        #     bsfl_loss, bsfl_acc = res["loss"][-1], res["accuracy"][-1]
        # elif res["loss"][-1] < bsfl_loss or res["accuracy"][-1] > bsfl_acc:
        #     print(f"early stopped because {selection_method} is better in this inputs")


if __name__ == "__main__":
    css = [BSFL, cs_ucb, RBCS_F, PowerOfChoice, Random_Selection]
    css_args = [
        {'warmup_iters': 0, 'beta': 2, 'alpha': 0.1},
        {'warmup_iters': 35000},
        {'warmup_iters': 35000},
        {'warmup_iters': 0},
        {'warmup_iters': 0}
    ]
    iid = True
    dataset_name = 'fashion_mnist'  # 'cifar10' 'lin_reg' 'fashion_mnist'
    time_bulks = 30     # iid: 30 non: 20
    n_clients = 1000
    selection_size = 10
    calc_regret = False
    lr = 5e-5      # iid: 5e-5 non: 2e-6


    selection_methods_compare(css, css_args, time_bulks, n_clients, selection_size, dataset_name=dataset_name, iid=iid, calc_regret=calc_regret, lr=lr)

