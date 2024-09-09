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

def selection_methods_compare(cs_methods, css_args, time_bulks, n_clients, selection_size, dataset_name='cifar10',
                              iid=True, calc_regret=False, lr=0.001, fast_clients_relation=None, slow_clients_relation=None, mid_clients_mean=None ):
    total_time = time_bulks * n_clients // selection_size

    # warmup load
    warmup_dir = f"results/warmup_methods/iid_{iid}__{n_clients}_{selection_size}__slow_{slow_clients_relation}__fast_{fast_clients_relation}__mid_mean_{mid_clients_mean}"
    with open(os.path.join(warmup_dir, f"clients_details.pkl"), "rb") as f:
        clients_details = pickle.load(f)

    # Load and preprocess the dataset
    train_dataset, test_dataset = get_data(dataset_name)

    # Split the dataset into smaller datasets for each client
    client_datasets, qs = split_data(train_dataset, n_clients, iid=iid, dataset_name=dataset_name, qs=clients_details["qs"], data_sizes=clients_details["data_sizes"])


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
    all_clients_dists = clients_details["all_clients_dists"]
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
        if warmup_iters:
            if cs_method == BSFL:
                with open(os.path.join(warmup_dir, f"{cs_method.__name__}__alpha_{alpha}__beta_{beta}.pkl"), "rb") as f:
                    cs_method_states = pickle.load(f)
            else:
                with open(os.path.join(warmup_dir, f"{cs_method.__name__}.pkl"), "rb") as f:
                    cs_method_states = pickle.load(f)
            selection_method = cs_method_states[warmup_iters]
        else:
            selection_method = cs_method(all_clients, total_time, n_clients, selection_size, iid, **args)
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
        {'warmup_iters': 200, 'beta': 2, 'alpha': 0.1},   # alpha, beta in [(0.1, 0.5), (0.1, 1), (0.1, 2), (1, 0.5), (1, 2), (10, 2)]
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
    fast_clients_relation = 0.05    # [0.05, 0.02, 0.1],
    slow_clients_relation = 0.2    # [0.2, 0.1],
    mid_clients_mean = (0.15, 0.2)   # [(0.15, 0.2), (0.75, 0.8), (0.4, 0.45), (0.15, 0.6)]

    selection_methods_compare(css, css_args, time_bulks, n_clients, selection_size, dataset_name=dataset_name, iid=iid,
                              calc_regret=calc_regret, lr=lr, fast_clients_relation=fast_clients_relation, slow_clients_relation=slow_clients_relation, mid_clients_mean= mid_clients_mean)

