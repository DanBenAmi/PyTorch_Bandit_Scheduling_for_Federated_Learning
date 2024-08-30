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
from typing import List
import yaml
from torch.utils.tensorboard import SummaryWriter

from Linear_regrression.LinearRegressionModel import LinearRegressionModel
from CNN.CNN_Model import CNNModel
from CNN.FlexibleCNN import FlexibleCNN
from FL import FederatedLearning, LRScheduler
from Client_Selection import *
from data_utils import *

Debug = False

def selection_methods_compare(cs_methods:List[Client_Selection], css_args, time_bulks, n_clients, selection_size, dataset_name='cifar10',
                              iid=True, calc_regret=False, lr_sched=0.001, fast_clients_relation=None,
                              slow_clients_relation=None, mid_clients_mean=None, warmup_temperature=None):
    total_time = time_bulks * n_clients // selection_size

    # Load and preprocess the dataset
    train_dataset, test_dataset = get_data(dataset_name)

    # Split the dataset into smaller datasets for each client
    client_datasets, qs = split_data(train_dataset, n_clients, iid=iid, dataset_name=dataset_name)

    if not warmup_temperature:
        warmup_temperature = [1]*n_clients

    # Define the global model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if Debug:
        device = "cpu"

    # res dir
    res_dir = os.path.join(f"results","methods_compare",f"{dataset_name}", f'{datetime.now().strftime("%Y-%m-%d_%H-%M")}'
        f'_{"" if iid else "non"}_iid__{dataset_name}__{n_clients}_{selection_size}__{time_bulks}t__'
        f'lr{int(-1 * np.log10(lr_sched.get_lr()))}')
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
    slow_mid_fast_relations = [slow_clients_relation, 1-slow_clients_relation-fast_clients_relation, fast_relation]
    slow_mid_fast_means = [0.11, np.average(list(mid_clients_mean)), 0.95]
    all_clients_dists = np.concatenate((
        np.random.uniform(low=[0.1, 0], high=[0.12, 0.03], size=(round(n_clients * slow_clients_relation), 2)),
        np.random.uniform(low=[mid_clients_mean[0], 0], high=[mid_clients_mean[1], 0.03],
                          size=(round(n_clients * (1 - slow_clients_relation - fast_clients_relation)), 2)),
        np.random.uniform(low=[0.93, 0], high=[0.97, 0.03], size=(round(n_clients * fast_clients_relation), 2))
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

    run_dict = {"lr_sched":lr_sched.scheduler_type, "calc_regret": calc_regret, "iid": iid, 'total_time': total_time, 'dataset':dataset_name, 'n_clientsn':n_clients, 'selection_size':selection_size, "fast_clients_relation": fast_clients_relation, "slow_clients_relation": slow_clients_relation, **css_args[0]}
    print(datetime.now().strftime("%Y-%m-%d_%H:%M"), "\n", {"lr": lr_sched.scheduler_type, "calc_regret": calc_regret, "iid": iid, 'total_time': total_time, 'dataset':dataset_name, 'n_clientsn':n_clients, 'selection_size':selection_size, "fast_clients_relation": fast_clients_relation, "slow_clients_relation": slow_clients_relation, "mid_clients_mean":mid_clients_mean}, "\n", css_args[0])
    # bsfl_loss, bsfl_acc = None, None
    alpha, beta = css_args[0]['alpha'], css_args[0]['beta']
    for cs_method, args, T in zip(cs_methods, css_args, warmup_temperature):
        warmup_iters = args.pop('warmup_iters')
        selection_method = cs_method(all_clients, total_time, n_clients, selection_size, iid, **args)
        tb_dir = os.path.join(res_dir, f"tb_{selection_method}")
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_hparams(run_dict, {})

        res = args.copy()
        res.update({'total_time': total_time, 'dataset':dataset_name, 'n_clientsn':n_clients, 'selection_size':selection_size, "fast_clients_relation": fast_clients_relation, "slow_clients_relation": slow_clients_relation})
        global_model.load_state_dict(global_weights)

        # Create Federated Learning simulation
        fl_simulation = FederatedLearning(global_model=global_model, all_clients=all_clients, test_data=test_dataset,
                                          device=device, track_observations=False, iid=iid, alpha=alpha, beta=beta, tb_writer=writer)
        res.update({"data_sizes": fl_simulation.data_size, "data_quality": fl_simulation.data_quality, "rate_dists": all_clients_dists, "lr": lr_sched})

        # client selection method
        if warmup_iters:
            selection_method.update_n_obs_warmup(warmup_iters, slow_mid_fast_means, slow_mid_fast_relations,
                                                 all_clients_dists, T=T)
            fl_simulation.selection_warmup_til_all_selected(selection_method, curr_iter=warmup_iters)
        warmup_n_observations = selection_method.n_observations.copy()

        # Train the global model using Federated Learning
        res.update(fl_simulation.train(selection_size, selection_method, total_time, calc_regret=calc_regret, warmup_selection_alg=warmup_iters, lr_sched=lr_sched))
        res.update({'n_observations_no_warmup': res['n_observations'] - warmup_n_observations, "warmup_iters": warmup_iters})

        with open(os.path.join(res_dir, f"{selection_method}.pkl"), 'wb') as f:
            pickle.dump(res, f)

        # Early stop
        # if isinstance(selection_method, BSFL):
        #     bsfl_loss, bsfl_acc = res["loss"][-1], res["accuracy"][-1]
        # elif res["loss"][-1] < bsfl_loss or res["accuracy"][-1] > bsfl_acc:
        #     print(f"early stopped because {selection_method} is better in this inputs")


# Function to load the YAML configuration
def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    # Load the config
    config_path = r'configs/non_iid_fashion.yml'
    config = load_config(config_path)

    css = [BSFL, cs_ucb, RBCS_F, PowerOfChoice, Random_Selection]

    # Extract variables from config
    css_args = []
    for arg in config['css_args']:
        # Convert values in each dict to floats where appropriate
        converted_arg = {key: float(value) if isinstance(value, (int, float, str)) and key == "warmup_iters" else value
                         for key, value in arg.items()}
        css_args.append(converted_arg)

    iid = config['iid']
    dataset_name = config['dataset_name']
    time_bulks = config['time_bulks']
    n_clients = config['n_clients']
    selection_size = config['selection_size']
    calc_regret = config['calc_regret']
    fast_relation = config['fast_relation']
    slow_relation = config['slow_relation']
    mid_clients_mean = tuple(config['mid_clients_mean'])
    warmup_temperature = config['warmup_temperature']

    # Extract LR scheduler parameters
    lr_sched_type = config['lr_scheduler']['type']
    lr_sched_params = {
        "first_lr": float(config['lr_scheduler']['first_lr']),
        "last_lr": float(config['lr_scheduler']['last_lr']),
        "num_iters": config['lr_scheduler']['num_iters']
    }

    # Assuming LRScheduler is a class that takes type and params
    lr_sched = LRScheduler(lr_sched_type, **lr_sched_params)

    # Your function call
    selection_methods_compare(css, css_args, time_bulks, n_clients, selection_size, dataset_name=dataset_name, iid=iid,
                              calc_regret=calc_regret, lr_sched=lr_sched, fast_clients_relation=fast_relation,
                              slow_clients_relation=slow_relation, mid_clients_mean=mid_clients_mean,
                              warmup_temperature=warmup_temperature)

