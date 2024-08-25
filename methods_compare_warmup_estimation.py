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
    res_dir = os.path.join(f"results/methods_compare/{dataset_name}", f'{datetime.now().strftime("%Y-%m-%d_%H:%M")}'
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

    run_dict = {"le_sched":lr_sched.scheduler_type, "calc_regret": calc_regret, "iid": iid, 'total_time': total_time, 'dataset':dataset_name, 'n_clientsn':n_clients, 'selection_size':selection_size, "fast_clients_relation": fast_clients_relation, "slow_clients_relation": slow_clients_relation, **css_args[0]}
    print(datetime.now().strftime("%Y-%m-%d_%H:%M"), "\n", {"lr": lr_sched, "calc_regret": calc_regret, "iid": iid, 'total_time': total_time, 'dataset':dataset_name, 'n_clientsn':n_clients, 'selection_size':selection_size, "fast_clients_relation": fast_clients_relation, "slow_clients_relation": slow_clients_relation}, "\n", css_args[0])
    # bsfl_loss, bsfl_acc = None, None
    alpha, beta = css_args[0]['alpha'], css_args[0]['beta']
    for cs_method, args, T in zip(cs_methods, css_args, warmup_temperature):
        tb_dir = os.path.join(res_dir, f"tb_{cs_method}")
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
        warmup_iters = args.pop('warmup_iters')
        selection_method = cs_method(all_clients, total_time, n_clients, selection_size, iid, **args)
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


if __name__ == "__main__":
    css = [BSFL, cs_ucb, RBCS_F, PowerOfChoice, Random_Selection]
    css_args = [
        {'warmup_iters': 1e6, 'beta': 1, 'alpha': 1}, # 1e7=>T=5
        {'warmup_iters': 1e6},
        {'warmup_iters': 1e6},
        {'warmup_iters': 0},
        {'warmup_iters': 0}
    ]
    # warmup_temp = [5,5,5,5,5]
    iid = True
    dataset_name = 'fashion_mnist'  # 'cifar10' 'lin_reg' 'fashion_mnist'
    time_bulks = 8     # iid: 30 non: 20
    n_clients = 500
    selection_size = 25
    calc_regret = False
    lr = 5e-5      # iid: 5e-5 non: 2e-6
    fast_relation = 0.03    # [0.05, 0.02, 0.1]
    slow_relation = 0.1     # [0.2, 0.1]
    mid_clients_mean = (0.15, 0.2)   # [(0.15, 0.2), (0.75, 0.8), (0.4, 0.45), (0.15, 0.6)]
    warmup_temperature = [1, 0.1, 5, 0, 0]  # for 1e6 warmup iters [1, 0.1, 5, 0, 0]

    # regular_decay: first_lr, last_lr, num_iters / exponential_decay: base_lr, gamma /
    # step_decay: base_lr, step_size, gamma / cosine_annealing: first_lr, T_max, eta_min /
    # cyclic_lr: base_lr, max_lr, step_size_up, step_size_down (optional)
    lr_sched = LRScheduler("regular_decay", **{"first_lr": lr, "last_lr": lr/10, "num_iters":time_bulks*7})

    selection_methods_compare(css, css_args, time_bulks, n_clients, selection_size, dataset_name=dataset_name, iid=iid,
                              calc_regret=calc_regret, lr_sched=lr_sched, fast_clients_relation=fast_relation,
                              slow_clients_relation=slow_relation, mid_clients_mean= mid_clients_mean, warmup_temperature=warmup_temperature)
