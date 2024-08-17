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
import itertools

from Linear_regrression.LinearRegressionModel import LinearRegressionModel
from CNN.CNN_Model import CNNModel
from CNN.FlexibleCNN import FlexibleCNN
from FL import FederatedLearning
from Client_Selection import *
from data_utils import *


def warmup_exp(irers, iters_bt_save, iid, n_clients__sel_size, fast_clients_relation, slow_clients_relation, mid_clients_mean):
    n_clients, selection_size = n_clients__sel_size

    if iid:
        qs = [None]*n_clients
        data_sizes = [60000/n_clients]*n_clients
    else:
        data_sizes = np.array(distribute_datapoints(len(dataset), n_clients, 2 / n_clients))
        np.random.shuffle(data_sizes)
        qs = [random.uniform(0.5, 1) for _ in range(n_clients)]  # Random noise levels between 0 and 1 for each client

    all_clients_dists = np.concatenate((
        np.random.uniform(low=[0.93, 0], high=[0.97, 0.03], size=(round(n_clients * fast_clients_relation), 2)),
        np.random.uniform(low=[0.1, 0], high=[0.12, 0.03], size=(round(n_clients * slow_clients_relation), 2)),
        np.random.uniform(low=[mid_clients_mean[0], 0], high=[mid_clients_mean[1], 0.03],
                          size=(round(n_clients * (1 - slow_clients_relation - fast_clients_relation)), 2))
    ))

    all_clients = [Client(id=i, local_model=None, data=[], mean_std_rate=all_clients_dists[i],
                          device="cpu", q=qs[i], data_size = data_sizes[i]) for i in range(n_clients)]


    for cs_methos in [BSFL, cs_ucb, RBCS_F]:












css = [BSFL, cs_ucb, RBCS_F]

exp_parameters = {
"iid": [True, False],
"n_clients__sel_size": [(1000, 10),(500,25)],
"fast_clients_relation":[0.05, 0.02, 0.1],
"slow_clients_relation": [0.2, 0.1],
"mid_clients_mean": [(0.15, 0.2), (0.75, 0.8), (0.4, 0.45), (0.15, 0.6)],
}

# Generate all combinations of hyperparameter values
keys, values = zip(*exp_parameters.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Number of combinations
print(f"Total combinations: {len(combinations)}")

# Example loop to evaluate each combination
for idx, combo in enumerate(combinations):












