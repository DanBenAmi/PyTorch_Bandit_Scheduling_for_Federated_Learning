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


def warmup_exp(warmup_iters, iters_bt_save, iid, n_clients__sel_size, fast_clients_relation, slow_clients_relation, mid_clients_mean):
    n_clients, selection_size = n_clients__sel_size
    len_data = 60000

    if iid:
        qs = [None]*n_clients
        data_sizes = [len_data/n_clients]*n_clients
    else:
        data_sizes = np.array(distribute_datapoints(len_data, n_clients, 2 / n_clients))
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

    # save dir
    res_dir = f"../results/warmup_methods/iid_{iid}__{n_clients}_{selection_size}__slow_{slow_clients_relation}__fast_{fast_clients_relation}__mid_mean_{mid_clients_mean}"
    os.makedirs(res_dir, exist_ok=True)
    clients_details = {
        "iid": iid,
        "n_clients__sel_size": n_clients__sel_size,
        "fast_clients_relation":fast_clients_relation,
        "slow_clients_relation": slow_clients_relation,
        "mid_clients_mean": mid_clients_mean,
        "data_sizes": data_sizes,
        "qs": qs,
        "all_clients_dists": all_clients_dists,
        "all_clients": all_clients,
        "warmup_iters": warmup_iters,
        "iters_bt_save": iters_bt_save
        }
    with open(os.path.join(res_dir, f"clients_details.pkl"), "wb") as f:
        pickle.dump(clients_details, f)

    tmp_model = FlexibleCNN((1,28,28))
    cs_method_states = {}
    for cs_method in [BSFL, cs_ucb, RBCS_F]:
        # Create Federated Learning simulation
        fl_simulation = FederatedLearning(global_model=tmp_model, all_clients=all_clients, device="cpu", track_observations=False, iid=iid, test_data=None)
        if cs_method == BSFL:
            for alpha, beta in [(0.1, 0.5), (0.1, 1), (0.1, 2), (1, 0.5), (1, 2), (10, 2)]:
                selection_method = cs_method(all_clients=all_clients, total_time=None, n_clients=n_clients,
                                             selection_size=selection_size, iid=iid, alpha=alpha, beta=beta)
                for j in range(int(warmup_iters // iters_bt_save)):
                    fl_simulation.selection_alg_warmup(selection_method, iters=iters_bt_save, curr_iter=j*iters_bt_save)
                    cs_method_states[(j + 1) * iters_bt_save] = copy.deepcopy(selection_method)

                with open(os.path.join(res_dir, f"{selection_method}__alpha_{alpha}__beta_{beta}.pkl"), "wb") as f:
                    pickle.dump(cs_method_states, f)
        else:
            selection_method = cs_method(all_clients=all_clients, total_time=None, n_clients=n_clients, selection_size=selection_size, iid=iid)
            for j in range(int(warmup_iters//iters_bt_save)):
                fl_simulation.selection_alg_warmup(selection_method, iters=iters_bt_save)
                cs_method_states[(j+1)*iters_bt_save] = copy.deepcopy(selection_method)

            with open(os.path.join(res_dir, f"{selection_method}.pkl"), "wb") as f:
                pickle.dump(cs_method_states, f)


css = [BSFL, cs_ucb, RBCS_F]

exp_parameters = {
"iid": [True],  # True/ False
"n_clients__sel_size": [(500, 25)],    # (1000, 10),(500,25)
"fast_clients_relation":[0.05, 0.1],    # [0.02, 0.05, 0.1]
"slow_clients_relation": [0.2, 0.1],
"mid_clients_mean": [(0.15, 0.2), (0.75, 0.8), (0.15, 0.6)],
}

print( exp_parameters)

# Generate all combinations of hyperparameter values
keys, values = zip(*exp_parameters.items())
combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Number of combinations
print(f"Total combinations: {len(combinations)}")

warmup_iters, iters_bt_save = 60000, 5000

# Example loop to evaluate each combination
for idx, combo in enumerate(combinations):
    print(f"combo number {idx} out of {len(combinations)}")
    warmup_exp(warmup_iters, iters_bt_save, **combo)












