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
from tqdm import tqdm
from typing import List
from itertools import combinations
from scipy.special import comb
from torch.optim.lr_scheduler import CosineAnnealingLR  # Add this import


from Client import *
from Client_Selection import *


# Define the Federated Learning Simulation class
class FederatedLearning:
    def __init__(self, global_model, all_clients:List[Client], test_data, device="cpu", iid=True, track_observations=False, beta=1, alpha=1, tau_min=0.1):
        self.device = device
        self.global_model = global_model
        self.global_model.to(self.device)
        self.all_clients = all_clients
        self.n_clients = len(all_clients)
        self.data_size = np.array([client.data_size for client in all_clients])
        self.last_selection_indices = []
        self.track_observations = [] if track_observations else False
        self.iid = iid
        self.tau_min = tau_min

        self.test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
        self.results = {'accuracy':[], 'loss':[], 'time':[], 'iters':[], "track observations": self.track_observations}

        # set criterion
        if len(self.test_loader.dataset[0][0].size()) == 1:  # data is 1d vector (linear regression task)
            self.criterion = nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # for regret analysis:
        self.alpha = alpha
        self.beta = beta
        if not iid:
            self.data_quality = np.array([client.q for client in all_clients])
            self.data_size = np.array([client.data_size for client in all_clients])
            self.sum_importance = np.sum(self.data_size * self.data_quality)
        else:
            self.data_size = None
            self.data_quality = None


    def distribute_model(self):
        global_weights = copy.deepcopy(self.global_model.state_dict())
        return global_weights

    def aggregate_models(self, client_updates):
        global_dict = self.global_model.state_dict()
        for k in global_dict.keys():
            global_dict[k] = client_updates[k]
        self.global_model.load_state_dict(global_dict)

    def calc_curr_regret(self, selection_method:Client_Selection, iter, tau_min=0.1):
        ''' receives the list of all clients, their distributions, the selection size, the alpha (for the reward
         calculation), and the indexes of a specific selection, finding the maximum energh selection and returning the
          difference between the maxumum energy and the received selection energy- i.e. the immediate regret. '''
        if comb(len(self.all_clients), selection_method.selection_size) > 300000:
            print("too complex for calculate the regret")
            exit(1)

        selection_indices = selection_method.last_selection_indices
        g = np.zeros(self.n_clients)
        for id in range(self.n_clients):
            if selection_method.n_observations[id] > 0:
                if self.iid:
                    tmp_res = selection_method.selection_size / selection_method.n_clients - selection_method.n_observations[id] / iter
                else:
                    tmp_res = selection_method.selection_size * self.data_size[id] * self.data_quality[id] / self.sum_importance - selection_method.n_observations[id] / iter
                g[id] = np.abs(tmp_res) ** self.beta * np.sign(tmp_res)

        selection_reward = min([self.all_clients[j].mean_rate for j in selection_indices]) + self.alpha * g[selection_indices].mean()

        max_reward = -np.inf
        for clients in combinations(self.all_clients, selection_method.selection_size):
            clients = list(clients)
            curr_selection_ids = [client.id for client in clients]
            curr_selection_reward = min([self.all_clients[j].mean_rate for j in curr_selection_ids]) + self.alpha * g[curr_selection_ids].mean()
            if curr_selection_reward > max_reward:
                max_reward = curr_selection_reward
                best_indices = curr_selection_ids
        # print('sel expeced energy: ', sel_energy, "sel idxes =", sel_idxes)
        # print('best expeced energy: ', max_reward, "best idxes =", best_idxes)
        return max_reward - selection_reward

    def evaluate_global_model(self, iters, time):
        # Test the global model
        self.global_model.eval()
        correct = 0
        total = 0
        acc = False
        with torch.no_grad():
            for datapoints, labels in self.test_loader:
                datapoints, labels = datapoints.to(self.device), labels.to(self.device)
                outputs = self.global_model(datapoints)
                if outputs.size()[-1]==1:
                    outputs = outputs.squeeze()
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    acc = correct / total
                loss = self.criterion(outputs, labels)



        if isinstance(acc, float):
            self.results['accuracy'].append(acc)
        self.results['loss'].append(loss.item())
        self.results['time'].append(time)
        self.results['iters'].append(iters)

        print("____________________________________________")
        for key, val in self.results.items():
            if isinstance(val, list) and val!=[]:
                print(f"{key}: {val[-1]}")
            else:
                print(f"{key}: {val}")


    def selection_alg_warmup(self, client_selection_method, iters):
        for iter in tqdm(range(1, iters+1), desc="selection algorithm warmup iterations"):
            trained_dict = {'iter_times':[None]*client_selection_method.selection_size}
            selected_clients_indices = client_selection_method.select_clients()
            selected_clients = [self.all_clients[i] for i in selected_clients_indices]
            for i, client in enumerate(selected_clients):
                iter_time = np.clip(self.tau_min / np.random.normal(client.mean_rate, client.std_rate), self.tau_min, 1)
                trained_dict["iter_times"][i] = iter_time
            trained_dict["iter"] = iter
            client_selection_method.post_iter_process(trained_dict)
            if isinstance(self.track_observations, list):
                if iter % 100 == 0:
                    self.track_observations.append(client_selection_method.n_observations.copy())


    def train(self, selection_size, client_selection_method:Client_Selection, total_time, time_bt_eval=3, warmup_selection_alg=0, calc_regret=False, lr=0.001):
        self.results['total_time'] = total_time
        time_left = total_time
        iter = warmup_selection_alg
        last_lr = lr/50
        lr_decay = (last_lr/lr)**(1/(total_time/time_bt_eval))

        # regret analysis
        if calc_regret:
            self.results["regret"] = []

        self.evaluate_global_model(iter, total_time - time_left)
        last_time_eval = time_left

        while time_left > 0:
            selected_clients_indices = client_selection_method.select_clients()
            time_left -= client_selection_method.selection_communication_time
            selected_clients = [self.all_clients[i] for i in selected_clients_indices]
            initial_weights = self.distribute_model()

            # Initialize client_updates with zeros
            client_updates = {k: torch.zeros_like(v).to(self.device) for k, v in initial_weights.items()}
            # coef = self.data_size[selected_clients_indices] / self.data_size[selected_clients_indices].sum()

            # train selection
            trained_dict = {'iter_times':[None]*selection_size, "loss":[None]*selection_size}
            for i, client in tqdm(enumerate(selected_clients)):
                client.local_model.load_state_dict(initial_weights)
                local_optimizer = optim.Adam(client.local_model.parameters(), lr=lr)
                iter_time, client_trained_dict = client.train(local_optimizer, self.criterion)

                trained_dict["iter_times"][i] = iter_time
                trained_dict["loss"][i] = client_trained_dict["loss"]

                # Accumulate the updates
                with torch.no_grad():
                    for k, v in client.local_model.state_dict().items():
                        client_updates[k] += v / selection_size      #* coef[i]

            trained_dict["iter"] = iter
            client_selection_method.post_iter_process(trained_dict)

            # Aggregate updates
            self.aggregate_models(client_updates)

            # update time and iters and eval
            time_left -= max(trained_dict["iter_times"])
            if time_left < last_time_eval-time_bt_eval or time_left<0:
                self.evaluate_global_model(iter, total_time-time_left)
                last_time_eval = time_left
                lr = lr * lr_decay

            # regret analysis
            if calc_regret:
                self.results["regret"].append(self.calc_curr_regret(client_selection_method, iter))

            iter += 1

        self.results["n_observations"] = client_selection_method.n_observations
        return self.results







