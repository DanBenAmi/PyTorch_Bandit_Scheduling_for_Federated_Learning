import numpy as np
import sys
import os
from typing import List
import math
from scipy.optimize import least_squares

from Client import *


class Client_Selection:
    def __init__(self, total_time, n_clients, selection_size):
        self.total_time = total_time
        self.time_left = total_time
        self.rounds = 0
        self.n_clients = n_clients
        self.selection_size = selection_size
        self.last_selection_indices = []
        self.n_observations = np.array([0]*n_clients)
        self.selection_communication_time = 0
        self.mu_rate = np.array([0] * n_clients, dtype=np.float32) # clients' averages iteration rate



    def select_clients(self):
        return None

    def post_iter_process(self, _):
        self.n_observations[self.last_selection_indices] += 1

    def softmax_with_temperature(self, x, temperature):
        x_scaled = x / temperature
        exp_x = np.exp(x_scaled)
        return exp_x / np.sum(exp_x)

    def update_n_obs_warmup(self, warmup_iters, slow_mid_fast_means, slow_mid_fast_relations, dists, T=0.1):   # BSFL:T=1
        # n_obs = np.concatenate((
        #     np.ones(int(self.n_clients*slow_mid_fast_relations[0]))* slow_mid_fast_means[0],
        #     np.ones(int(self.n_clients*slow_mid_fast_relations[1]))* slow_mid_fast_means[1],
        #     np.ones(int(self.n_clients*slow_mid_fast_relations[2]))* slow_mid_fast_means[2]
        # ))
        n_obs = dists[:,0]
        n_obs = self.softmax_with_temperature(n_obs, T)
        n_obs = n_obs * warmup_iters * self.selection_size
        self.n_observations = n_obs.astype(int)
        # self.n_observations = np.ones(self.n_clients)*int(self.selection_size * warmup_iters / self.n_clients)
        self.mu_rate = np.random.normal(loc=dists[:,0], scale=dists[:,1]/warmup_iters)


class BSFL(Client_Selection):
    def __init__(self, all_clients, total_time, n_clients, selection_size, iid, alpha=1, beta=5, tau_min=0.1):
        super().__init__(total_time, n_clients, selection_size)
        self.iid = iid
        self.mu_rate = np.array([0] * n_clients, dtype=np.float32) # clients' averages iteration rate
        self.ucb = np.array([100000] * n_clients, dtype=np.float32)
        self.g = np.array([100000] * n_clients, dtype=np.float32)
        self.n_observations = np.array([0] * n_clients) # clients' selected counter (C_k for k in K)
        self.alpha = alpha
        self.beta = beta
        self.tau_min = tau_min

        if not iid:
            self.data_quality = np.array([client.q for client in all_clients])
            self.data_size = np.array([client.data_size for client in all_clients])
            self.sum_importance = np.sum(self.data_size * self.data_quality)

    def __repr__(self):
        return "BSFL"

    def calc_indices_energy(self, indices, climb_iters=5):
        ucbs = self.ucb[indices]
        gs = self.g[indices]
        # energy = ucbs.min + (alpha / self.tau_min) * gs.mean()     #energy is function of min normalized time which is 0.1 TODO: remove taumin here!!
        energy = ucbs.min() + self.alpha * gs.mean()
        return energy

    def simulated_annealing(self, iters, climb_iters=50, ret_max=True):
        selection_indices = list(np.random.choice(self.n_clients, self.selection_size, replace=False))
        selection_energy = self.calc_indices_energy(selection_indices)  # initial random selection
        k = 1
        cntr1 = 0
        cntr2 = 0
        cntr3 = 0
        history = [0] * (iters + self.n_clients * climb_iters - 1)
        max_energy = -1 * float('inf')
        while k < iters + np.sqrt(self.n_clients) * climb_iters:
            history[k - 1] = selection_energy
            k += 1
            new_client_num = np.random.choice(list(set(range(self.n_clients)) - set(selection_indices)))  # new client num
            new_selection_indices = selection_indices.copy() + [new_client_num]
            if bool(random.getrandbits(1)):
                client_to_remove = selection_indices[self.ucb[selection_indices].argmin()]
            else:
                client_to_remove = selection_indices[self.g[selection_indices].argmin()]
            new_selection_indices.remove(client_to_remove)
            new_energy = self.calc_indices_energy(new_selection_indices)
            T = 1 - ((k + 1) / 5000) ** 0.02  # 2/np.log(k)    #1- ((k+1)/iters)**0.5     #1-(k+1)/iters
            if selection_energy < new_energy:
                cntr1 += 1
                selection_indices = new_selection_indices
                selection_energy = new_energy
            elif np.e ** ((new_energy - selection_energy) / T) > random.uniform(0, 1) and k < iters:
                cntr2 += 1
                selection_indices = new_selection_indices
                selection_energy = new_energy
            else:
                cntr3 += 1

            if ret_max and selection_energy >= max_energy:
                max_energy = selection_energy
                max_indices = selection_indices

        # print(f"Simulated Annealing: up steps:{cntr1}, down steps:{cntr2}, stay steps:{cntr3}")
        if ret_max:
            return max_indices
        return selection_indices

    def select_clients(self):

        # Initialize - first select each client once
        unselected_clients_indices = [id for id in range(self.n_clients) if self.n_observations[id]==0]
        if unselected_clients_indices != []:
            selected_indices = unselected_clients_indices[:self.selection_size]
            if len(selected_indices)<self.selection_size:
                possible_ind_to_add = list(set(range(self.selection_size)) - set(selected_indices)) # all indices that weren't selected
                selected_indices += possible_ind_to_add[:self.selection_size-len(selected_indices)]

        else:
            # main loop:
            selected_indices = self.simulated_annealing(iters=5000)     #TODO change iters to 1000

        self.last_selection_indices = selected_indices
        return selected_indices

    def post_iter_process(self, trained_dict):
        iter_times = trained_dict['iter_times']
        curr_rates = self.tau_min/np.array(iter_times)
        iter = trained_dict["iter"]

        # update selected clients mu_rate and n_observations
        self.mu_rate[self.last_selection_indices] = (self.mu_rate[self.last_selection_indices] * self.n_observations[self.last_selection_indices] + curr_rates) / (self.n_observations[self.last_selection_indices]+1)
        self.n_observations[self.last_selection_indices] += 1

        # update all clients g and ucb
        for client in range(self.n_clients):
            if self.n_observations[client] > 0 and iter > 0:
                self.ucb[client] = self.mu_rate[client] + np.sqrt((self.selection_size + 1) * np.log(iter) / self.n_observations[client])
                if self.iid:
                    self.g[client] = np.abs(self.selection_size / self.n_clients - self.n_observations[client] / iter) ** self.beta * np.sign(
                        self.selection_size / self.n_clients - self.n_observations[client] / iter)
                else:
                    tmp_res = self.selection_size * self.data_size[client] * self.data_quality[client] / self.sum_importance - self.n_observations[client] / iter
                    self.g[client] = np.abs(tmp_res) ** self.beta * np.sign(tmp_res)



class cs_ucb(Client_Selection):
    def __init__(self, all_clients, total_time, n_clients, selection_size, iid):
        super().__init__(total_time, n_clients, selection_size)
        self.n_observations = np.array([0] * n_clients) # clients' selected counter (C_k for k in K)
        self.sample_mean_reward = np.array([0] * n_clients, dtype=np.float32) # y(t)
        self.ucb = np.array([100000] * n_clients, dtype=np.float32)
        self.iid = iid
        if not iid:
            self.data_size = np.array([client.data_size for client in all_clients])
            self.phi = 0.5
            self.total_data_size = sum([client.data_size for client in all_clients])
            self.min_data_size = min([client.data_size for client in all_clients])
            self.cmin = self.phi * selection_size * self.min_data_size / self.total_data_size
            self.c = [client.data_size / self.min_data_size * self.cmin for client in
                      all_clients]  # fairness constrained for each client
            self.D = np.zeros(n_clients)
            self.y_hat = np.ones(n_clients)
            self.b = np.zeros(n_clients)
            self.beta = 0.5

    def __repr__(self):
        if self.iid:
            return "CS-UCB"
        else:
            return "CS-UCB-Q"

    def cs_ucb_selection(self):
        # Initialize - first select each client once
        unselected_clients_indices = [id for id in range(self.n_clients) if self.n_observations[id] == 0]
        if unselected_clients_indices != []:
            selected_indices = unselected_clients_indices[:self.selection_size]
            if len(selected_indices) < self.selection_size:
                possible_ind_to_add = list(
                    set(range(self.selection_size)) - set(selected_indices))  # all indices that weren't selected
                selected_indices += possible_ind_to_add[:self.selection_size - len(selected_indices)]

        else:
            # main loop:
            selected_indices = np.argsort(self.ucb)[-1 * self.selection_size:]

        self.last_selection_indices = selected_indices
        return selected_indices

    def cs_ucb_q_selection(self):
        for id in range(self.n_clients):
            if self.n_observations[id]>0:
                self.y_hat[id] = min(1, self.ucb[id])
            else:
                self.y_hat[id] = 1
            self.D[id] = max(self.D[id]+self.c[id]-self.b[id], 0)

        selected_indices = sorted(list(range(self.n_clients)), key=lambda id:(1-self.beta)*self.y_hat[id]+self.beta*self.D[id], reverse=True)[:self.selection_size]

        self.last_selection_indices = selected_indices
        return selected_indices

    def select_clients(self):
        if self.iid:
            return self.cs_ucb_selection()

        else:
            return self.cs_ucb_q_selection()


    def post_iter_process(self, trained_dict):
        iter_times = trained_dict['iter_times']
        rewards = 1 - np.array(iter_times)
        iter = trained_dict["iter"]

        # update selected clients mu_rate and n_observations
        self.sample_mean_reward[self.last_selection_indices] = (self.sample_mean_reward[self.last_selection_indices] * self.n_observations[
            self.last_selection_indices] + rewards) / (self.n_observations[self.last_selection_indices] + 1)
        self.n_observations[self.last_selection_indices] += 1

        # updata all client's ucb
        for client in range(self.n_clients):
            if self.n_observations[client] > 0:
                if self.iid:
                    self.ucb[client] = self.sample_mean_reward[client] + np.sqrt(
                        (self.selection_size + 1) * np.log(iter) / self.n_observations[client])
                else:
                    self.ucb[client] = self.sample_mean_reward[client] + np.sqrt(
                        2 * np.log(iter) / self.n_observations[client])

    # def update_n_obs_warmup(self, warmup_iters, slow_mid_fast_means, slow_mid_fast_relations, dists):
    #     def equations(vars):
    #         n_obs = vars
    #         common_expr = slow_mid_fast_means[0] + np.sqrt(((self.selection_size + 1) * np.log(warmup_iters)) / n_obs[0])
    #
    #         eqs = []
    #         # Common equality for all mu + sqrt(((m+1)*ln t)/x_i)
    #         for i in range(1, len(n_obs)):
    #             eqs.append(slow_mid_fast_means[i] + np.sqrt(((self.selection_size + 1) * np.log(warmup_iters)) / n_obs[i]) - common_expr)
    #
    #         # Linear sum constraint
    #         eq_sum = slow_mid_fast_relations @ n_obs - (self.selection_size * warmup_iters / self.n_clients)
    #         eqs.append(eq_sum)
    #
    #         return eqs
    #
    #     initial_guesses = [1e4, warmup_iters/1e4, warmup_iters]
    #     result = least_squares(equations, initial_guesses, bounds=(0, np.inf))
    #     n_obs_per_group = result.x
    #     self.n_observations = np.concatenate((
    #         np.ones(int(self.n_clients*slow_mid_fast_relations[0]))*int(n_obs_per_group[0]),
    #         np.ones(int(self.n_clients*slow_mid_fast_relations[1]))*int(n_obs_per_group[1]),
    #         np.ones(int(self.n_clients*slow_mid_fast_relations[2]))*int(n_obs_per_group[2])
    #     ))
    #     self.sample_mean_reward = 1 - 0.1/np.random.normal(loc=dists[:,0], scale=dists[:,1]/warmup_iters)

    def update_n_obs_warmup(self, warmup_iters, slow_mid_fast_means, slow_mid_fast_relations, dists, T):
        super().update_n_obs_warmup(warmup_iters, slow_mid_fast_means, slow_mid_fast_relations, dists, T)
        self.sample_mean_reward = 1 - 0.1/np.random.normal(loc=dists[:,0], scale=dists[:,1]/warmup_iters)



class RBCS_F(Client_Selection):
    def __init__(self, all_clients, total_time, n_clients, selection_size, iid):
        super().__init__(total_time, n_clients, selection_size)

        self.all_clients = all_clients
        self.m = selection_size
        self.beta = 0.5
        self.lamda = 1
        self.alpha = 0.1
        # initialize
        self.H_of_clients = [self.lamda * np.eye(3) for i in range(n_clients)]
        self.b = np.zeros((3, n_clients)).transpose()
        self.z = np.zeros(n_clients)
        self.theta_hat = np.zeros((n_clients, 3))
        self.tau_hat = np.zeros(n_clients)
        self.c = np.zeros((n_clients, 3))
        self.x = np.zeros(n_clients)
        self.V = 5
        self.tau_bar = np.zeros(n_clients)
        # A typical TCP handshake involves several steps: SYN, SYN-ACK, and ACK. Let's assume that each handshake takes
        # approximately 20 milliseconds, which is a reasonable estimate for a basic network round-trip time.
        # therefore in each round, for each client to send the context is 0.02, in selection_size orthogonal channels.
        self.selection_communication_time = 0.02 * n_clients / selection_size / 5 # we supppose the training time is between 1 sec to 10 sec and it is normalized..

    def __repr__(self):
        return "RBCS-F"

    def div_and_conc(self):
        indices_of_z_sorted = np.argsort(-self.z) #sort in descending order. i.e. z[indices_of_z_sorted[0]] is biggest
        F_max = np.ones(len(self.tau_bar))*np.inf
        S_max = [0]*len(self.tau_bar)
        for n_max in range(len(self.tau_bar)):
            S_n_max = []
            for n in indices_of_z_sorted:
                if self.tau_bar[n] <= self.tau_bar[n_max]:
                    S_n_max.append(n)
                if len(S_n_max) == self.m:
                    S_max[n_max] = S_n_max
                    F_max[n_max] = self.V*max([self.tau_bar[i] for i in S_n_max])-sum([self.z[i] for i in S_n_max])
                    break
        n_star = np.argmin(F_max)
        x = np.zeros(len(self.tau_bar))
        for i in S_max[n_star]:
            x[i] = 1
        return x, S_max[n_star]

    def select_clients(self):
        for id in range(self.n_clients):
            self.theta_hat[id] = np.linalg.inv(self.H_of_clients[id])@self.b[id]
            # c = [1⁄mu , s , M/B] where mu is available CPU ratio of the client, s is A binary indicator indicates if
            # client n has participated in training in the last round, M is the size of the model’s parameters
            # (measured by bit) and B indicates the allocated bandwidth.
            mu = np.clip(np.random.normal(self.all_clients[id].mean_rate, self.all_clients[id].std_rate*6)+np.random.randn()/5,0.1,1) # assume CPU is proportional to the actual time but with
            self.c[id] = np.array([1/mu, self.x[id], 1])
            self.tau_hat[id] = self.c[id].transpose()@self.theta_hat[id]
            self.tau_bar[id] = self.tau_hat[id] - \
                                  self.alpha*np.sqrt(self.c[id].transpose() @ self.H_of_clients[id] @ self.c[id])
        self.x, selected_indices = self.div_and_conc()
        self.last_selection_indices = selected_indices
        return selected_indices

    def post_iter_process(self, trained_dict):
        self.n_observations[self.last_selection_indices] += 1
        iter_times = trained_dict['iter_times']
        self.z = np.array([max([self.z[i] +self.beta - self.x[i], 0]) for i in range(self.n_clients)])
        for i, id in enumerate(self.last_selection_indices):
            self.H_of_clients[id] += self.x[id] * self.c[id] @ self.c[id].transpose()
            self.b[id] += self.x[id] * iter_times[i] * self.c[id]

    def update_n_obs_warmup(self, warmup_iters, slow_mid_fast_means, slow_mid_fast_relations, dists, T):
        super().update_n_obs_warmup(warmup_iters, slow_mid_fast_means, slow_mid_fast_relations, dists, T)
        self.tau_hat = 1 - 0.1/np.random.normal(loc=dists[:,0], scale=dists[:,1]/warmup_iters)



class Random_Selection(Client_Selection):
    def __init__(self, all_clients, total_time, n_clients, selection_size, iid):
        super().__init__(total_time, n_clients, selection_size)
        self.iid = iid
        if not iid:
            self.data_size = np.array([client.data_size for client in all_clients])
            self.probs = self.data_size / self.data_size.sum()
        else:
            self.probs = 1 / (np.ones(n_clients)*n_clients)

    def __repr__(self):
        return "Random Selection"

    def select_clients(self):
        selected_indices = np.random.choice(self.n_clients, self.selection_size, replace=False, p=self.probs)
        self.last_selection_indices = selected_indices
        return selected_indices


class PowerOfChoice(Client_Selection):
    def __init__(self, all_clients, total_time, n_clients, selection_size, iid):
        super().__init__(total_time, n_clients, selection_size)
        self.clients_loss = np.ones(n_clients) * 100000

    def __repr__(self):
        return "Power-of-Choice"

    def select_clients(self):
        selected_indices = np.argsort(self.clients_loss)[-1*self.selection_size:]
        self.last_selection_indices = selected_indices
        return selected_indices

    def post_iter_process(self, trained_dict):
        self.n_observations[self.last_selection_indices] += 1

        for loss, id in zip(trained_dict["loss"], self.last_selection_indices):
            self.clients_loss[id] = (self.clients_loss[id]*(self.n_observations[id]-1)+loss) / self.n_observations[id]



