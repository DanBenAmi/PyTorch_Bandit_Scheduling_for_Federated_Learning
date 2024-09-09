import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np


class Client:
    ''' Client object represent a client in the FL training process, each client holds a data and labels, the server
     holds the g function value, the ucb, the counter of the participation and the id of every client'''
    def __init__(self, id, data, local_model, mean_std_rate, device="cpu", q=1, tau_min=0.1, data_size=False):
        # -------------------------- inside client -----------------------------------------
        self.data = data
        self.local_model = local_model
        self.q = q  # Quality of the data
        self.mean_rate = mean_std_rate[0]
        self.std_rate = mean_std_rate[1]
        self.device = device
        self.batch_size = 32
        # -------------------------- in the server -----------------------------------------
        self.id = id  # ID i.e. idx in all_clients list
        if data_size:
            self.data_size = data_size
        else:
            self.data_size = len(data)
        self.tau_min = tau_min

    def train(self, optimizer, criterion, epochs=1):
        self.local_model.train()
        batch_losses = []           # List to store all batch losses
        train_loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        for epoch in range(epochs):
            for datapoint, labels in train_loader:
                datapoint, labels = datapoint.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.local_model(datapoint)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                # Debug: Print loss values
                # print(f"Client {self.client_id}, Epoch {epoch}, Loss: {loss.item()}")
        mean_loss = np.mean(batch_losses)    # Calculate mean loss
        iter_time = np.clip(self.tau_min/np.random.normal(self.mean_rate, self.std_rate), self.tau_min, 1)  # Simulate iteration time
        return iter_time, {"loss":mean_loss}



def shuffle_and_divide_dataset(images, labels, num_of_clients):
    ''' gets arrays of images and labels, shuffle them and reshape them to
     number of clients X number of images in each client X image shape'''
    perm = np.random.permutation(len(images))
    images = images[perm]
    labels = labels[perm]
    client_data_len = len(images)//num_of_clients
    clients_images = images.reshape((num_of_clients, client_data_len, *images.shape[1:]))
    clients_labels = labels.reshape((num_of_clients, client_data_len, *labels.shape[1:]))
    return clients_images, clients_labels


def show_img(img):
    ''' plot the image '''
    plt.figure()
    plt.imshow(img)
    plt.show()


def make_clients(clients_images, clients_labels, num_of_clients):
    ''' gets arrays that each tensor in the first dimension represent the imges/clients of each client, and returns
     a list of Client objects with the images and labels inside each client'''
    all_clients = [0]*num_of_clients
    for i in range(num_of_clients):
        all_clients[i] = Client(i, 10000, 10000, clients_images[i], clients_labels[i], 0)
    return all_clients



def init_clients(all_clients, all_clients_dists, selection_size, num_of_clients, t=0, cs_ucb=False):
    ''' initialize the clients attributes in all_clients list according to the selection rule'''
    if t==0:
        for client in all_clients:
            client.ucb = 100000
            client.g = 0
            client.num_of_observations = 0
    else:
        for client in all_clients:
            if cs_ucb:
                client.u_time = 1 - all_clients_dists[client.id][0]
            else:
                client.u_time = 1 / all_clients_dists[client.id][0]

            client.num_of_observations = t*selection_size//num_of_clients
            client.update_ucb_g(t, selection_size, num_of_clients)



def make_clients_non_iid(train_images_sorted, train_labels_sorted, num_of_clients, data_size_gama_teta=1000, class_gama_teta=5):
    ''' receive arrays of images and labels sorted by classes (all the class images together at the first of the array
     and so on..) and returns list of clients with the data divided in a non iid and imbalanced way, i.e. each client
      has different amount of images and classes of the images inside each client distributed not uniformly. '''
    classes_idxes = np.zeros(10, dtype=int)
    all_clients = []
    class_size = len(train_labels_sorted)//10
    for j in range(num_of_clients):

        class_data_size = [max(min(class_size-classes_idxes[k], int(np.random.gamma(1, data_size_gama_teta) / data_size_gama_teta
                * (np.shape(train_labels_sorted)[0]) / num_of_clients//10+1)), 1) for k in range(10)]
        client_data = np.empty([0, *np.shape(train_images_sorted)[1:]], dtype=float)
        client_labels = np.empty([0, *np.shape(train_labels_sorted)[1:]], dtype=float)
        for i in range(10):
            if class_data_size[i] == 1:
                classes_idxes[i] -= 1
            client_data = np.concatenate((client_data, train_images_sorted[(i * class_size + classes_idxes[i]):(
                        i * class_size + classes_idxes[i] + class_data_size[i]), :, :, :]))
            client_labels = np.concatenate((client_labels, train_labels_sorted[
                                                           i * class_size + classes_idxes[i]:i * class_size + classes_idxes[i] +
                                                                                       class_data_size[i]]))

        classes_idxes += class_data_size
        # change images quality
        q = np.random.uniform(0,1)
        gauss_noise = np.random.normal(0, 0.05*(1-q), client_data.shape)
        client_data += gauss_noise
        all_clients.append(Client(j, 10000, 10000, client_data, client_labels, 0, q=q))

    np.random.shuffle(all_clients)
    for i in range(num_of_clients):
        all_clients[i].id = i
    return all_clients

