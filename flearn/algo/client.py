
import numpy as np
from ..common import dataset
from torch.utils.data import dataloader
import torch


class Client:
    def __init__(self, name, group, train_data, test_data, model, opt, lossf, data_seed=0, lamD=0):
        self.name = name
        self.model = model
        self.opt = opt
        self.group = group
        self.lossf = lossf
        self.lamD = lamD

        data_x = torch.cat((train_data['x'], test_data['x']), 0)
        data_y = torch.cat((train_data['y'], test_data['y']), 0)

        if data_seed == 0:
            # Keep data ordering
            self.train_data = train_data
            self.val_data = test_data
            self.test_data = test_data
        else:
            np.random.seed(data_seed)
            indices = np.arange(len(data_x))
            np.random.shuffle(indices)
            data_x = data_x[indices]
            data_y = data_y[indices]

            self.train_data = {
                'x': data_x[:int(len(data_x) * 0.8)],
                'y': data_y[:int(len(data_x) * 0.8)]
            }

            self.val_data = {
                'x': data_x[int(len(data_x) * 0.8):int(len(data_x) * 0.9)],
                'y': data_y[int(len(data_x) * 0.8):int(len(data_x) * 0.9)]
            }

            self.test_data = {
                'x': data_x[int(len(data_x) * 0.9):],
                'y': data_y[int(len(data_x) * 0.9):]
            }
        self.train_dataset = dataset.FEDDataset(**self.train_data)
        self.valid_dataset = dataset.FEDDataset(**self.val_data)
        self.test_dataset = dataset.FEDDataset(**self.test_data)

    def __str__(self):
        return self.name

    def get_num_samples(self, name='train'):
        if name == 'train':
            return len(self.train_data['y'])
        elif name == 'valid':
            return len(self.train_data['y'])
        else:
            return len(self.test_data['y'])

    def set_weights(self, wdict):
        """Set model parameters"""
        for key in wdict:
            self.model.state_dict()[key] = wdict[key]

    def get_weights(self):
        """Set model parameters"""
        weight_dct = {}
        for name, param in self.model.named_parameters():
            weight_dct[name] = param
        return weight_dct

    def get_grads(self):
        """Set model parameters"""
        grad_dct = {}
        for name, param in self.model.named_parameters():
            grad_dct[name] = param.grad
        return grad_dct

    def get_lambda(self):
        return self.lamD

    def update_lambda(self,lamD):
        self.lamD=lamD

    def get_train_error(self):
        y_bar = self.model(self.train_data['x'])
        return self.lossf(y_bar, self.train_data['y']).item()

    def get_val_error(self):
        y_bar = self.model(self.val_data['x'])
        return self.lossf(y_bar, self.val_data['y']).item()

    def get_test_error(self):
        y_bar = self.model(self.test_data['x'])
        return self.lossf(y_bar, self.test_data['y']).item()

    def get_train_accuracy(self):
        y_bar = self.model(self.train_data['x']).max(dim=1)[1]
        return (y_bar == self.train_data['y']).int().sum().item() / len(self.train_data['y'])

    def get_val_accuracy(self):
        y_bar = self.model(self.val_data['x']).max(dim=1)[1]
        return (y_bar == self.val_data['y']).int().sum().item() / len(self.val_data['y'])

    def get_test_accuracy(self):
        y_bar = self.model(self.test_data['x']).max(dim=1)[1]
        return (y_bar == self.test_data['y']).int().sum().item() / len(self.test_data['y'])

    def solve_avg(self, num_epochs, batch_size):
        """Run stochastic gradient descent on local data and return weight to server"""
        loader = dataloader.DataLoader(dataset=self.train_dataset,
                                       batch_size=batch_size,
                                       shuffle=False)
        for e in range(num_epochs):
            for x, y in loader:
                self.opt.zero_grad()
                y_bar = self.model(x)
                loss = self.lossf(y_bar, y)
                loss.backward()
                self.opt.step()
        return self.get_weights(), self.get_train_error(), self.get_train_accuracy()

    def solve_sgd(self, num_epochs=1, batch_size=-1):
        """Run stochastic gradient descent on local data and return gradient to server"""
        if batch_size == -1:
            batch_size = len(self.train_dataset)
        loader = dataloader.DataLoader(dataset=self.train_dataset,
                                       batch_size=batch_size,
                                       shuffle=True)
        for e in range(num_epochs):
            for x, y in loader:
                self.opt.zero_grad()
                y_bar = self.model(x)
                loss = self.lossf(y_bar, y)
                loss.backward()
        return self.get_grads(), loss.item(), self.get_train_accuracy()

# class DL_Client(Client):
#     def __init__(self, name, group, train_data, test_data, model, opt, lossf, data_seed=0, lamD=0):
#         super().__init__( name, group, train_data, test_data, model, opt, lossf, data_seed)
#         self.lamD=lamD
#
#     def get_lambda(self):
#         return self.lamD
#
#     def update_lambda(self,lamD):
#         self.lamD=lamD

