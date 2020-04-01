
import numpy as np
import torch
import random

from .models import MLP

# Reference paper: "Federated Learning of Deep Networks using Model Averaging",
# https://uk.arxiv.org/pdf/1602.05629v1.pdf


def get_weight(model):
    weight_dct = {}
    for name, param in model.named_parameters():
        weight_dct[name] = param
    return weight_dct


class Client:
    def __init__(self, name, data, configs={}):
        self.configs = configs
        self.n_epochs = configs.get('n_epochs', 5)
        self.batch_size = configs.get('batch_size', 8)
        self.x = data['x']
        self.y = data['y']
        self.n = len(self.x)
        self.name = name

    def get_num_samples(self):
        return self.n

    def get_name(self):
        return self.name

    def local_update(self, model, opt, loss_func):
        for _ in range(self.n_epochs):
            # Randomly order training data
            indices = np.arange(self.n)
            random.shuffle(indices)
            x = self.x[indices]
            y = self.y[indices]

            for i in range(self.n // self.batch_size):
                # Slice the data
                xs = x[i: i + self.batch_size]
                ys = y[i: i + self.batch_size]
                # Start training on current batch
                model.train()
                opt.zero_grad()
                ys_bar = model(xs)
                loss = loss_func(ys_bar, ys)
                loss.backward()
                opt.step()
        return self.n, get_weight(model)


def server(base_model,
           base_opt,
           loss_func,
           clients,
           num_rounds):
    model = base_model
    opt = base_opt
    n = 0
    for clt in clients:
        n += clt.get_num_samples()
    for r in range(num_rounds):
        for clt in clients:
            nk, ws = clt.local_update(model, opt, loss_func)
            for key in ws.keys():
                model.state_dict()[key] += nk/n * ws[key]
    return model

