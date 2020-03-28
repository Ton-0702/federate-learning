
import numpy as np
import torch
import random

from .models import MLP

# Reference paper: "Federated Learning of Deep Networks using Model Averaging",
# https://uk.arxiv.org/pdf/1602.05629v1.pdf


def get_grad(model):
    grad_dict = {}
    for name, param in model.named_parameters():
        grad_dict[name] = param.grad
    return grad_dict


class Client:
    def __init__(self, data, configs={}):
        self.configs = configs
        self.n_epochs = configs.get('n_epochs', 5)
        self.batch_size = configs.get('batch_size', 8)
        self.x = data['x']
        self.y = data['y']
        self.n = len(self.x)

    def get_num_samples(self):
        return self.n

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
                loss = loss_func(ys, ys_bar)
                loss.backward()
                opt.step()
        # Return client number of samples and gradient
        return self.n, get_grad(model)


def client(data,
           model,
           opt,
           loss_func,
           configs={}
           ):
    """
    # data: dictionary containing 'x' matrix and 'y' vector
    # model: the model
    # opt: optimizer
    # loss_func: loss functions
    # configs: dictionary containing training configs
    """
    x = data['x']
    y = data['y']
    n = len(x)
    n_epochs = configs.get('n_epochs', 5)
    batch_size = configs.get('batch_size', 8)

    for e in range(n_epochs):
        # Randomly order training data
        indices = np.arange(n)
        random.shuffle(indices)
        x = x[indices]
        y = y[indices]

        for i in range(n // batch_size):
            # Slice the data
            xs = x[i: i + batch_size]
            ys = y[i: i + batch_size]
            # Start training on current batch
            model.train()
            opt.zero_grad()
            ys_bar = model(xs)
            loss = loss_func(ys, ys_bar)
            loss.backward()
            opt.step()
    # Return client number of samples and gradient
    return n, get_grad(model)


def server(base_model,
           base_opt,
           loss_func,
           clients,
           num_rounds,
           model_configs,
           opt_configs):
    model = base_model(**model_configs)
    opt = base_opt(**opt_configs)
    n = 0
    for clt in clients:
        n += clt.get_n_samples()
    for r in range(num_rounds):
        # First iter, get number of samples from all clients
        for clt in clients:
            nk, grad = clt.local_update(model, opt, loss_func)
            for key in grad.keys():
                model.state_dict()[key] += nk/n * grad[key]


if __name__ == '__main__':
    pass

