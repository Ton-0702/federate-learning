import random
import torch
import numpy as np
import functools
import operator
from ..common.metrics import Metrics


def norm_grad(x):
    return torch.sqrt(torch.norm(x))


def norm_grad_dict(grads):
    grad_norms = {}
    for key in grads.keys():
        grad_norms[key] = norm_grad(grads[key])
    return grad_norms


class BaseServer:
    def __init__(self, model, opt, lossf, clients, train_data, test_data,
                 dataset_name, method_name, configs=None, metric_dir='experiments'):

        if configs is None:
            configs = {
                'num_rounds': 10,
                'pct_client_per_round': 0.3,
                'num_epochs': 3,
                'batch_size': 8,
                'lr': 0.1,
                'q': 1
            }
        for key, val in configs.items():
            setattr(self, key, val)

        self.model = model
        self.opt = opt
        self.lossf = lossf
        self.clients = clients
        self.train_data = train_data
        self.test_data = test_data
        self.dataset_name = dataset_name
        self.method_name = method_name
        self.metrics_dir = metric_dir

        self.metrics = Metrics([c.name for c in self.clients], configs,
                               self.dataset_name, self.method_name, self.metrics_dir)

    def save_model(self):
        pass

    def evaluate(self):
        for clt in self.clients:
            clt.set_weights(self.model.state_dict())
            train_acc = clt.get_train_accuracy()
            test_acc = clt.get_test_accuracy()
            # print('Name: ', clt.name)
            # print('Train accuracy: ', train_acc)
            # print('Test accuracy: ', test_acc)
            self.metrics.update(-1, clt.name, train_acc, test_acc, None)
    def my_mod_evaluate(self,r):
        for clt in self.clients:
            clt.set_weights(self.model.state_dict())
            train_acc = clt.get_train_accuracy()
            test_acc = clt.get_test_accuracy()
            # print('Name: ', clt.name)
            # print('Train accuracy: ', train_acc)
            # print('Test accuracy: ', test_acc)
            self.metrics.update(-r, clt.name, train_acc, test_acc, None)

    def train(self):
        pass

    def report(self):
        self.metrics.write()

    def get_nks(self):
        return [c.get_num_samples() for c in self.clients]

    def sample_clients(self):
        nks = self.get_nks()
        return np.random.choice(
            self.clients,
            int(len(self.clients)*self.pct_client_per_round),
            p=[e/sum(nks) for e in nks],
            replace=False
        )


class FedAvgServer(BaseServer):
    def __init__(self, model, opt, lossf, clients, train_data, test_data,
                 dataset_name, method_name, configs=None):
        super().__init__(model, opt, lossf, clients, train_data, test_data,
                         dataset_name, method_name, configs)

    def train(self):
        for r in range(self.num_rounds):
            n = 0
            sub_clients = self.sample_clients()
            for clt in sub_clients:
                n += clt.get_num_samples()
            for clt in sub_clients:
                clt.set_weights(self.model.state_dict())
                ws, error, acc = clt.solve_avg(self.num_epochs, self.batch_size)
                self.metrics.update(r, clt.name, error, acc, None)
                for key in ws.keys():
                    self.model.state_dict()[key] += clt.get_num_samples() / n * ws[key]


class FedSgdServer(BaseServer):
    def __init__(self, model, opt, lossf, clients, train_data, test_data,
                 dataset_name, method_name, configs=None):
        super().__init__(model, opt, lossf, clients, train_data, test_data,
                         dataset_name, method_name, configs)

    def train(self):
        temp_grads = {}
        for r in range(self.num_rounds):
            n = 0
            for name, param in self.model.named_parameters():
                temp_grads[name] = torch.zeros_like(param)
            sub_clients = self.sample_clients()
            for clt in sub_clients:
                n += clt.get_num_samples()
            for clt in sub_clients:
                clt.set_weights(self.model.state_dict())
                grads, error, acc = clt.solve_sgd()
                self.metrics.update(r, clt.name, error, acc, norm_grad_dict(grads))
                for key in grads.keys():
                    temp_grads[key] += clt.get_num_samples() / n * grads[key]

            for key in self.model.state_dict():
                self.model.state_dict()[key] -= self.lr * temp_grads[key]


class QFedSgdServer(BaseServer):
    def __init__(self, model, opt, lossf, clients, train_data, test_data,
                 dataset_name, method_name, configs=None):
        super().__init__(model, opt, lossf, clients, train_data, test_data,
                         dataset_name, method_name, configs)

    def train(self):
        deltas = {}
        hs = {}
        for r in range(self.num_rounds):
            for name, param in self.model.named_parameters():
                deltas[name] = []
                hs[name] = []

            sub_clients = self.sample_clients()
            for clt in sub_clients:
                clt.set_weights(self.model.state_dict())
                grads, error, acc = clt.solve_sgd()
                self.metrics.update(r, clt.name, error, acc, norm_grad_dict(grads))
                for key in grads.keys():
                    deltas[key].append(np.float_power(error + 1e-10, self.q) * grads[key])
                    hs[key].append(self.q * np.float_power(error + 1e-10, (self.q - 1)) *
                                   norm_grad(grads[key]) ** 2 + (1.0 / self.lr) * np.float_power(error + 1e-10, self.q))

            for key in self.model.state_dict():
                total_delta = functools.reduce(operator.add, deltas[key])
                total_h = functools.reduce(operator.add, hs[key])
                self.model.state_dict()[key] -= total_delta / total_h


class QFedAvgServer(BaseServer):
    def __init__(self, model, opt, lossf, clients, train_data, test_data,
                 dataset_name, method_name, configs=None):
        super().__init__(model, opt, lossf, clients, train_data, test_data,
                         dataset_name, method_name, configs)

    def train(self):
        simulated_grads = {}
        deltas = {}
        hs = {}
        for r in range(1,self.num_rounds+1):
            print('training round : {}'.format(r))
            for name, param in self.model.named_parameters():
                simulated_grads[name] = param.clone()
                deltas[name] = []
                hs[name] = []

            sub_clients = self.sample_clients()
            for clt in sub_clients:
                clt.set_weights(self.model.state_dict())
                ws, error, acc = clt.solve_avg(self.num_epochs, self.batch_size)
                self.metrics.update(r, clt.name, error, acc, None)
                for key in ws.keys():
                    simulated_grads[key] = self.model.state_dict()[key]-ws[key]
                    simulated_grads[key] *= 1.0/self.lr
                    deltas[key].append(np.float_power(error + 1e-10, self.q) * simulated_grads[key])
                    hs[key].append(self.q * np.float_power(error + 1e-10, (self.q - 1)) *
                                   norm_grad(simulated_grads[key]) ** 2
                                   + (1.0 / self.lr) * np.float_power(error + 1e-10, self.q))
            for key in self.model.state_dict():
                total_delta = functools.reduce(operator.add, deltas[key])
                total_h = functools.reduce(operator.add, hs[key])
                self.model.state_dict()[key] -= total_delta / total_h
            self.my_mod_evaluate(r)


class DL_FedAvgServer(BaseServer):
    def __init__(self, model, opt, lossf, clients, train_data, test_data,
                 dataset_name, method_name, configs=None):
        super().__init__(model, opt, lossf, clients, train_data, test_data,
                         dataset_name, method_name, configs)
    def DL_get_nks(self):
        return [c.get_num_samples()*c.get_lambda()+1e-20 for c in self.clients]

    def DL_sample_clients(self,):
        nks = self.DL_get_nks()
        return np.random.choice(
            self.clients,
            int(len(self.clients)*self.pct_client_per_round),
            p=[e/sum(nks) for e in nks],
            replace=False
        )

    def train(self,):
        for r in range(self.num_rounds):
            n = 0
            Ls = []
            sub_clients = self.DL_sample_clients()
            for clt in sub_clients:
                n += clt.get_num_samples()*clt.get_lambda()+1e-20
            for clt in sub_clients:
                clt.set_weights(self.model.state_dict())
                ws, error, acc = clt.solve_avg(self.num_epochs, self.batch_size)
                Ls.append(error)
                self.metrics.update(r, clt.name, error, acc, None)
                for key in ws.keys():
                    self.model.state_dict()[key] += ((clt.get_num_samples()*clt.get_lambda()+1e-20) / n) * ws[key]
            for clt,Li in zip(sub_clients,Ls):
                clt.update_lambda(clt.get_lambda()+self.s*np.abs(Li-sum(Ls)/len(Ls)))





