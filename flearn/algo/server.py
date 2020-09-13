from tqdm import tqdm
import torch
import numpy as np
import functools
import operator
import copy
from ..common.metrics import Metrics


def norm_grad(x):
    return torch.norm(x)


def norm_grad_flatten(dct):
    arr = []
    for key in dct.keys():
        arr.append(torch.flatten(dct[key]))
    return norm_grad(torch.cat(arr))


def deep_copy_state_dict(sd):
    res = {}
    for key in sd:
        res[key] = sd[key].clone()
    return res


def norm_grad_dict(grads):
    grad_norms = {}
    for key in grads.keys():
        grad_norms[key] = norm_grad(grads[key])
    return grad_norms


class BaseServer:
    def __init__(self, model, opt, lossf, clients, train_data, test_data,
                 dataset_name, method_name, configs=None,
                 metric_dir='experiments'):

        default_configs = {
            # Model configs
            'layer_sizes': [60, 10], 'act_funcs': ['softmax'],
            'dataset_name': 'synthetic',
            'method_name': 'QFedAvgServer',
            # Server configs
            'num_rounds': 200,
            'pct_client_per_round': 0.1,
            'num_epochs': 1,
            'batch_size': 10,
            'lr': 0.1,
            'q': 1,
            'disable_tqdm': True
        }

        if configs is not None:
            default_configs.update(configs)
            for key, val in default_configs.items():
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
        self.disable_tqdm = configs['disable_tqdm'] if 'disable_tqdm' in configs else False

        self.metrics = Metrics([c.name for c in self.clients], default_configs,
                               self.dataset_name, self.method_name, self.metrics_dir)

    def save_model(self):
        pass

    def evaluate(self):
        for clt in self.clients:
            clt.set_weights(self.model.state_dict())
            train_acc = clt.get_train_accuracy()
            test_acc = clt.get_test_accuracy()
            self.metrics.update(-1, clt.name, train_acc, test_acc, None)

    def evaluate_round(self, r):
        y_true_train = np.array([])
        y_pred_train = np.array([])
        y_true_val = np.array([])
        y_pred_val = np.array([])
        y_true_test = np.array([])
        y_pred_test = np.array([])
        for clt in self.clients:
            clt.set_weights(self.model.state_dict())

            y_prob_train = clt.model(clt.train_data['x'])
            y_prob_val = clt.model(clt.val_data['x'])
            y_prob_test = clt.model(clt.test_data['x'])

            # if len(y_prob_train.shape) == 1:
            #     y_bar_train = (y_prob_train > 0.5).type(torch.float).numpy()
            #     y_bar_val = (y_prob_val > 0.5).type(torch.float).numpy()
            #     y_bar_test = (y_prob_test > 0.5).type(torch.float).numpy()
            # else:
            #     y_bar_train = y_prob_train.max(dim=1)[1].numpy()
            #     y_bar_val = y_prob_val.max(dim=1)[1].numpy()
            #     y_bar_test = y_prob_test.max(dim=1)[1].numpy()

            y_bar_train = torch.sign(y_prob_train).detach().numpy()
            y_bar_val = torch.sign(y_prob_val).detach().numpy()
            y_bar_test = torch.sign(y_prob_test).detach().numpy()

            y_true_train = np.append(y_true_train, clt.train_data['y'].numpy())
            y_pred_train = np.append(y_pred_train, y_bar_train) # clt.model(clt.train_data['x']).max(dim=1)[1].numpy()

            y_true_val = np.append(y_true_val, clt.val_data['y'].numpy())
            y_pred_val = np.append(y_pred_val, y_bar_val)

            y_true_test = np.append(y_true_test, clt.test_data['y'].numpy())
            y_pred_test = np.append(y_pred_test, y_bar_test)

            train_loss = clt.get_train_error()
            train_acc = clt.get_train_accuracy()
            val_loss = clt.get_val_error()
            val_acc = clt.get_val_accuracy()
            test_loss = clt.get_test_error()
            test_acc = clt.get_test_accuracy()
            # if val_loss is None:
            #     print('Val loss None')
            # print(clt.name, "val_loss", val_loss, "val_acc", val_acc, "test_loss", test_loss, "test_acc", test_acc)
            self.metrics.update(rnd=r,
                                c_name=clt.name,
                                train_loss=train_loss,
                                train_acc=train_acc,
                                val_loss=val_loss,
                                val_acc=val_acc,
                                test_loss=test_loss,
                                test_acc=test_acc,
                                grad_norm=None)

        # Global accuracy
        global_train_acc = (y_true_train == y_pred_train).sum() / y_true_train.shape[0]
        global_val_acc = (y_true_val == y_pred_val).sum() / y_true_val.shape[0]
        global_test_acc = (y_true_test == y_pred_test).sum()/y_true_test.shape[0]
        self.metrics.update(rnd=r,
                            c_name='global',
                            train_acc=global_train_acc,
                            val_acc=global_val_acc,
                            test_acc=global_test_acc
                            )

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
            int(len(self.clients) * self.pct_client_per_round),
            p=[e / sum(nks) for e in nks],
            replace=False
        )


class FedAvgServer(BaseServer):
    def __init__(self, model, opt, lossf, clients, train_data, test_data,
                 dataset_name, method_name, configs=None):
        super().__init__(model, opt, lossf, clients, train_data, test_data,
                         dataset_name, method_name, configs)

    def get_nks(self):
        return [c.get_num_samples() * c.get_lambda() + 1e-20 for c in self.clients]

    def sample_clients(self):
        nks = self.get_nks()
        return np.random.choice(
            self.clients,
            int(len(self.clients) * self.pct_client_per_round),
            p=[e / sum(nks) for e in nks],
            replace=False
        )

    def train(self):
        for r in tqdm(range(self.num_rounds), disable=self.disable_tqdm):
            n = 0
            Ls = []
            sub_clients = self.sample_clients()
            for clt in sub_clients:
                n += clt.get_num_samples()*clt.get_lambda()+1e-20
            for clt in sub_clients:
                clt.set_weights(self.model.state_dict())
                ws, error, acc = clt.solve_avg(self.num_epochs, self.batch_size)
                Ls.append(error)
                # self.metrics.update(r, clt.name, error, acc, None)
                for key in ws.keys():
                    self.model.state_dict()[key] += ((clt.get_num_samples()*clt.get_lambda()+1e-20) / n) * ws[key]
            # for clt, Li in zip(sub_clients, Ls):
            #     clt.update_lambda(clt.get_lambda() + self.s * np.abs(Li - sum(Ls) / len(Ls)))
            self.evaluate_round(r)

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
                # self.metrics.update(r, clt.name, error, acc, norm_grad_dict(grads))
                for key in grads.keys():
                    temp_grads[key] += clt.get_num_samples() / n * grads[key]

            for key in self.model.state_dict():
                self.model.state_dict()[key] -= self.lr * temp_grads[key]


class QFedSgdServer(BaseServer):
    def __init__(self, model, opt, lossf, clients, train_data, test_data,
                 dataset_name, method_name, configs=None, disable_tqdm=False):
        super().__init__(model, opt, lossf, clients, train_data, test_data,
                         dataset_name, method_name, configs, disable_tqdm)

    def train(self):
        deltas = {}
        hs = {}
        for r in tqdm(range(1, self.num_rounds + 1), disable=self.disable_tqdm):
            # print(f"Training round {r}")
            for name, param in self.model.named_parameters():
                deltas[name] = []
                hs[name] = []

            sub_clients = self.sample_clients()
            for clt in sub_clients:
                clt.set_weights(self.model.state_dict())
                grads, error, acc = clt.solve_sgd()
                # self.metrics.update(r, clt.name, error, acc, norm_grad_dict(grads))
                for key in grads.keys():
                    deltas[key].append(np.float_power(error + 1e-10, self.q) * grads[key])
                    hs[key].append(self.q * np.float_power(error + 1e-10, (self.q - 1)) *
                                   norm_grad_flatten(grads) ** 2 + (1.0 / self.lr)
                                   * np.float_power(error + 1e-10, self.q))

            for key in self.model.state_dict():
                total_delta = functools.reduce(operator.add, deltas[key])
                total_h = functools.reduce(operator.add, hs[key])
                self.model.state_dict()[key] -= total_delta / total_h
            self.evaluate_round(r)


class QFedAvgServer(BaseServer):
    def __init__(self, model, opt, lossf, clients, train_data, test_data,
                 dataset_name, method_name, configs=None, disable_tqdm=False):
        super().__init__(model, opt, lossf, clients, train_data, test_data,
                         dataset_name, method_name, configs, disable_tqdm)

    def train(self):
        simulated_grads = {}
        deltas = {}
        hs = []
        for r in tqdm(range(1, self.num_rounds + 1), disable=self.disable_tqdm):
            for name, param in self.model.named_parameters():
                simulated_grads[name] = param.clone()
                deltas[name] = []

            sub_clients = self.sample_clients()
            pre_weight = deep_copy_state_dict(self.model.state_dict())  # Trung fixed
            for clt in sub_clients:
                clt.set_weights(self.model.state_dict())
                error = clt.get_train_error() # Huy fixed
                ws, _error, acc = clt.solve_avg(self.num_epochs, self.batch_size)
                # self.metrics.update(rnd=r, c_name=clt.name, train_loss=error, train_acc=acc, grad_norm=None)
                for key in ws.keys():
                    simulated_grads[key] = pre_weight[key] - ws[key]  # Trung & Giang fixed
                    simulated_grads[key] *= 1.0/self.lr
                    deltas[key].append(np.float_power(error + 1e-10, self.q) * simulated_grads[key])
                hs.append(
                    self.q * np.float_power(error + 1e-10, (self.q - 1)) *
                    norm_grad_flatten(simulated_grads) ** 2
                    + (1.0 / self.lr) * np.float_power(error + 1e-10, self.q)
                )  # Trung fixed norm_grad_flatten
            for key in self.model.state_dict():
                total_delta = torch.sum(torch.stack(deltas[key]), dim=0)
                total_h = sum([e.item() for e in hs])
                self.model.state_dict()[key] -= total_delta / total_h
            self.evaluate_round(r)

class DL_FedAvgServer(BaseServer):
    def __init__(self, model, opt, lossf, clients, train_data, test_data,
                 dataset_name, method_name, configs=None, disable_tqdm=False):
        super().__init__(model, opt, lossf, clients, train_data, test_data,
                         dataset_name, method_name, configs, disable_tqdm)
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
    # temp_model=self.model()
    def train(self,):
        for r in tqdm(range(self.num_rounds), disable=self.disable_tqdm):
            n = 0
            Ls = []
            sub_clients = self.sample_clients()
            for clt in sub_clients:
                n += clt.get_num_samples()*clt.get_lambda()+1e-20
            # print("round",r,";sum probability",sum([(clt.get_num_samples()*clt.get_lambda()+1e-20) / n for clt in sub_clients]))

            pre_weight = deep_copy_state_dict(self.model.state_dict())
            # print("pr_bf_lp",self.model.state_dict()[list(pre_weight.keys())[0]][1][:5])
            for clt in sub_clients:
                # print(clt.name)
                # print("sv_bf_cl",self.model.state_dict()[list(pre_weight.keys())[0]][1][:5])
                clt.set_weights(pre_weight)
                ws, error, acc = clt.solve_avg(self.num_epochs, self.batch_size)
                Ls.append(error)
                # self.metrics.update(r, clt.name, error, acc, None)
                # print("sv_bf_up",self.model.state_dict()[list(pre_weight.keys())[0]][1][:5])
                for key in ws.keys():
                    update_weight=((clt.get_num_samples()*clt.get_lambda()+1e-20) / n) * ws[key].clone()
                    if clt!=sub_clients[0]:
                        self.model.state_dict()[key].copy_(self.model.state_dict()[key] + update_weight) 
                    else:
                        self.model.state_dict()[key].copy_(update_weight)
                    
                # print("sv_at_up",self.model.state_dict()[list(pre_weight.keys())[0]][1][:5])
            for clt,Li in zip(sub_clients,Ls):
                new_lambda=clt.get_lambda()+self.s*np.abs(Li-sum(Ls)/len(Ls))
                clt.update_lambda(new_lambda)
            # print(Ls)
            k = 0
            for clt in self.clients:
                k += clt.get_lambda()
            for clt in self.clients:
                clt.update_lambda(clt.get_lambda()/k)

            self.evaluate_round(r)
            pass

