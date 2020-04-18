
import random
import torch


class BaseServer:
    def __init__(self, model, opt, lossf, clients, train_data, test_data, configs=None):
        if configs is None:
            configs = {
                'num_rounds': 15,
                'pct_client_per_round': 1,
                'num_epochs': 3,
                'batch_size': 8,
                'lr': 0.1
            }
        for key, val in configs.items():
            setattr(self, key, val)

        self.model = model
        self.opt = opt
        self.lossf = lossf
        self.clients = clients
        self.train_data = train_data
        self.test_data = test_data

    def save_model(self):
        pass

    def evaluate(self):
        for clt in self.clients:
            clt.set_weights(self.model.state_dict())
            print('Name: ', clt.name)
            print('Train accuracy: ', clt.get_train_accuracy())
            print('Test accuracy: ', clt.get_test_accuracy())

    def train(self):
        pass

    def sample_clients(self):
        return random.sample(
            self.clients,
            int(self.pct_client_per_round * len(self.clients))
        )


class FedAvgServer(BaseServer):
    def __init__(self, model, opt, lossf, clients, train_data, test_data, configs=None):
        super().__init__(model, opt, lossf, clients, train_data, test_data, configs)

    def train(self):
        n = 0
        for r in range(self.num_rounds):
            sub_clients = self.sample_clients()
            for clt in sub_clients:
                n += clt.get_num_samples()
            for clt in sub_clients:
                clt.set_weights(self.model.state_dict())
                ws, error = clt.solve_avg(self.num_epochs, self.batch_size)
                for key in ws.keys():
                    self.model.state_dict()[key] += clt.get_num_samples() / n * ws[key]


class FedSgdServer(BaseServer):
    def __init__(self, model, opt, lossf, clients, train_data, test_data, configs=None):
        super().__init__(model, opt, lossf, clients, train_data, test_data, configs)

    def train(self):
        n = 0
        temp_grads = {}
        for r in range(self.num_rounds):
            for name, param in self.model.named_parameters():
                temp_grads[name] = torch.zeros_like(param)
            sub_clients = self.sample_clients()
            for clt in sub_clients:
                n += clt.get_num_samples()
            for clt in sub_clients:
                clt.set_weights(self.model.state_dict())
                grads, error = clt.solve_sgd()
                for key in grads.keys():
                    temp_grads[key] += clt.get_num_samples() / n * grads[key]

            for key in self.model.state_dict():
                self.model.state_dict()[key] -= self.lr * temp_grads[key]


# class QFedSgdServer(BaseServer):
#     def __init__(self, model, opt, lossf, clients, train_data, test_data, configs=None):
#         super().__init__(model, opt, lossf, clients, train_data, test_data, configs)
#
#     def train(self):
#         n = 0
#         for r in range(self.num_rounds):
#             temp_grads = {}
#             sub_clients = self.sample_clients()
#             for clt in sub_clients:
#                 n += clt.get_num_samples()
#             for clt in sub_clients:
#                 grads, error = clt.solve_sgd()
#                 for key in grads.keys():
#                     temp_grads[key] += clt.get_num_samples() / n * grads[key]
#
#             for key in self.model.state_dict():
#                 self.model.state_dict()[key] -= self.lr * temp_grads[key]

