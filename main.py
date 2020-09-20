from flearn.algo.client import Client
from flearn.algo.server import FedAvgServer, FedSgdServer, QFedSgdServer, QFedAvgServer, DL_FedAvgServer
from flearn.model.mlp import MLP
from flearn.model.vehicle.svm import SVM, HingeLoss
from flearn.utils import read_data
import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


SERVER = {
    'FedAvgServer': FedAvgServer,
    'FedSgdServer': FedSgdServer,
    'QFedSgdServer': QFedSgdServer,
    'QFedAvgServer': QFedAvgServer,
    "DL_FedAvgServer":DL_FedAvgServer,
}


def run_app(train_dir,
            test_dir,
            configs=dict(),
            report=True
            ):
    client_names, groups, train_data, test_data = read_data(train_dir, test_dir, torch.long)

    # Logistic regression model
    layer_sizes = configs.get('layer_sizes')
    act_funcs = configs.get('act_funcs')
    lr = configs.get('lr')
    lossf = nn.BCELoss() if act_funcs[-1] == 'sigmoid' else nn.CrossEntropyLoss()

    # base_model = MLP(layer_sizes, act_funcs)
    
    clients = []
    base_model={}
    base_opt={}
    for c_name in client_names:
        base_model[c_name] = MLP(layer_sizes, act_funcs)
        base_opt[c_name] = optim.SGD(params=base_model[c_name].parameters(), lr=lr, weight_decay=1e-3)
        clients.append(Client(c_name, [], train_data[c_name], test_data[c_name], base_model[c_name], base_opt[c_name], lossf))
    base_model_sv = MLP(layer_sizes, act_funcs)
    base_opt_sv = optim.SGD(params=base_model_sv.parameters(), lr=lr)
    server = SERVER[configs['method_name']](model=base_model_sv,
                                            opt=base_opt_sv,
                                            lossf=lossf,
                                            clients=clients,
                                            train_data=train_data,
                                            test_data=test_data,
                                            dataset_name=configs['dataset_name'],
                                            method_name=configs['method_name'],
                                            configs=configs
                                            )
    server.train()
    # server.evaluate()
    if report:
        server.report()

    return server, clients


def run_vehicle(train_dir,
                test_dir,
                configs=dict(),
                report=True
                ):
    client_names, groups, train_data, test_data = read_data(train_dir, test_dir, torch.long)

    # Logistic regression model
    layer_sizes = configs.get('layer_sizes')
    act_funcs = configs.get('act_funcs')
    lr = configs.get('lr')
    # lossf = nn.BCELoss() if act_funcs[-1] == 'sigmoid' else nn.CrossEntropyLoss()
    lossf = HingeLoss()
    # base_model = MLP(layer_sizes, act_funcs)

    clients = []
    base_model = {}
    base_opt = {}
    for c_name in client_names:
        base_model[c_name] = SVM(layer_sizes, act_funcs)
        base_opt[c_name] = optim.SGD(params=base_model[c_name].parameters(), lr=lr)
        clients.append(
            Client(c_name, [], train_data[c_name], test_data[c_name], base_model[c_name], base_opt[c_name], lossf))
    base_model_sv = MLP(layer_sizes, act_funcs)
    base_opt_sv = optim.SGD(params=base_model_sv.parameters(), lr=lr)
    server = SERVER[configs['method_name']](model=base_model_sv,
                                            opt=base_opt_sv,
                                            lossf=lossf,
                                            clients=clients,
                                            train_data=train_data,
                                            test_data=test_data,
                                            dataset_name=configs['dataset_name'],
                                            method_name=configs['method_name'],
                                            configs=configs
                                            )
    server.train()
    # server.evaluate()
    if report:
        server.report()

    return server, clients


if __name__ == '__main__':
    # run_app(train_dir='data/synthetic/train/',
    #         test_dir='data/synthetic/test/',
    #         configs={
    #             # Model configs
    #             'layer_sizes': [60, 10], 'act_funcs': ['softmax'],
    #             'dataset_name': 'synthetic',
    #             'method_name': 'FedAvgServer',
    #             # Server configs
    #             'num_rounds': 2000,
    #             'pct_client_per_round': 0.1,
    #             'num_epochs': 1,
    #             'batch_size': 10,
    #             'lr': 0.1,
    #             'q': 1
    #         }
    #         )
    run_vehicle(train_dir='data/vehicle/train/',
            test_dir='data/vehicle/test/',
            configs={
                # Model configs
                'layer_sizes': [100, 1], 'act_funcs': ['none'],
                'dataset_name': 'vehicle',
                'method_name': 'QFedAvgServer',
                # Server configs
                'num_rounds': 20,
                'pct_client_per_round': 10.0 / 23.0,
                'num_epochs': 1,
                'batch_size': 64,
                'lr': 0.01,
                's':0.1,
                'q': 0,
                'disable_tqdm': True
            },
            report=False
            )