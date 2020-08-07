from flearn.algo.client import Client
from flearn.algo.server import FedAvgServer, FedSgdServer, QFedSgdServer, QFedAvgServer
from flearn.model.mlp import MLP
from flearn.utils import read_data
import torch
import torch.optim as optim
import torch.nn as nn

SERVER = {
    'FedAvgServer': FedAvgServer,
    'FedSgdServer': FedSgdServer,
    'QFedSgdServer': QFedSgdServer,
    'QFedAvgServer': QFedAvgServer
}


def run_app(train_dir,
            test_dir,
            configs,
            server_configs=None,
            return_flags=False
            ):
    client_names, groups, train_data, test_data = read_data(train_dir, test_dir, torch.long)

    # Logistic regression model
    layer_sizes = configs.get('layer_sizes')
    act_funcs = configs.get('act_funcs')
    lossf = nn.BCELoss() if act_funcs[-1] == 'sigmoid' else nn.CrossEntropyLoss()

    base_model = MLP(layer_sizes, act_funcs)
    base_opt = optim.SGD(params=base_model.parameters(), lr=0.05)

    clients = []
    for c_name in client_names:
        clients.append(Client(c_name, [], train_data[c_name], test_data[c_name], base_model, base_opt, lossf))

    server = SERVER[configs['method_name']](model=base_model,
                                            opt=base_opt,
                                            lossf=lossf,
                                            clients=clients,
                                            train_data=train_data,
                                            test_data=test_data,
                                            dataset_name=configs['dataset_name'],
                                            method_name=configs['method_name']
                                            )
    server.train()
    server.evaluate()
    server.report()


if __name__ == '__main__':
    run_app(train_dir='data/synthetic/train/',
            test_dir='data/synthetic/test/',
            configs={
                'layer_sizes': [60, 10], 'act_funcs': ['softmax'],
                'dataset_name': 'synthetic',
                'method_name': 'QFedAvgServer'
            },
            server_configs={
                'num_rounds': 200,
                'pct_client_per_round': 0.1,
                'num_epochs': 1,
                'batch_size': 8,
                'lr': 0.1,
                'q': 1
            }
            )
