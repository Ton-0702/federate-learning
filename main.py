
from flearn.algo.client import Client
from flearn.algo.server import FedAvgServer, FedSgdServer, QFedSgdServer, QFedAvgServer
from flearn.model.mlp import MLP
from flearn.utils import read_data
import torch
import torch.optim as optim
import torch.nn as nn


def run_app(train_dir,
            test_dir,
            configs
            ):
    client_names, groups, train_data, test_data = read_data(train_dir, test_dir, torch.long)

    # Logistic regression model
    layer_sizes = configs.get('layer_sizes')
    act_funcs = configs.get('act_funcs')
    lossf = nn.BCELoss() if act_funcs[-1] == 'sigmoid' else nn.NLLLoss()

    base_model = MLP(layer_sizes, act_funcs)
    base_opt = optim.SGD(params=base_model.parameters(), lr=0.05)

    clients = []
    for c_name in client_names:
        clients.append(Client(c_name, [], train_data[c_name], test_data[c_name], base_model, base_opt, lossf))

    server = QFedAvgServer(base_model, base_opt, lossf, clients, train_data, test_data, configs['dataset_name'])
    server.train()
    server.evaluate()
    server.report()


if __name__ == '__main__':
    run_app('data/synthetic/train/',
            'data/synthetic/test/',
            {
                'layer_sizes': [60, 10], 'act_funcs': ['softmax'],
                'dataset_name': 'synthetic'
             },
            )

