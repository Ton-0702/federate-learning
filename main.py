from flearn.algo.client import Client, DL_Client
from flearn.algo.server import FedAvgServer, FedSgdServer, QFedSgdServer, QFedAvgServer, DL_FedAvgServer
from flearn.model.mlp import MLP
from flearn.utils import read_data
import torch
import torch.optim as optim
import torch.nn as nn


def run_app(train_dir,
            test_dir,
            configs,
            server_configs=None,
            return_flag=False,
            ):
    client_names, groups, train_data, test_data = read_data(train_dir, test_dir, torch.long)

    # Logistic regression model
    layer_sizes = configs.get('layer_sizes')
    act_funcs = configs.get('act_funcs')
    lossf = nn.BCELoss() if act_funcs[-1] == 'sigmoid' else nn.CrossEntropyLoss()

    base_model = MLP(layer_sizes, act_funcs)
    base_opt = optim.SGD(params=base_model.parameters(), lr=0.05)
    if 'DL_' in configs['method_name']:
        clients = []
        for c_name in client_names:
            clients.append(DL_Client(c_name, [], train_data[c_name], test_data[c_name], base_model, base_opt, lossf))
    else:
        clients = []
        for c_name in client_names:
            clients.append(Client(c_name, [], train_data[c_name], test_data[c_name], base_model, base_opt, lossf))
    switch={
      'FedAvgServer':FedAvgServer,
      'FedSgdServer':FedSgdServer,
      'QFedSgdServer':QFedSgdServer,
      'QFedAvgServer':QFedAvgServer,
      'DL_FedAvgServer':DL_FedAvgServer,
    }
    server = switch[configs['method_name']](model=base_model,
                          opt=base_opt,
                          lossf=lossf,
                          clients=clients,
                          train_data=train_data,
                          test_data=test_data,
                          dataset_name=configs['dataset_name'],
                          method_name=configs['method_name'],
                          configs=server_configs,
                          )
    server.train()
    # server.evaluate()
    server.report()
    if return_flag:
      return server


if __name__ == '__main__':
    run_app('data/synthetic/train/',
            'data/synthetic/test/',
            {
                'layer_sizes': [60, 10], 'act_funcs': ['softmax'],
                'dataset_name': 'synthetic',
                'method_name': 'QFedAvgServer'
            },
            )
