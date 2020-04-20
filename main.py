
from flearn.algo.client import Client
from flearn.algo.server import FedAvgServer, FedSgdServer, QFedSgdServer, QFedAvgServer
from flearn.model.neural_net import MLP
from flearn.utils import read_data
import torch.optim as optim
import torch.nn as nn


def run_app(train_dir,
            test_dir,
            configs
            ):
    # Logistic regression model
    layer_sizes = configs.get('layer_sizes')
    act_funcs = configs.get('act_funcs')
    lossf = nn.BCELoss() if act_funcs[-1] == 'sigmoid' else nn.NLLLoss()

    base_model = MLP(layer_sizes, act_funcs)
    base_opt = optim.SGD(params=base_model.parameters(), lr=0.05)

    client_names, groups, train_data, test_data = read_data(train_dir, test_dir)

    clients = []
    for c_name in client_names:
        clients.append(Client(c_name, [], train_data[c_name], test_data[c_name], base_model, base_opt, lossf))

    server = QFedAvgServer(base_model, base_opt, lossf, clients, train_data, test_data)
    server.train()
    server.evaluate()


if __name__ == '__main__':
    run_app('data/adult/data/train',
            'data/adult/data/test',
            {'layer_sizes': [99, 128, 128, 1], 'act_funcs': ['relu', 'relu', 'sigmoid']}
            )

