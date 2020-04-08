
from flearn.models import MLP
from flearn.fed_avg import server, Client
from flearn.utils import read_data, auc
import torch.optim as optim
import torch.nn as nn


def run_app(train_dir,
            test_dir,
            configs
            ):

    client_names, groups, train_data, test_data = read_data(train_dir, test_dir)

    clients = []
    for c_name in client_names:
        clients.append(Client(c_name, train_data[c_name]))

    # Logistic regression model
    layer_sizes = configs.get('layer_sizes')
    act_funcs = configs.get('act_funcs')
    loss_func = nn.BCELoss() if act_funcs[-1] == 'sigmoid' else nn.NLLLoss()
    num_rounds = configs.get('num_rounds', 1)

    base_model = MLP(layer_sizes, act_funcs)
    base_opt = optim.SGD(params=base_model.parameters(), lr=0.05)

    model = server(base_model,
                   base_opt,
                   loss_func,
                   clients,
                   num_rounds)

    # Use trained model to predict test dataset
    multi_class = True if layer_sizes[-1] > 1 else False
    for i, c_name in enumerate(client_names):
        if multi_class:
            i_name = int(c_name.split(':')[0])
            scores = model(test_data[c_name]['x'])[:, i_name]
        else:
            scores = model(test_data[c_name]['x'])
        auc_score = auc(test_data[c_name]['y'], scores)
        print('client: ', c_name)
        print('num samples: ', test_data[c_name]['x'].shape)
        print('AUC: ', auc_score)


if __name__ == '__main__':
    run_app('data/adult/data/train',
            'data/adult/data/test',
            {'layer_sizes': [99, 128, 1], 'act_funcs': ['relu', 'sigmoid']}
            )

