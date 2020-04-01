
from flearn.models import MLP
from flearn.fed_avg import server, Client
from flearn.utils import read_data, auc
import torch.optim as optim
import torch.nn as nn


def run_app():
    train_dir = 'data/adult/data/train'
    test_dir = 'data/adult/data/test'

    client_names, groups, train_data, test_data = read_data(train_dir, test_dir)

    clients = []
    for c_name in client_names:
        clients.append(Client(c_name, train_data[c_name]))

    # Logistic regression model
    base_model = MLP([99, 1], ['sigmoid'])
    base_opt = optim.SGD(params=base_model.parameters(), lr=0.05)
    loss_func = nn.BCELoss()

    model = server(base_model,
                   base_opt,
                   loss_func,
                   clients,
                   num_rounds=1)

    # Use trained model to predict test dataset
    phd_scores = model(test_data['phd']['x'])
    phd_auc = auc(test_data['phd']['y'], phd_scores)
    non_phd_scores = model(test_data['non-phd']['x'])
    non_phd_auc = auc(test_data['non-phd']['y'], non_phd_scores)

    print('phd auc: ', phd_auc)
    print('non phd auc: ', non_phd_auc)


if __name__ == '__main__':
    run_app()

