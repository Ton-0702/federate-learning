
import pandas as pd
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import os
import glob
import json
import torch


def read_data(train_data_dir, test_data_dir, convert_tensor=True):
    """parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users

    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    """
    clients = []
    groups = []
    train_data = {}
    test_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json')]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json')]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    if convert_tensor:
        for client in clients:
            train_data[client]['x'] = torch.tensor(train_data[client]['x'],
                                                   dtype=torch.float32)
            train_data[client]['y'] = torch.tensor(train_data[client]['y'],
                                                   dtype=torch.long)
            test_data[client]['x'] = torch.tensor(test_data[client]['x'],
                                                  dtype=torch.float32)
            test_data[client]['y'] = torch.tensor(test_data[client]['y'],
                                                  dtype=torch.long)

    return clients, groups, train_data, test_data


class FEDDataset(data.Dataset):
    def __init__(self, x, y):
        self.data_x = x
        self.data_y = y

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]

    def get_raw_data(self):
        return self.data_x, self.data_y


