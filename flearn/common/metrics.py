
import os
import pandas as pd
from datetime import datetime


class Metrics:
    def __init__(self, client_names, params, dataset_name, metric_dir):
        self.client_names = client_names
        self.metrics = {}
        for key, val in params.items():
            self.metrics[key] = val
        self.metrics['cs'] = {
            'c_name': [],
            'round': [],
            'train_loss': [],
            'train_acc': [],
            'grad_norms': []
        }
        self.metric_dir = metric_dir
        self.dataset_name = dataset_name

    def update(self, rnd, c_name, train_loss, train_acc, grad_norm):
        self.metrics['cs']['round'].append(rnd)
        self.metrics['cs']['c_name'].append(c_name)
        self.metrics['cs']['train_loss'].append(train_loss)
        self.metrics['cs']['train_acc'].append(train_acc)
        self.metrics['cs']['grad_norms'].append(str(grad_norm))

    def write(self):
        write_path = os.path.join(self.metric_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(write_path, exist_ok=True)
        data_path = os.path.join(write_path, self.dataset_name + '.data.csv')
        meta_path = os.path.join(write_path, self.dataset_name + '.meta.csv')
        pd.DataFrame(self.metrics['cs']).to_csv(data_path, index=False)
        del self.metrics['cs']
        with open(meta_path, 'w') as f:
            for key in self.metrics.keys():
                f.writelines([key + ': ' + str(self.metrics[key]) + '\n'])






