
import os
import pandas as pd
from datetime import datetime


class Metrics:
    def __init__(self, client_names, params, dataset_name, method_name, metric_dir):
        self.client_names = client_names
        self.metrics = {}
        if params is not None:
            for key, val in params.items():
                self.metrics[key] = val
        self.metrics['cs'] = {
            'c_name': [],
            'round': [],
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'grad_norms': []
        }
        self.metric_dir = metric_dir
        self.metrics['dataset_name'] = dataset_name
        self.metrics['method_name'] = method_name
        self.dataset_name = dataset_name

    def update(self, rnd, c_name,
               train_loss=None, train_acc=None,
               val_loss=None, val_acc=None,
               test_loss=None, test_acc=None,
               grad_norm=None):
        self.metrics['cs']['round'].append(rnd)
        self.metrics['cs']['c_name'].append(c_name)
        self.metrics['cs']['train_loss'].append(train_loss)
        self.metrics['cs']['train_acc'].append(train_acc)
        self.metrics['cs']['val_loss'].append(val_loss)
        self.metrics['cs']['val_acc'].append(val_acc)
        self.metrics['cs']['test_loss'].append(test_loss)
        self.metrics['cs']['test_acc'].append(test_acc)
        self.metrics['cs']['grad_norms'].append(str(grad_norm))

    def write(self):
        write_path = os.path.join(self.metric_dir,
                                  self.metrics['method_name'],
                                  datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(write_path, exist_ok=True)
        data_path = os.path.join(write_path, self.dataset_name + '.data.csv')
        meta_path = os.path.join(write_path, self.dataset_name + '.meta.csv')
        pkl_path = os.path.join(write_path, self.dataset_name + '.pkl')
        pd.DataFrame(self.metrics['cs']).to_csv(data_path, index=False)
        pd.DataFrame(self.metrics['cs']).to_pickle(pkl_path)
        # self.metrics['cs']
        with open(meta_path, 'w') as f:
            for key in self.metrics.keys():
                if key != 'cs':
                    f.writelines([key + ': ' + str(self.metrics[key]) + '\n'])






