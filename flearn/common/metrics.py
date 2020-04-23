
import os
import json


class Metrics:
    def __init__(self, client_names, params, dataset_name, metric_dir):
        self.client_names = client_names
        self.metrics = {}
        for key, val in params.items():
            self.metrics[key] = val
        self.metrics['cs'] = {}
        self.metric_dir = metric_dir
        self.dataset_name = dataset_name

    def update(self, rnd, c_name, train_loss, train_acc, grad_norm):
        self.metrics['cs'][c_name].append([rnd, train_loss, train_acc, grad_norm])

    def write(self):
        write_path = os.path.join(self.metric_dir, self.dataset_name + '.json')
        os.makedirs(write_path, exist_ok=True)
        with open(write_path, 'w') as f:
            json.dump(self.metrics, f)




