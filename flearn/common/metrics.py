
import os
import json


class Metrics:
    def __init__(self, clients, params):
        self.params = params
        num_rounds = params['num_rounds']
        self.bytes_written = {c.id: [0] * num_rounds for c in clients}
        self.client_computations = {c.id: [0] * num_rounds for c in clients}
        self.bytes_read = {c.id: [0] * num_rounds for c in clients}
        self.accuracies = []
        self.train_accuracies = []

    def update(self, rnd, cid, stats):
        bytes_w, comp, bytes_r = stats
        self.bytes_written[cid][rnd] += bytes_w
        self.client_computations[cid][rnd] += comp
        self.bytes_read[cid][rnd] += bytes_r

    def write(self):
        metrics = {'dataset': self.params['dataset'],
                   'num_rounds': self.params['num_rounds'],
                   'eval_every': self.params['eval_every'],
                   'learning_rate': self.params['learning_rate'],
                   'mu': self.params['mu'],
                   'num_epochs': self.params['num_epochs'],
                   'batch_size': self.params['batch_size'],
                   'accuracies': self.accuracies,
                   'train_accuracies': self.train_accuracies,
                   'client_computations': self.client_computations,
                   'bytes_written': self.bytes_written,
                   'bytes_read': self.bytes_read}
        metrics_dir = os.path.join('out', self.params['dataset'],
                                   'metrics_{}_{}_{}_{}_{}.json'.format(self.params['seed'],
                                                                        self.params['optimizer'],
                                                                        self.params['learning_rate'],
                                                                        self.params['num_epochs'],
                                                                        self.params['mu']))
        if not os.path.exists(os.path.join('out', self.params['dataset'])):
            os.mkdir(os.path.join('out', self.params['dataset']))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)


