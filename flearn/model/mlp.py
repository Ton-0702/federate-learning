
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, layer_sizes, act_funcs=['sigmoid']):
        super(MLP, self).__init__()
        assert len(layer_sizes) - 1 == len(act_funcs)
        self.layer_sizes = layer_sizes
        self.act_funcs = act_funcs
        self.layers = nn.ModuleList()
        for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
            layer = nn.Linear(in_size, out_size)
            self.layers.append(layer)

    def forward(self, x):
        for layer, act_func in zip(self.layers, self.act_funcs):
            x = layer(x)
            if act_func == 'relu':
                x = torch.relu(x)
            elif act_func == 'sigmoid':
                x = torch.sigmoid(x)
            elif act_func == 'softmax':
                pass
                # x = torch.softmax(x, dim=-1)
            elif act_func == 'none':
                x = layer(x)
        if self.layer_sizes[-1] == 1:
            return x.squeeze(-1)
        return x


