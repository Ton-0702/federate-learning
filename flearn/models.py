
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        self.input_layer = nn.Linear(input_size, hidden_sizes[0])
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)
        self.hidden_layers = []
        for i in range(len(hidden_sizes) - 1):
            self.hidden_layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1]))

    def forward(self, data):
        data = F.relu(self.input_layer(data))
        for layer in self.hidden_layers:
            data = F.relu(layer(data))
        data = self.output_layer(data)
        return data


if __name__ == '__main__':
    net = MLP(10, 5, [64, 64])
    data = torch.rand(4, 10)
    labels = torch.rand(4)*10 // 5
    print(data)
    print(labels)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    optimizer.zero_grad()
    outputs = net(data)
    loss = criterion(outputs, labels.long())
    print('loss', loss)
    print('outputs', outputs)
    # print('pre-grad', net.input_layer.weight.grad)
    loss.backward()
    optimizer.step()
    # print('post-grad', net.input_layer.weight.grad)
    # print('MODEL params:')
    # grad_dict = {}
    # for name, param in net.named_parameters():
    #     grad_dict[name] = param.grad
    # print(grad_dict)
    print('state dict')
    print(net.state_dict()['input_layer.bias'])

