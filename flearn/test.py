

# if __name__ == '__main__':
#     net = MLP(10, 5, [64, 64])
#     data = torch.rand(4, 10)
#     labels = torch.rand(4)*10 // 5
#     print(data)
#     print(labels)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
#     optimizer.zero_grad()
#     outputs = net(data)
#     loss = criterion(outputs, labels.long())
#     print('loss', loss)
#     print('outputs', outputs)
#     # print('pre-grad', net.input_layer.weight.grad)
#     loss.backward()
#     optimizer.step()
#     # print('post-grad', net.input_layer.weight.grad)
#     # print('MODEL params:')
#     # grad_dict = {}
#     # for name, param in net.named_parameters():
#     #     grad_dict[name] = param.grad
#     # print(grad_dict)
#     print('state dict')
#     print(net.state_dict()['input_layer.bias'])