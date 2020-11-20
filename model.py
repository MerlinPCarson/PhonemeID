import torch
import torch.nn as nn
import math


def init_weights(model):
    for layer in model:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_normal_(layer.weight.data, a=0, mode='fan_in')
        if isinstance(layer, nn.BatchNorm1d):
            layer.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
            nn.init.constant_(layer.bias.data, 0.0)


class SimpleCNN(nn.Module):
    def __init__(self, num_channels, num_phns, num_layers=17, kernel_size=3, stride=1, num_filters=64):
        super(SimpleCNN, self).__init__()

        # create module list
        self.layers = []
        self.layers.append(nn.Conv1d(in_channels=num_channels, 
                                     out_channels=num_filters, kernel_size=kernel_size, bias=True))
        self.layers.append(nn.BatchNorm1d(num_filters))
        self.layers.append(nn.ReLU(inplace=True))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(9*num_filters, num_phns))

        self.model = nn.ModuleList(self.layers)
        init_weights(self.model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x 
