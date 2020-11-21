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
    def __init__(self, num_channels, num_phns, num_layers=17, kernel_size=3, stride=1, num_filters=16):
        super(SimpleCNN, self).__init__()

        padding = int((kernel_size-stride)/2)

        # create module list
        self.layers = []
        self.layers.append(nn.Conv1d(in_channels=num_channels, out_channels=num_filters*4, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers.append(nn.BatchNorm1d(num_filters*4))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Dropout(0.25))
        self.layers.append(nn.Conv1d(in_channels=num_filters*4, out_channels=num_filters*2, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers.append(nn.BatchNorm1d(num_filters*2))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Dropout(0.25))
        self.layers.append(nn.Conv1d(in_channels=num_filters*2, out_channels=num_filters, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers.append(nn.BatchNorm1d(num_filters))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Dropout(0.25))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(11*num_filters, num_phns))

        self.model = nn.ModuleList(self.layers)
        init_weights(self.model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x
