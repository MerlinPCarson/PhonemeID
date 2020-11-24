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

class SimpleCNN2D(nn.Module):
    def __init__(self, num_channels, num_phns, num_layers=17, kernel_size=3, stride=1, num_filters=16):
        super(SimpleCNN2D, self).__init__()

        padding = int((kernel_size-stride)/2)

        # create module list
        self.layers = []
        self.layers.append(nn.Conv2d(in_channels=1, out_channels=num_filters, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers.append(nn.BatchNorm2d(num_filters))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Dropout(0.25))
        self.layers.append(nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers.append(nn.BatchNorm2d(num_filters*2))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Dropout(0.25))
        self.layers.append(nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers.append(nn.BatchNorm2d(num_filters*4))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Dropout(0.25))
        self.layers.append(nn.Conv2d(in_channels=num_filters*4, out_channels=num_filters*2, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers.append(nn.BatchNorm2d(num_filters*2))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Dropout(0.25))
        self.layers.append(nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers.append(nn.BatchNorm2d(num_filters))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Dropout(0.25))

        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(13*11*num_filters, 250))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Dropout(0.25))
        self.layers.append(nn.Linear(250, num_phns))

        self.model = nn.ModuleList(self.layers)
        #init_weights(self.model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class DualCNN2D(nn.Module):
    def __init__(self, num_channels, num_phns, num_layers=17, kernel_size=3, stride=1, num_filters=16):
        super(DualCNN2D, self).__init__()

        padding = int((kernel_size-stride)/2)
        padding = 0 

        # create module list
        self.layers_mfccs = []
        self.layers_mfccs.append(nn.Conv2d(in_channels=1, out_channels=num_filters, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers_mfccs.append(nn.BatchNorm2d(num_filters))
        self.layers_mfccs.append(nn.PReLU())
        self.layers_mfccs.append(nn.Dropout(0.25))
        self.layers_mfccs.append(nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers_mfccs.append(nn.BatchNorm2d(num_filters*2))
        self.layers_mfccs.append(nn.PReLU())
        self.layers_mfccs.append(nn.Dropout(0.25))
        self.layers_mfccs.append(nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers_mfccs.append(nn.BatchNorm2d(num_filters*4))
        self.layers_mfccs.append(nn.PReLU())
        self.layers_mfccs.append(nn.Dropout(0.25))
        self.layers_mfccs.append(nn.Flatten())

        self.model_mfccs = nn.ModuleList(self.layers_mfccs)
        init_weights(self.model_mfccs)

        # create module list
        self.layers_mels = []
        self.layers_mels.append(nn.Conv2d(in_channels=1, out_channels=num_filters, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers_mels.append(nn.BatchNorm2d(num_filters))
        self.layers_mels.append(nn.PReLU())
        self.layers_mels.append(nn.Dropout(0.25))
        self.layers_mels.append(nn.Conv2d(in_channels=num_filters, out_channels=num_filters*2, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers_mels.append(nn.BatchNorm2d(num_filters*2))
        self.layers_mels.append(nn.PReLU())
        self.layers_mels.append(nn.Dropout(0.25))
        self.layers_mels.append(nn.Conv2d(in_channels=num_filters*2, out_channels=num_filters*4, 
                                     kernel_size=kernel_size, bias=True, padding=padding))
        self.layers_mels.append(nn.BatchNorm2d(num_filters*4))
        self.layers_mels.append(nn.PReLU())
        self.layers_mels.append(nn.Dropout(0.25))
        self.layers_mels.append(nn.Flatten())

        self.model_mels = nn.ModuleList(self.layers_mels)
        init_weights(self.model_mels)

        # create module list
        self.layers_class = []
        self.layers_class.append(nn.Linear(7*5*num_filters*4 + 16*5*num_filters*4, 500))
        self.layers_class.append(nn.PReLU())
        self.layers_mels.append(nn.Dropout(0.25))
        self.layers_class.append(nn.Linear(500, num_phns))

        self.model_class = nn.ModuleList(self.layers_class)
        #init_weights(self.model_class)

    def forward(self, x_mfccs, x_mels):
        for layer in self.model_mfccs:
            #print(x_mfccs.shape)
            x_mfccs = layer(x_mfccs)

        for layer in self.model_mels:
            #print(x_mels.shape)
            x_mels = layer(x_mels)

        x = torch.cat((x_mfccs, x_mels), axis=1)
        #print(x.shape)

        for layer in self.model_class:
            x = layer(x)

        return x
