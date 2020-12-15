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

# Fully Convolutional Neural Network
class FCNN(nn.Module):
    def __init__(self, args, padding=0):
        super(FCNN, self).__init__()

        # check for padding to maintain input shape
        if args.padding_same is True:
            print('adding padding')
            padding = (args.filter_size - args.stride) // 2

        # create module list
        self.layers = []

        # create CNN blocks
        self.layers.append(nn.Conv2d(in_channels=args.num_channels, out_channels=args.num_filters, 
                                     kernel_size=args.kernel_size, bias=True, padding=padding))
        self.layers.append(nn.BatchNorm2d(args.num_filters))
        self.layers.append(nn.PReLU(args.num_filters))
        self.layers.append(nn.Dropout(0.4))
        for i in range(1, args.num_cnn_blocks):
            self.layers.append(nn.Conv2d(in_channels=args.num_filters//i, out_channels=args.num_filters//(i+1), 
                                         kernel_size=args.kernel_size, bias=True, padding=padding))
            self.layers.append(nn.BatchNorm2d(args.num_filters//(i+1)))
            self.layers.append(nn.PReLU(args.num_filters//(i+1)))
            self.layers.append(nn.Dropout(0.4))

        self.model = nn.ModuleList(self.layers)
        init_weights(self.model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

# Fully connected network
class FCN(nn.Module):
    def __init__(self, args, num_neurons=500):
        super(FCN, self).__init__()

        # create module list
        self.layers = []
        
        self.layers.append(nn.Linear(args.num_features, num_neurons))
        self.layers.append(nn.BatchNorm1d(num_neurons))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Dropout(args.dropout))
        self.layers.append(nn.Linear(num_neurons, args.num_phonemes))

        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class PhonemeID_CNN(nn.Module):
    def __init__(self, args):
        super(PhonemeID_CNN, self).__init__()

        self.feature_model = FCNN(args)
        self.pred_model = FCN(args, num_neurons=args.num_features//3)

    def forward(self, x):

        # base features/model
        features = self.feature_model(x).flatten(start_dim=1)

        # prediction model
        out = self.pred_model(features)
        return out
