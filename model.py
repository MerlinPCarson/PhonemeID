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
        self.layers.append(nn.Dropout(args.dropout))
        for _ in range(args.num_cnn_blocks-1):
            self.layers.append(nn.Conv2d(in_channels=args.num_filters, out_channels=args.num_filters, 
                                         kernel_size=args.kernel_size, bias=True, padding=padding))
            self.layers.append(nn.BatchNorm2d(args.num_filters))
            self.layers.append(nn.PReLU(args.num_filters))
            self.layers.append(nn.Dropout(args.dropout))

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
        
        self.layers.append(nn.Linear(args.num_features, 500))
        self.layers.append(nn.BatchNorm1d(500))
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Dropout(args.dropout))
        self.layers.append(nn.Linear(500, args.num_phonemes))

        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class MultiHeadCNN(nn.Module):
    def __init__(self, args):
        super(MultiHeadCNN, self).__init__()

        self.mfcc_model = FCNN(args)
        self.dist_model = FCNN(args, padding=(args.kernel_size - args.stride)//2)
        self.delta_model = FCNN(args)
        self.delta2_model = FCNN(args)
        self.pred_model = FCN(args)

    def forward(self, mfccs, dists=None, deltas=None, deltas2=None):
        dists_out = deltas_out = deltas2_out = None

        # base features/model
        mfccs_out = self.mfcc_model(mfccs).flatten(start_dim=1)

        # additional (optional) features/models
        if dists is not None:
            dists_out = self.dist_model(dists)
            out = torch.cat((mfccs_out, dists_out.flatten(start_dim=1)), axis=1)
        if deltas is not None:
            deltas_out = self.delta_model(deltas)
            out = torch.cat((out, deltas_out.flatten(start_dim=1)), axis=1)
        if deltas2 is not None:
            deltas2_out = self.delta2_model(deltas)
            out = torch.cat((out, deltas2_out.flatten(start_dim=1)), axis=1)

        out = self.pred_model(out)
        return out
