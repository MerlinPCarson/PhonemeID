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
    def __init__(self, num_channels, num_cnn_blocks=3, kernel_size=3, stride=1, num_filters=16, padding=0):
        super(FCNN, self).__init__()

        # create module list
        self.layers = []

        # create CNN blocks
        for i in range(num_cnn_blocks):
            self.layers.append(nn.Conv2d(in_channels=1, out_channels=num_filters, 
                                        kernel_size=kernel_size, bias=True, padding=padding))
            self.layers.append(nn.BatchNorm2d(num_filters))
            self.layers.append(nn.PReLU())
            self.layers.append(nn.Dropout(0.25))

        self.model = nn.ModuleList(self.layers)
        init_weights(self.model)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

# Fully connected network
class FCN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(FCN, self).__init__()

        # create module list
        self.layers_class = []
        self.layers_class.append(nn.Linear(num_inputs, 500))
        self.layers_mels.append(nn.BatchNorm1d(500))
        self.layers_class.append(nn.PReLU())
        self.layers_mels.append(nn.Dropout(0.25))
        self.layers_class.append(nn.Linear(500, num_outputs))

        self.model_class = nn.ModuleList(self.layers_class)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x

class MultiHeadCNN(nn.Module):
    def __init__(self, num_features, num_phns, num_channels=1, num_cnn_blocks=3, kernel_size=3, stride=1, num_filters=16):
        super(MultiHeadCNN, self).__init__()

        self.mfcc_model = FCNN(num_channels, num_cnn_blocks, kernel_size, stride, num_filters)
        self.dist_model = FCNN(num_channels, num_cnn_blocks, kernel_size, stride, num_filters)
        self.delta_model = FCNN(num_channels, num_cnn_blocks, kernel_size, stride, num_filters)
        self.delta2_model = FCNN(num_channels, num_cnn_blocks, kernel_size, stride, num_filters)
        self.pred_model = FCN(num_features, num_phns)

    def forward(self, mfccs, dists=None, deltas=None, deltas2=None):
        dists_out = deltas_out = deltas2_out = None

        # base features/model
        mfccs_out = nn.Flatten(self.mfcc_model(mfccs))

        # additional (optional) features/models
        if dists is not None:
            dists_out = self.dist_model(dists)
            out = torch.cat((mfccs_out, nn.Flatten(dists_out)), axis=1)
        if deltas is not None:
            deltas_out = self.delta_model(deltas)
            out = torch.cat((out, nn.Flatten(deltas_out)), axis=1)
        if deltas2 is not None:
            deltas2_out = self.delta2_model(deltas)
            out = torch.cat((out, nn.Flatten(deltas2_out)), axis=1)

        out = self.pred_model()
        return out