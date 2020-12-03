import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset


def preprocess_data(train_features, train_labels, test_features, test_labels, args, test_size=0.15):

    trainX = train_features['mfccs']
    if args.use_dists:
        trainX = np.concatenate((trainX, train_features['dists'] ), axis=1)
    if args.use_deltas:
        trainX = np.concatenate((trainX, train_features['deltas'] ), axis=1)
    if args.use_deltas2:
        trainX = np.concatenate((trainX, train_features['deltas'] ), axis=1)

    # split data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(trainX, train_labels, test_size = test_size, random_state=args.seed)

    # Pre-process MFCCS
    train_features = {'mfccs': x_train[:, :args.num_mfccs, :], 'dists': None, 'deltas': None, 'deltas2': None}
    val_features = {'mfccs': x_val[:, :args.num_mfccs, :], 'dists': None, 'deltas': None, 'deltas2': None}
    train_stats = {'mfccs_mean': None, 'mfccs_std': None,
                   'dists_mean': None, 'dists_std': None,
                   'deltas_mean': None, 'deltas_std': None,
                   'deltas2_mean': None, 'deltas2_std': None}

    # train set feature stats
    mfccs_mean = train_features['mfccs'].mean()
    mfccs_std = train_features['mfccs'].std()
    train_stats['mfccs_mean'] = mfccs_mean
    train_stats['mfccs_std'] = mfccs_std

    # standardize MFCCS
    train_features['mfccs'] = (train_features['mfccs'] - mfccs_mean) / mfccs_std
    val_features['mfccs'] = (val_features['mfccs'] - mfccs_mean) / mfccs_std
    test_features['mfccs'] = (test_features['mfccs'] - mfccs_mean) / mfccs_std

    # end of MFCCS, start of next feature
    feature_start = args.num_mfccs
    feature_end = args.num_mfccs

    # Pre-process MFCC Distances
    if args.use_dists:
        feature_end = feature_end + test_features['dists'].shape[1]
        train_features['dists'] = x_train[:,feature_start:feature_end,:]
        val_features['dists'] = x_val[:,feature_start:feature_end,:]
        feature_start = feature_end

        # train set feature stats
        dists_mean = train_features['dists'].mean()
        dists_std = train_features['dists'].std()
        train_stats['dists_mean'] = dists_mean
        train_stats['dists_std'] = dists_std

        # standardize MFCC Distances
        train_features['dists'] = (train_features['dists'] - dists_mean) / dists_std
        val_features['dists'] = (val_features['dists'] - dists_mean) / dists_std
        test_features['dists'] = (test_features['dists'] - dists_mean) / dists_std


    # Pre-process MFCC 1st order deltas 
    if args.use_deltas:
        feature_end = feature_start + test_features['deltas'].shape[1]
        train_features['deltas'] = x_train[:,feature_start:feature_end,:]
        val_features['deltas'] = x_val[:,feature_start:feature_end,:]
        feature_start = feature_end

        # train set feature stats
        deltas_mean = train_features['deltas'].mean()
        deltas_std = train_features['deltas'].std()
        train_stats['deltas_mean'] = deltas_mean
        train_stats['deltas_std'] = deltas_std

        # standardize MFCC Distances
        train_features['deltas'] = (train_features['deltas'] - deltas_mean) / deltas_std
        val_features['deltas'] = (val_features['deltas'] - deltas_mean) / deltas_std
        test_features['deltas'] = (test_features['deltas'] - deltas_mean) / deltas_std

    # Pre-process MFCC 2nd order deltas 
    if args.use_deltas2:
        feature_end = feature_start + test_features['deltas2'].shape[1]
        train_features['deltas2'] = x_train[:,feature_start:feature_end,:]
        val_features['deltas2'] = x_val[:,feature_start:feature_end,:]
        feature_start = feature_end

        # train set feature stats
        deltas2_mean = train_features['deltas2'].mean()
        deltas2_std = train_features['deltas2'].std()
        train_stats['deltas2_mean'] = deltas2_mean
        train_stats['deltas2_std'] = deltas2_std

        # standardize MFCC Distances
        train_features['deltas2'] = (train_features['deltas2'] - deltas2_mean) / deltas2_std
        val_features['deltas2'] = (val_features['deltas2'] - deltas2_mean) / deltas2_std
        test_features['deltas2'] = (test_features['deltas2'] - deltas2_mean) / deltas2_std

    # load data for training
    train_dataset = TensorDataset(torch.Tensor(train_features['mfccs']).unsqueeze(1),
                                  torch.Tensor(train_features['dists']).unsqueeze(1),
                                  torch.Tensor(train_features['deltas']).unsqueeze(1),
                                  torch.Tensor(train_features['deltas2']).unsqueeze(1),
                                  torch.LongTensor(y_train))

    val_dataset = TensorDataset(torch.Tensor(val_features['mfccs']).unsqueeze(1),
                                  torch.Tensor(val_features['dists']).unsqueeze(1),
                                  torch.Tensor(val_features['deltas']).unsqueeze(1),
                                  torch.Tensor(val_features['deltas2']).unsqueeze(1),
                                  torch.LongTensor(y_val))

    test_dataset = TensorDataset(torch.Tensor(test_features['mfccs']).unsqueeze(1),
                                  torch.Tensor(test_features['dists']).unsqueeze(1),
                                  torch.Tensor(test_features['deltas']).unsqueeze(1),
                                  torch.Tensor(test_features['deltas2']).unsqueeze(1),
                                  torch.LongTensor(test_labels))

    return train_dataset, val_dataset, test_dataset, train_stats
