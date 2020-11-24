import os
import sys
import time
import h5py
import pickle
import argparse
import numpy as np

from build_timit import TimitDictionary, TimitDataLoader

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import SimpleCNN, SimpleCNN2D, DualCNN2D

from sklearn.model_selection import train_test_split
from tqdm import tqdm

        
def parse_args():
    parser = argparse.ArgumentParser(description='Timit dataset builder')
    parser.add_argument('--timit_path', type=str, default='../timit/data',
                         help='location of Timit Train and Test directories')
    parser.add_argument('--phoneme_dict', type=str, default='../timit/TIMITDIC.TXT',
                         help='location of phoneme dictionary')
    parser.add_argument('--out_dir', type=str, default='data', help='location to save datasets')
    parser.add_argument('--num_ffts', type=int, default=60, help='n_fft for feature extraction')
    parser.add_argument('--hop_length', type=int, default=160, help='hop_length for feature extraction')
    parser.add_argument('--num_mels', type=int, default=22, help='number of mels')
    parser.add_argument('--num_mfccs', type=int, default=13, help='number of mfccs')

    parser.add_argument('--model_dir', type=str, default='models', help='location of model files')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='number of epochs of no improvment before early stopping')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--num_filters', type=int, default=32, help='base number of filters for CNN layers')

    return parser.parse_args()

def setup_gpus():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    device_ids = [i for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, device_ids))
    return device_ids

def preds_accuracy(preds, target):
    preds = torch.nn.functional.softmax(preds, dim=1)
    _, top_class = preds.topk(k=1, dim=1)
    equals = top_class == target.view(top_class.shape)
    return torch.mean(equals.type(torch.float))

def main(args):
    start = time.time()

    # build timit dictionary from timit dictionary file
    timit_dict = TimitDictionary(args.phoneme_dict)
    print(f'Number of phonemes in dictionary: {timit_dict.nphonemes}')

    # create timit dataset object 
    timit_data = TimitDataLoader(args.timit_path, timit_dict, args.num_ffts, 
                                 args.hop_length, args.num_mels, args.num_mfccs)

    # load dataset from H5 files
    timit_data.load_from_h5(args.out_dir)
    
    # show/verify sizes of datasets
    timit_data.dataset_stats()

    # make sure output dirs exists
    os.makedirs(args.model_dir, exist_ok=True)

    # detect gpus and setup environment variables
    device_ids = setup_gpus()
    print(f'Cuda devices found: {[torch.cuda.get_device_name(i) for i in device_ids]}')

    # applying random seed for reproducability
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trainX = np.concatenate((timit_data.trainX_mfccs, timit_data.trainX_mels), axis=1)

    # split data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(trainX, timit_data.trainY, test_size = 0.1, random_state=args.seed)

    x_train_mfccs, x_train_mels = x_train[:,:args.num_mfccs:], x_train[:,args.num_mfccs:,:]
    x_val_mfccs, x_val_mels = x_val[:,:args.num_mfccs:], x_val[:,args.num_mfccs:,:]

    print(f'Training data mfccs: {x_train_mfccs.shape}, validation data mfccs: {x_val_mfccs.shape}, test data mfccs: {timit_data.testX_mfccs.shape}')
    print(f'Training data mels: {x_train_mels.shape}, validation data mels: {x_val_mels.shape}, test data mels: {timit_data.testX_mels.shape}')
    print(f'Training target: {y_train.shape}, validation target: {y_val.shape}, test target {timit_data.testY.shape}')

    # get standardization parameters from training set
    train_mean_mfccs = np.mean(x_train_mfccs)
    train_std_mfccs = np.std(x_train_mfccs)
    train_mean_mels = np.mean(x_train_mels)
    train_std_mels = np.std(x_train_mels)

    # apply standardization parameters to training and validation sets
    x_train_mfccs = (x_train_mfccs-train_mean_mfccs)/train_std_mfccs
    x_val_mfccs = (x_val_mfccs-train_mean_mfccs)/train_std_mfccs
    x_train_mels = (x_train_mels-train_mean_mels)/train_std_mels
    x_val_mels = (x_val_mels-train_mean_mels)/train_std_mels
    x_test_mfccs = (timit_data.testX_mfccs-train_mean_mfccs)/train_std_mfccs
    x_test_mels = (timit_data.testX_mels-train_mean_mels)/train_std_mels

    # input shape for each example to network, NOTE: channels first
    num_channels = x_train.shape[1]
    #print(f'Input shape to model forward will be: ({args.batch_size}, {num_channels}, {num_features})')

    # load data for training
    train_dataset = TensorDataset(torch.Tensor(x_train_mfccs).unsqueeze(1), torch.Tensor(x_train_mels).unsqueeze(1), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.Tensor(x_val_mfccs).unsqueeze(1), torch.Tensor(x_val_mels).unsqueeze(1), torch.LongTensor(y_val))
    test_dataset = TensorDataset(torch.Tensor(x_test_mfccs).unsqueeze(1), torch.Tensor(x_test_mels).unsqueeze(1), torch.LongTensor(timit_data.testY))

    print(f'Number of training examples: {len(train_dataset)}')
    print(f'Number of validation examples: {len(val_dataset)}')
    print(f'Number of test examples: {len(test_dataset)}')

    # create batched data loaders for model
    train_loader = DataLoader(dataset=train_dataset, num_workers=os.cpu_count(), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=os.cpu_count(), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, num_workers=os.cpu_count(), batch_size=args.batch_size, shuffle=True)

    # create model
    model = DualCNN2D(num_channels, timit_dict.nphonemes, num_filters=args.num_filters)

    # prepare model for data parallelism (use multiple GPUs)
    #model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    print(model)

    # setup loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()#.cuda()
    #criterion = torch.nn.NLLLoss()#.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # schedulers
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=args.patience//2)

    # data struct to track training and validation losses per epoch
    model_params = {'num_filters': args.num_filters, 
                    'train_mean_mfccs': train_mean_mfccs, 'train_std_mels': train_std_mfccs,
                    'train_mean_mels': train_mean_mels, 'train_std_mels': train_std_mels}

    # save model parameters
    history = {'model': model_params, 'loss':[], 'acc': [], 
                                      'val_loss':[], 'val_acc':[], 
                                      'num_filters': args.num_filters}
    pickle.dump(history, open(os.path.join(args.model_dir, 'model.npy'), 'wb'))

    # intializiang best values for regularization via early stopping 
    best_val_loss = 99999.0
    best_val_acc = 0.0
    epochs_since_improvement = 0

    # Main training loop
    for epoch in range(1, args.epochs+1):
        print(f'Starting epoch {epoch}/{args.epochs} with learning rate {optimizer.param_groups[0]["lr"]}')

        epoch_train_loss = 0.0
        epoch_train_acc = 0.0
        epoch_val_loss = 0.0
        epoch_val_acc = 0.0

        model.train()
        # iterate through batches of training examples
        for (mfccs, mels, phns) in tqdm(train_loader):
            model.zero_grad()

            # move batch to GPU
            #segs = Variable(segs.cuda())

            # make predictions
            preds = model(mfccs, mels)

            # running accuracy 
            epoch_train_acc += preds_accuracy(preds, phns)

            # calculate loss
            loss = criterion(preds, phns)
            epoch_train_loss += loss.item()

            # backprop
            loss.backward()
            optimizer.step()

        # start evaluation
        print(f'Validating Model')
        model.eval() 
        with torch.no_grad():
            for (mfccs, mels, phns) in tqdm(val_loader):

                # move batch to GPU
                #segs = Variable(segs.cuda())

                # make predictions
                preds = model(mfccs, mels)
                
                # running accuracy 
                epoch_val_acc += preds_accuracy(preds, phns)

                # calculate loss
                val_loss = criterion(preds, phns)
                epoch_val_loss += val_loss.item()

        # epoch summary
        epoch_train_loss /= len(train_loader) 
        epoch_train_acc /= len(train_loader)
        epoch_val_loss /= len(val_loader) 
        epoch_val_acc /= len(val_loader)

        # reduce learning rate if validation has leveled off
        scheduler.step(epoch_val_loss)

        # exponential decay of learning rate
#        scheduler.step()

        # save epoch stats
        history['loss'].append(epoch_train_loss)
        history['acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        print(f'Training loss: {epoch_train_loss}')
        print(f'Training accuracy: {epoch_train_acc}')
        print(f'Validation loss: {epoch_val_loss}')
        print(f'Eval accuracy: {epoch_val_acc}')

        # save if best model, reset patience counter
        if epoch_val_loss < best_val_loss:
            print('Saving best model')
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            best_epoch = epoch
            epochs_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pt'))
            pickle.dump(history, open(os.path.join(args.model_dir, 'best_model.npy'), 'wb'))
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement > args.patience:
            print('Initiating early stopping')
            break

    # saving final model
    print('Saving final model')
    torch.save(model.state_dict(), os.path.join(args.model_dir, 'final_model.pt'))
    pickle.dump(history, open(os.path.join(args.model_dir, 'final_model.npy'), 'wb'))

    # report best stats
    print(f'Best epoch: {best_epoch}')
    print(f' val loss: {best_val_loss}')
    print(f' val acc: {best_val_acc}')

    # test model
    model.load_state_dict(torch.load(os.path.join(args.model_dir, 'best_model.pt')))
    model.eval()
    with torch.no_grad():
        test_acc = 0.0
        print('Testing model')
        for (mfccs, mels, phns) in tqdm(test_loader):
            preds = model(mfccs, mels)
            # running average of accuracy
            preds = torch.nn.functional.softmax(preds, dim=1)
            _, top_class = preds.topk(k=1, dim=1)
            equals = top_class == phns.view(top_class.shape)
            test_acc += torch.mean(equals.type(torch.float))

    # average of running accuracy
    print(f' test acc: {test_acc/len(test_loader)}')

    print(f'Script completed in {time.time()-start:.2f} secs')
    return 0

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
