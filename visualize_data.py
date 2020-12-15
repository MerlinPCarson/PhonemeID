import os
import sys
import time
import pickle
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import sounddevice as sd
from math import ceil
from confusion_matrix_pretty_print import plot_confusion_matrix_from_data
from sklearn.metrics import confusion_matrix
import pandas as pd


# From Phoneme Boundary Detection Using Learnable Segmental Features, Felix Kreuk et al.
def mfcc_dist(mfcc):
    """mfcc_dist
    calc 4-dimensional dist features like in HTK
    
    :param mfcc:
    """
    d = []
    for i in range(2, 9, 2):
        pad = int(i/2)
        d_i = np.concatenate([np.zeros(pad), ((mfcc[:, i:] - mfcc[:, :-i]) ** 2).sum(0) ** 0.5, np.zeros(pad)], axis=0)
        d.append(d_i)
    return np.stack(d)

def extract_features(samples, sr):

    mfccs = librosa.feature.mfcc(samples,
                                     sr=sr,
                                     n_fft=60,
                                     hop_length=160,
                                     n_mels=22,
                                     n_mfcc=13)

    #mfccs = mfccs[1:]    
    dists = mfcc_dist(mfccs)
    deltas  = librosa.feature.delta(mfccs, order=1)
    deltas2 = librosa.feature.delta(mfccs, order=2)
    return mfccs, dists, deltas, deltas2

def plot_features(features, sr):
    titles = ['Distance', '1st Delta', '2nd Delta']
    for i, (title, feature) in enumerate(zip(titles, features[1:]), start=1):
        plt.subplot(3,1,i)
        librosa.display.specshow(feature, x_axis='time', sr=sr, hop_length=160)
        #plt.imshow(feature, interpolation='nearest', aspect='auto')
        #plt.imshow(feature)
        #plt.xlabel('Time')
        #plt.ylabel('MFCC Index')
        plt.ylabel(title)
        plt.colorbar()

    #plt.tight_layout()
    plt.show()

def plot_data_phn(samples, phns, sr, max_len=3200):

    start = int(phns[4][0])
    end = int(phns[4][1])
    length = end - start

    # centered features in zero padded array of max length         
    diff = max_len - length
    pad = diff / 2
    seg = np.pad(samples[start:end], (int(pad), ceil(pad)), 'constant', constant_values=(0,0)) 

    plt.subplot(2,1,1)
    plt.plot(seg)
    plt.annotate(phns[4][2], xy=(0.46, 0.05), fontsize=12, color='red', xycoords='axes fraction')
    plt.xlabel('Time')
    plt.ylabel('Intensity')

    features = extract_features(seg, sr)
    plt.subplot(2,1,2)
    librosa.display.specshow(features[0], x_axis='time', sr=sr, hop_length=160)
    plt.ylabel('MFCCs')

    plt.tight_layout()
    plt.show()

    return seg


def plot_data(samples, phns, sr):
    plt.plot(samples)

    for phn in phns:
        if int(phn[1]) < sr:
            plt.axvline(int(phn[0]), linestyle='dashed', color='green')
            plt.axvline(int(phn[1]), linestyle='dashed', color='green')
            phn_local = (int(phn[0]) + (int(phn[1])-int(phn[0]))/2)
            #print(phn_local, phn[2])
            #plt.annotate(phn[2], xy=(phn_local/sr, 0.05), fontsize=12, color='red', xycoords='axes fraction')

    phn_xs = [2620.5, 2955.5, 4195.5, 6453.0, 8395.0, 10314.5, 12303.0, 14040.0]
    phn_labels = [phn[2] for phn in phns[1:] if int(phn[1]) < sr]

    for x, phn in zip(phn_xs, phn_labels):
        plt.annotate(phn, xy=(x/sr, 0.05), fontsize=12, color='red', xycoords='axes fraction')

    words_xs = [2620, 6500, 10000]
    words_labels = ['The', 'full', 'moon']

    for x, word in zip(words_xs, words_labels):
        plt.annotate(word, xy=(x/sr, 0.95), fontsize=16, color='black', xycoords='axes fraction')


    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.tight_layout()
    plt.show()

def extract_phonemes(phn_file):
    with open(phn_file, 'r') as f:
        lines = f.read().splitlines()
        phonemes = [line.split(' ') for line in lines]

    return phonemes 

def plot_loss(model_file, lr_reduce, lr_rates):
    data = pickle.load(open(model_file, 'rb'))

    epochs = np.arange(1,len(data['loss'])+1)

    plt.plot(epochs, data['loss'], label='training')
    plt.plot(epochs, data['val_loss'], label='validation')
    plt.ylabel('loss')
    plt.xlabel('epoch')

    for x, lr in zip(lr_reduce, lr_rates):
        plt.axvline(x, linestyle='dashed', color='black', label=f'lr={lr}')

    best_epoch = epochs[np.argmin(np.array(data['val_loss']))]
    plt.axvline(best_epoch, linestyle='-.', color='green', label=f'best epoch={best_epoch}')

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def main():
    start = time.time()

    #model_dir = 'models_LR_tst_63.64'
    model_dir = 'models_PRF_64.14'

    # for confusion matrix
    #pd.set_option("display.max_rows", 45, "display.max_columns", 45)
    pd.set_option('display.expand_frame_repr', False)
    results = pickle.load(open(os.path.join(model_dir, 'test_preds.npy'), 'rb'))
    phn_dict = pickle.load(open('dict.npy','rb'))
    labels = [x for x in range(45)]
    cf = confusion_matrix(results['targets'], results['preds'], labels=labels)
    cf_df = pd.DataFrame(cf, index=phn_dict, columns=phn_dict)
    print(cf_df)
    #plot_confusion_matrix_from_data(results['targets'], results['preds'], fz=8, columns=labels)

    # for training curves
    lr_reduce = [1, 30, 39]
    lr_rates = [0.001, 0.0001, 0.00001]
    model_training_file = os.path.join(model_dir, 'final_model.npy')
    plot_loss(model_training_file, lr_reduce, lr_rates)


    #wav_file = sys.argv[1]
    #wav_file = '../timit/data/TRAIN/DR1/FCJF0/SA1.WAV.wav'
    wav_file = '/home/mpc6/projects/ASR/timit/data/TRAIN/DR1/MWAR0/SX325.WAV.wav'
    samples, sr = sf.read(wav_file)
    #sd.play(samples, sr)

    phns = extract_phonemes(wav_file.replace('.WAV.wav', '.PHN'))
    plot_data(samples[:sr], phns, sr)
    seg = plot_data_phn(samples[:sr], phns, sr)

    #features = extract_features(samples[int(phns[6][0]):int(phns[6][1])], sr)
    features = extract_features(seg, sr)
    print(features[2].mean(), features[2].std())
    print(features[3].mean(), features[3].std())
    plot_features(features, sr)
 
    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
