import sys
import time
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import sounddevice as sd


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
        
    dists = mfcc_dist(mfccs)
    deltas  = librosa.feature.delta(mfccs, order=1)
    deltas2 = librosa.feature.delta(mfccs, order=2)
    return mfccs, dists, deltas, deltas2

def plot_features(features):
    titles = ['MFCC', 'Distance', '1st Derivative', '2nd Derivative']
    for title, feature in zip(titles, features):
        plt.figure()
        plt.title(title)
        librosa.display.specshow(feature, x_axis='time')
        #plt.imshow(feature, interpolation='nearest', aspect='auto')
        #plt.imshow(feature)
        plt.xlabel('Time')
        plt.ylabel('Frequency')

    #plt.tight_layout()
    #plt.colorbar()
    plt.show()

def plot_data(samples, phns, sr):
    plt.plot(samples)

    for phn in phns:
        if int(phn[1]) < sr:
            plt.axvline(int(phn[0]), linestyle='dashed', color='green')
            plt.axvline(int(phn[1]), linestyle='dashed', color='green')
            phn_local = (int(phn[0]) + (int(phn[1])-int(phn[0]))/2)
            #print(phn_local, phn[2])
            #plt.annotate(phn[2], xy=(phn_local/sr, 0.05), fontsize=12, color='red', xycoords='axes fraction')

    phn_xs = [1500.0, 2620.5, 2955.5, 4195.5, 6453.0, 8395.0, 10314.5, 12303.0, 14040.0]
    phn_labels = [phn[2] for phn in phns if int(phn[1]) < sr]

    for x, phn in zip(phn_xs, phn_labels):
        plt.annotate(phn, xy=(x/sr, 0.05), fontsize=12, color='red', xycoords='axes fraction')

    words_xs = [2620, 6500, 10000]
    words_labels = ['The', 'full', 'moon']

    for x, word in zip(words_xs, words_labels):
        plt.annotate(word, xy=(x/sr, 0.95), fontsize=16, color='black', xycoords='axes fraction')


    plt.tight_layout()
    plt.show()

def extract_phonemes(phn_file):
    with open(phn_file, 'r') as f:
        lines = f.read().splitlines()
        phonemes = [line.split(' ') for line in lines]

    return phonemes 

def main():
    start = time.time()

    #wav_file = sys.argv[1]
    #wav_file = '../timit/data/TRAIN/DR1/FCJF0/SA1.WAV.wav'
    wav_file = '/home/mpc6/projects/ASR/timit/data/TRAIN/DR1/MWAR0/SX325.WAV.wav'
    samples, sr = sf.read(wav_file)
    sd.play(samples, sr)

    phns = extract_phonemes(wav_file.replace('.WAV.wav', '.PHN'))
    plot_data(samples[:sr], phns, sr)

    #features = extract_features(samples[int(phns[6][0]):int(phns[6][1])], sr)
    features = extract_features(samples[:sr], sr)
    plot_features(features)
 

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
