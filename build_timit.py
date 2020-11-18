import re
import sys
import time
import argparse
import numpy as np
from tqdm import tqdm

class TimitDictionary():
    def __init__(self, dict_file):
        self.dict_file = dict_file
        self.parse_timit_dict()

    def parse_timit_dict(self):
        phonemes = []
        # load phonemes for each word in dictionary 
        with open(self.dict_file, 'r') as f:
            for line in f.readlines():
                if line[0].isalpha():
                    line_phonemes = line.split('/')[1]
                    line_phonemes = re.sub(r'[0-9]+', '', line_phonemes).split(' ')
                    
                    phonemes.extend(line_phonemes)

        # remove duplicates
        phonemes = set(phonemes)
        self.nphonemes = len(phonemes)

        self.idx_phonemes = {key: value for key, value in enumerate(phonemes)}
        self.phonemes_idx = {value: key for key, value in self.idx_phonemes.items()}


def main(args):
    start = time.time()

    timit_dict = TimitDictionary(args.phoneme_dict)
    print(f'Number of phonemes in dictionary: {timit_dict.nphonemes}')

    print(f'Script completed in {time.time()-start:.2f} secs')

    return 0

def parse_args():
    parser = argparse.ArgumentParser(description='Timit dataset builder')
    parser.add_argument('--timit_datasets', type=str, default='../timit/data',
                         help='location of Timit Train and Test directories')
    parser.add_argument('--phoneme_dict', type=str, default='../timit/TIMITDIC.TXT',
                         help='location of phoneme dictionary')
    parser.add_argument('--n_mfcc', type=int, default=13, help='number of mfccs')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    sys.exit(main(args))
