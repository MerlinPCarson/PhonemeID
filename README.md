# PhonemeID
### by Merlin Carson
Phoneme classification project using Deep Learning

# Requirements
* Pytorch (1.7.0)
* Sklearn (0.22.1)
* h5py (2.10.0)
* Numpy (1.18.1)
* SoundFile (0.10.3)
* Librosa (0.8.0)
* tqdm (4.51.0)

\* note: The above package version were used for development and testing, but exact versions shouldn't be required.

# Download Timit dataset:
wget https://data.deepai.org/timit.zip 

# Scripts

- build_timit.py

    Takes location of TIMIT train and test directories as argument and creates train and test h5py files used for the training script.
  
 - train.py
 
    Takes location of train and test h5py files and trains/tests multi-headed CNN for phoneme classification.
  
  - model.py
  
    Multi-Headed CNN class and helper functions.
  
  - prep_data.py
  
    Preprocessing steps for preparing train and test data for training PyTorch model.
  
  - visualize_data.py
  
    Plots examples of raw TIMIT data along with examples of the features used for training (MFCCs, MFCC Distances, MFCC Delta, MFCC Delta Delta).
  
## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mpc6/PhonemeID/blob/master/LICENSE.txt)
This work is released under MIT license.
