import os
from scipy.io import wavfile
import numpy as np
from mfcc_implementation import mfcc_base


def read_wav(data_path_root):
    dataset = []
    for dir_name, subdir_list,file_list in os.walk(data_path_root):
        for file_name in file_list:
            full_file_name = os.path.join(dir_name, file_name)

            if full_file_name.endswith(".wav"):
                try:
                    (fs,data) = wavfile.read(full_file_name)
                    dataset.append([fs,data])
                except ValueError as e:
                    print("Data is not in the desired .wav format")

    dataset = np.asarray(dataset)
    return dataset



def calculate_mfcc(dataset):

    mfcc_set = []

    for data_sample in dataset:
        [fs, signal] = data_sample
        signal_mfcc = mfcc_base.mfcc(signal, fs, 0.025, 0.01)
        mfcc_set.extend(signal_mfcc)

    mfcc_set = np.asarray(mfcc_set)
    return mfcc_set

# dataset = read_wav('/home/erika/Documents/Licenta/datasets/data/lisa/data/timit/raw/TIMIT/TRAIN')
# mfcc_pool = calculate_mfcc(dataset)
#
# print("Number of feature vectors is %d while the dimension of each is %d\n"%(len(mfcc_pool), mfcc_pool.shape[1]))
