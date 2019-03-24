from mfcc_implementation import mfcc_dataset_parameters
from sklearn import mixture


def read_mfcc(data_path_root):
    dataset = mfcc_dataset_parameters.read_wav(data_path_root)
    mfcc_features_pool = mfcc_dataset_parameters.calculate_mfcc(dataset)

    return mfcc_features_pool

#
# mfcc_features_pool = read_mfcc('/home/erika/Documents/Licenta/datasets/data/lisa/data/timit/raw/TIMIT/TRAIN')
# print(len(mfcc_features_pool))


def train_gmm(data_path_root):
    gmm = mixture.GaussianMixture(n_components=50, covariance_type='full')
    X = read_mfcc(data_path_root)
    gmm.fit(X)

    
train_gmm('/home/erika/Documents/Licenta/datasets/data/lisa/data/timit/raw/TIMIT/TRAIN')

