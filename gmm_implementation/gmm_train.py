from mfcc_implementation import mfcc_dataset_parameters
from sklearn import mixture


def read_mfcc(data_path_root):
    dataset = mfcc_dataset_parameters.read_wav(data_path_root)
    print(len(dataset))

    mfcc_features_pool = mfcc_dataset_parameters.calculate_mfcc(dataset)

    return mfcc_features_pool


mfcc_features_pool = read_mfcc('/home/erika/Documents/Licenta/datasets/timit_full/TIMIT/data/lisa/data/timit/raw/TIMIT'
                               '/DATA/TRAIN/Keywords')
print(len(mfcc_features_pool))


# def train_gmm(data_path_root, nr_components = 50):
#     gmm = mixture.GaussianMixture(n_components=nr_components, covariance_type='full')
#     X = read_mfcc(data_path_root)
#     gmm.fit(X)
#
#
# train_gmm('/home/erika/Documents/Licenta/datasets/data/lisa/data/timit/raw/TIMIT/TRAIN', nr_components=5)

