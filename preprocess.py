from __future__ import print_function, division

import numpy as np
import argparse
import os
from sklearn.cluster import KMeans

from sklearn.base import BaseEstimator, ClusterMixin


parser = argparse.ArgumentParser(
    description='Features categorization')


parser.add_argument('-i', '--input', action='store', dest='mfcc_list',
                    required=True,
                    type=str, help='File listing features files locations')
parser.add_argument('-Q', '--cluster_num', action='store', dest='num_clusters',
                    default=39,
                    type=int, help='Embedding dimension')
parser.add_argument('-de', '--dim_embed', action='store', dest='embed_dim',
                    default=39,
                    type=int, help='Embedding dimension')
parser.add_argument('-m', '--method', action='store', dest='method',
                    default='random', choices=['random', 'kmeans', 'partial_kmeans'],
                    type=str, help='File listing features files locations')
parser.add_argument('-o', '--output', action='store', dest='out_dir',
                    required=True,
                    type=str, help='Output directory')


class SemiKMeansCluster(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters=7, size_subset=10000):
        self.n_clusters = n_clusters
        self.size_subset = size_subset

    def fit_predict(self, X, y=None):
        subset = np.random.choice(len(X), size=self.size_subset)
        sub_model = KMeans(self.n_clusters)
        sub_model.fit(X[subset])
        labels = sub_model.predict(X)
        return labels

    def get_params(self):
        return self.n_clusters

    def set_params(self, n_clusters=7):
        self.n_clusters = n_clusters


class RandomCluster(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters=7):
        self.n_clusters = n_clusters

    def fit_predict(self, X, y=None):
        c_indexes = np.random.choice(len(X), size=self.n_clusters)
        centroids = [X[i] for i in c_indexes]
        labels = np.argmin(np.array(
            [np.linalg.norm(X-c.reshape(1, -1), ord=2, axis=-1) for c in centroids]), axis=0)
        return labels

    def get_params(self):
        return self.n_clusters

    def set_params(self, n_clusters=7):
        self.n_clusters = n_clusters


def int_2_onehot(i, MAX):
    ret = np.zeros(MAX)
    ret[i] = 1
    return ret


def load_features(filename):
    if filename.endswith('.npy'):
        feats = np.load(filename)
    else:
        feats = np.loadtxt(filename)
    feats = feats-np.mean(feats, axis=0)
    feats = feats/np.std(feats, axis=0)
    return feats


def mfcc2states(mfcc_dir, out_dir, num_clusters=8, method='random', subset_size=10000):

    input_files = []

    input_files = [
        mfcc_dir+'/'+f for f in os.listdir(mfcc_dir) if f.endswith('.npy')]

    X = []
    file2feats = dict()
    de = None
    for f in input_files:
        current_inputs = load_features(f)
        # get embedding dimension
        if de is None:
            de = current_inputs.shape[1]
        # Check for embedding dimension coherence
        if de != current_inputs.shape[1]:
            print('Wrong embedding dimension',
                  current_inputs.shape[1], ', expected', de)
            continue

        file2feats[f.split('/')[-1]] = (len(X), len(current_inputs))
        X += current_inputs.tolist()
    X = np.array(X)
    print(X.shape)
    if method == 'random':
        model = RandomCluster(n_clusters=num_clusters)
    elif method == 'kmeans':
        model = KMeans(n_clusters=num_clusters)
    elif method == 'partial_kmeans':
        model = SemiKMeansCluster(
            n_clusters=num_clusters, size_subset=subset_size)

    labels = model.fit_predict(X)
    print(labels.shape)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for f, (s, n) in file2feats.iteritems():
        current_outputs = np.array(
            [int_2_onehot(labels[i], num_clusters) for i in range(s, s+n)])
        out_dir = out_dir if out_dir.endswith('/') else out_dir+'/'
        out_name = out_dir+f
        np.save(out_name, current_outputs)


if __name__ == '__main__':
    opt = parser.parse_args()
    mfcc2states(opt.mfcc_list, opt.out_dir,
                num_clusters=opt.num_clusters, method=opt.method, subset_size=10000)
