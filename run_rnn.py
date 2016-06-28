from __future__ import print_function, division

import numpy as np
import os
import argparse

from scipy.io.wavfile import read as wavread
from scipy.signal import resample

import kmodels

from keras.callbacks import EarlyStopping


def load_features(filename):
    if filename.endswith('.npy'):
        feats = np.load(filename)
    else:
        feats = np.loadtxt(filename)
    # feats=feats-np.mean(feats,axis=0)
    # feats=feats/np.std(feats,axis=0)
    return feats


def load_wav(filename):
    if filename.endswith('.wav'):
        fs, x = wavread(filename)
        if fs != 8000:
            x = resample(x, int(16000/fs*len(x)))
        return x
    return np.array([])


def save_model_weights(filename, model):
    np.savez(filename, list(model.get_weights()))


def load_model_weights(filename):
    weights = np.load(filename)
    weights = [weights[i] for i in range(weights.shape[0])]
    return weights


def prep_train(data, chunk_size=6000):
    train_x = []
    train_y = []

    for i in range(0, len(data)-chunk_size):
        end = min(len(data)-1, i+chunk_size)
        train_x.append(data[i:end])
        train_y.append(data[i+1:end+1])

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y


class StatefulDataGenerator:

    def __init__(self, data, span, batch_size, model=None):
        self.files = data
        np.random.shuffle(self.files)
        self.file = np.random.randint(0, len(data), size=(batch_size))
        self.idxs = np.zeros(batch_size)
        self.n = len(data)
        self.batch_size = batch_size
        self.span = span
        self.model = model

    def __iter__(self):
        return self

    def next(self):
        for i in range(self.batch_size):
            if self.idxs[i]+self.span+1 >= len(self.files[self.file[i]]):
                self.idxs[i] = 0
                self.file[i] = (self.file[i]+1) % self.n
                if self.model is not None:
                    self.model.reset_states()

        x = np.array([
            self.files[self.file[i]][self.idxs[i]:self.idxs[i]+self.span]
            for i in range(self.batch_size)
        ])
        y = np.array([
            self.files[self.file[i]][self.idxs[i]+self.span+1]
            for i in range(self.batch_size)
        ])

        self.idxs[i] += self.span
        return (x, y)

    def get(self):
        return self.next()

    def size():
        return sum(len(f) for f in self.files)

class DataGenerator:

    def __init__(self, data, span, batch_size, model=None):
        self.data = data

        self.instances = []
        for i, f in enumerate(data):
            for j in range(len(f)-span-1):
                if sum(data[i][j+span-1] * data[i][j+span])<0.0001:# or np.random.uniform() > 0.9:
                    self.instances.append([i, j, j+span])

        self.instances = np.array(self.instances)
        self.n = len(self.instances)
        self.idxs = np.array([
            np.random.permutation(self.n)
            for i in range(batch_size)
        ])
        self.span = span
        self.batch_size = batch_size
        self.i = 0

    def __iter__(self):
        return self

    def next(self):
        x = np.array([
            self.data[f][start:end]
            for [f, start, end] in self.instances[self.idxs[:, self.i]]
        ])
        y = np.array([
            self.data[f][end]
            for [f, start, end] in self.instances[self.idxs[:, self.i]]
        ])
        self.i = (self.i+1) % self.n
        return (x, y)

    def get(self):
        return self.next()

    def size(self):
        return self.n


def train(
    train_list,
    train_model=None,
    train_model_type='simple_rnn',
    embed_dim=10,
    hidden_dim=20,
    optim='sgd',
    loss='mse',
    stateful=False,
    span=40,
    batch_size=128,
    verbose=False
):
    '''
    Runs the whole training/testing pipeline
    '''
    if verbose:
        print('----------------------------------------')
        print('            Creating model              ')
        print(' - input dim :', embed_dim)
        print(' - hidden dim :', hidden_dim)
        print(' - output dim :', embed_dim)
        print('----------------------------------------')

    weights = None
    if train_model is not None:
        if verbose:
            print(' Initializing model with custom weights ')
            print(' - source :', os.path.basename(model))
        weights = load_model_weights(train_model)
    else:
        if verbose:
            print(' Initializing model with random weights')

    if hasattr(kmodels, 'build_' + train_model_type):
        build_model = getattr(kmodels, 'build_' + train_model_type)
    else:
        print('Unknown model type :/ you\'ll have to code it yourself')
        exit()
    model = build_model(
        embed_dim, hidden_dim, embed_dim, span, weights, batch_size)
    if verbose:
        print(' Compiling model with ')
        print(' - optimizer : ' + optim)
        print(' - loss function : ' + loss)
    model.compile(
        optimizer=optim,
        loss=loss  # kmodels.last_mse
    )

    if verbose:
        print('----------------------------------------')
        print('       Running training on corpus       ')
        print('----------------------------------------')

    train_files = []

    with open(train_list, 'r') as f:
        train_files = [l for l in f.read().split('\n') if l.endswith('.npy')]
    train_data = []
    valid_data = []
    raw_data = []
    for tf in train_files:
        if verbose:
            print('         Loading training file          ')
            print(' - file :', tf)
            print('----------------------------------------')

        current_inputs = load_features(tf)
        if current_inputs.shape[1] != embed_dim:
            print('Wrong input dim :', current_inputs.shape[1],
                  ', expected', embed_dim,
                  'in file', f, ', skipping to next file.')
            continue
        raw_data.append(current_inputs)
    np.random.shuffle(raw_data)
    valid_split = int(0.1*len(train_files))
    valid_data = raw_data[:valid_split]
    train_data = raw_data[valid_split:]

    if stateful:
        datagen = StatefulDataGenerator
    else:
        datagen = DataGenerator
    batch_generator = datagen(train_data, span, batch_size, model)
    validation_generator = datagen(valid_data, span, batch_size, model)
    print(next(batch_generator)[0].shape, next(batch_generator)[1].shape)
    model.fit_generator(
        batch_generator,
        batch_generator.size(),  # Sample per epoch
        1000,  # epoch
        validation_data=validation_generator,
        nb_val_samples=10,
        callbacks=[
            EarlyStopping(
                monitor='val_loss',
                patience=0,
                verbose=1,
                mode='auto'
            )
        ],
        verbose=1
    )

    return model


def test(
    test_list,
    out_dir,
    trained_model=None,
    test_model_type='simple_rnn',
    embed_dim=10,
    hidden_dim=20,
    optim='sgd',
    error='mse',
    verbose=False
):
    if verbose:
        print('----------------------------------------')
        print('       Running testing on corpus        ')
        print('----------------------------------------')

    if hasattr(kmodels, 'build_' + test_model_type):
        build_model = getattr(kmodels, 'build_' + test_model_type)
    else:
        print('Unknown model type :/ you\'ll have to code it yourself')
        exit()
    test_model = build_model(
        embed_dim, hidden_dim, embed_dim, trained_model.get_weights())
    test_model.compile(
        optimizer=optim,
        loss=error
    )

    test_files = []

    with open(test_list, 'r') as f:
        test_files = [l for l in f.read().split('\n') if l.endswith('.npy')]

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for i, f in enumerate(test_files):
        current_inputs = load_features(f)
        if current_inputs.shape[1] != embed_dim:
            print('Wrong input dim :', current_inputs.shape[1],
                  ', expected', embed_dim,
                  'in file', f, ', skipping to next file.')
            continue
        if verbose:
            print('Running through file', f, '(', i, '/', len(test_files), ')')
        y = np.zeros((len(current_inputs), embed_dim))
        loss = np.zeros(len(current_inputs))
        test_x = np.array(current_inputs[:-1])
        test_x = test_x.reshape(1, len(test_x), -1)
        test_y = current_inputs[1:]

        y[1:] = test_model.predict_on_batch(test_x)

        if error == 'mse':
            loss = np.mean(np.square(y-current_inputs), axis=-1)
        elif error == 'categorical_crossentropy':
            loss = -(current_inputs*np.log(y+0.0000001)).sum(axis=-1)
        elif error == 'cosine_proximity':
            dotprod = np.sum(current_inputs*y, axis=-1)
            norm_y = np.linalg.norm(y[1:], ord=2, axis=-1)
            norm_gold = np.linalg.norm(current_inputs, ord=2, axis=-1)
            cosine = dotprod/(norm_y+norm_gold + 0.000001)
            loss = 1-cosine

        out_file = out_dir + '/' + os.path.basename(f)[:-4] + '_loss.npy'
        np.save(out_dir + '/' + os.path.basename(f), y)
        np.save(out_file, loss)

    if verbose:
        print('----------------------------------------')
        print('      Testing over, dumping model       ')
        print('----------------------------------------')

    save_model_weights(out_dir+"/dump_model", test_model)
