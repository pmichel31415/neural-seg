from __future__ import print_function, division

import numpy as np
import os
import argparse

from scipy.io.wavfile import read as wavread
from scipy.signal import resample
from features import mfcc


import kmodels
import post_process_rnn_error
import run_rnn
import preprocess
import arguments

from keras.callbacks import EarlyStopping


def make_features(wav_dir, mfcc_dir, energy=False, n=13):

    if not os.path.exists(mfcc_dir):
        os.mkdir(mfcc_dir)

    for f in os.listdir(wav_dir):
        if f.endswith('.wav'):
            fs, w = wavread(wav_dir + '/' + f)
            m = mfcc(w, samplerate=fs, appendEnergy=energy, numcep=n)
            np.save(mfcc_dir + '/' + f[:-3] + 'npy', m)


if __name__ == '__main__':
    opt = arguments.parser.parse_args()
    tasks = set(opt.workflow.split('|'))

    model = None

    # Compute MFCC features from waveforms
    if 'mfcc' in tasks:
        make_features(
            opt.wav_dir,
            opt.mfcc_dir,
            energy=opt.mfcc_energy,
            n=opt.numcep
        )
    # Compute discrete states vector using clustering
    if 'preprocess' in tasks:
        preprocess.mfcc2states(
            opt.mfcc_dir,
            opt.states_dir,
            num_clusters=opt.num_clusters,
            method=opt.preprocess_method,
            subset_size=opt.preprocess_subset_size
        )
    # Train model
    if 'train' in tasks:
        model = run_rnn.train(
            opt.train_list,
            train_model=model,
            train_model_type=opt.train_model_type,
            embed_dim=opt.embed_dim,
            hidden_dim=opt.hidden_dim,
            optim='sgd',
            loss='mse',
            span=40,
            batch_size=128,
            stateful=opt.stateful,
            verbose=opt.verbose
        )
    # Test model
    if 'test' in tasks:
        run_rnn.test(
            opt.test_list,
            opt.out_dir,
            test_model=model,
            test_model_type=opt.test_model_type,
            embed_dim=opt.embed_dim,
            hidden_dim=opt.hidden_dim,
            optim=opt.optim,
            error=opt.mse,
            verbose=opt.verbose
        )
    # Post processing
    if 'postprocess' in tasks:
        post_process_rnn_error.run(
            opt.out_dir,
            opt.out_dir+'_syldet',
            method=opt.postprocess_method,
            time_dir=opt.time_dir,
            rate=opt.rate,
            ker_len=opt.ker_len,
            clip=opt.clip,
            threshold=opt.threshold
        )


