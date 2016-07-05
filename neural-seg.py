from __future__ import print_function, division

import numpy as np
import os
import argparse
import json

from scipy.io.wavfile import read as wavread
from scipy.signal import resample
from features import mfcc

import post_process_rnn_error
import run_rnn
import preprocess
import arguments
from seg_eval import run_eval


class Options:

    def __init__(self, **entries):
        self.__dict__.update(entries)


def make_features(wav_dir, mfcc_dir, energy=False, n=13):

    if not os.path.exists(mfcc_dir):
        os.mkdir(mfcc_dir)

    for f in os.listdir(wav_dir):
        if f.endswith('.wav'):
            fs, w = wavread(wav_dir + '/' + f)
            m = mfcc(w, samplerate=fs, appendEnergy=energy, numcep=n)
            np.save(mfcc_dir + '/' + f[:-3] + 'npy', m)


def make_list(folder, out_file):
    lst = [os.path.abspath(folder + '/' + f) for f in os.listdir(folder)]
    np.savetxt(out_file, lst, fmt='%s')


def dir2lists(opt):
    if not os.path.exists('splits'):
        os.mkdir('splits')
    if os.path.isdir(opt.train_list):
        make_list(opt.train_list, 'splits/'+opt.name+'_train.lst')
        opt.train_list = os.path.abspath('splits/'+opt.name+'_train.lst')
    if os.path.isdir(opt.test_list):
        make_list(opt.test_list, 'splits/'+opt.name+'_test.lst')
        opt.test_list = os.path.abspath('splits/'+opt.name+'_test.lst')


def summarize(opt):
    tasks = set(opt.workflow.split('|'))

    summary = 'Neural segmentation pipeline\n'
    summary += 'Experiment : '+opt.name+'\n'
    summary += 'Tasks executed : '+', '.join(opt.workflow.split('|'))+'\n'

    if 'mfcc' in tasks:
        summary += 'Made mfcc with parameters :\n'
        summary += 'Wav dir : ' + opt.wav_dir + '\n'
        summary += 'MFCC dir : ' + opt.mfcc_dir + '\n'
        summary += 'With energy : ' + opt.mfcc_energy + '\n'
        summary += 'Num cep : ' + str(opt.numcep) + '\n'
    if 'preprocess' in tasks:
        summary += 'Preprocessing with parameters :\n'
        summary += 'MFCC dir : ' + opt.mfcc_dir + '\n'
        summary += 'States dir : ' + opt.states_dir + '\n'
        summary += 'Method : ' + opt.preprocess_method + '\n'
        if opt.preprocess_method in ['kmeans', 'partial_kmeans']:
            summary += 'Number of clusters : ' + str(opt.num_clusters) + '\n'
        if opt.preprocess_method == 'partial_kmeans':
            summary += 'Size of the subset : ' + str(opt.num_clusters) + '\n'
    if 'train' in tasks:
        summary += 'Training with parameters :\n'
        summary += 'Train list : ' + opt.train_list + '\n'
        summary += 'Model type : ' + opt.train_model_type + '\n'
        summary += 'Embedding dimension : ' + str(opt.embed_dim) + '\n'
        summary += 'Hidden dimension : ' + str(opt.hidden_dim) + '\n'
        summary += 'Optimization method : ' + opt.optim + '\n'
        summary += 'Loss function : ' + opt.loss + '\n'
        summary += 'BPTT span : ' + str(opt.span) + '\n'
        summary += 'Batch size : ' + str(opt.batch_size) + '\n'
        summary += 'Stateful : ' + str(opt.stateful) + '\n'
    if 'test' in tasks:
        summary += 'Testing with parameters :\n'
        summary += 'Test list : ' + opt.test_list + '\n'
        summary += 'Output dir : ' + opt.out_dir + '\n'
        summary += 'Model type : ' + opt.test_model_type + '\n'
        summary += 'Embedding dimension : ' + str(opt.embed_dim) + '\n'
        summary += 'Hidden dimension : ' + str(opt.hidden_dim) + '\n'
        summary += 'Optimization method : ' + opt.optim + '\n'
        summary += 'Error function : ' + opt.loss + '\n'
    if 'postprocess' in tasks:
        summary += 'Postprocessing with parameters :\n'
        summary += 'Output dir dir : ' + opt.out_dir+'_syldet' + '\n'
        summary += 'Method : ' + opt.postprocess_method + '\n'
        if opt.time_dir is not None:
            summary += 'Times directory : ' + opt.time_dir + '\n'
        summary += 'Threshold : ' + str(opt.threshold) + '\n'
        summary += 'Min threshold : ' + str(opt.min_threshold) + '\n'
        if opt.preprocess_method == 'manual':
            summary += 'Kernel size : ' + str(opt.ker_len) + '\n'
            summary += 'Clip limit : ' + str(opt.clip) + '\n'
    if 'eval' in tasks:
        summary += 'Evaluation with parameters :\n'
        summary += 'Gold dir : ' + opt.gold_dir + '\n'
        summary += 'Result file : ' + 'This file dumbass' + '\n'
        summary += 'Gap : ' + str(opt.gap) + '\n'

    return summary


if __name__ == '__main__':

    opt = vars(arguments.parser.parse_args())

    if opt['json'] is not None:
        with open(opt['json']) as f:
            opt.update(json.load(f))

    opt = Options(**opt)
    tasks = set(opt.workflow.split('|'))

    model_weights = None
    if opt.model_weights is not None:
        print(opt.model_weights)
        model_weights = np.load(opt.model_weights).items()[0][1]

    dir2lists(opt)

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
        model_weights = run_rnn.train(
            opt.train_list,
            opt.out_dir,
            train_model_weights=model_weights,
            train_model_type=opt.train_model_type,
            embed_dim=opt.embed_dim,
            hidden_dim=opt.hidden_dim,
            optim=opt.optim,
            loss=opt.loss,
            span=opt.span,
            batch_size=opt.batch_size,
            stateful=opt.stateful,
            verbose=opt.verbose
        )
    # Test model
    if 'test' in tasks:
        run_rnn.test(
            opt.test_list,
            opt.out_dir,
            trained_model_weights=model_weights,
            test_model_type=opt.test_model_type,
            embed_dim=opt.embed_dim,
            hidden_dim=opt.hidden_dim,
            optim=opt.optim,
            error=opt.loss,
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
            threshold=opt.threshold,
            min_threshold=opt.min_threshold
        )
    # Evaluation
    if 'eval' in tasks:
        run_eval.run(
            opt.out_dir+'_syldet',
            opt.gold_dir,
            opt.
            res_file,
            gap=opt.gap,
            summary=summarize(opt),
            remove_trailing_silences=opt.no_silences
        )
