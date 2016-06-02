from __future__ import print_function, division

import numpy as np
import os
import argparse

from scipy.io.wavfile import read as wavread
from scipy.signal import resample

from kmodels import build_model

parser = argparse.ArgumentParser(
    description='Neural speech segmentation')


parser.add_argument('-train', '--train_data', action='store', dest='train_list',
                    required=True,
                    type=str, help='File listing training files locations')
parser.add_argument('-test', '--test_data', action='store', dest='test_list',
                    required=True,
                    type=str, help='File listing test files locations')
parser.add_argument('-m', '--model', action='store', dest='model',
                    default=None,
                    type=str, help='Previously saved model')
parser.add_argument('-de', '--dim_embed', action='store', dest='embed_dim',
                    default=39,
                    type=int, help='Embedding dimension')
parser.add_argument('-dh', '--dim_hidden', action='store', dest='hidden_dim',
                    default=50,
                    type=int, help='Hidden layer dimension')
parser.add_argument('-s', '--span', action='store', dest='span',
                    default=7,
                    type=int, help='BPTT time span')
parser.add_argument('-c', '--chunk', action='store', dest='chunk_size',
                    default=6000,
                    type=int, help='Chunks in which to divide the training files')
parser.add_argument('-optim', '--optimizer', action='store', dest='optim',
                    default='sgd',
                    type=str, help='Optimizer')
parser.add_argument('-loss', '--loss_function', action='store', dest='loss',
                    default='mse',
                    type=str, help='Loss function')
parser.add_argument('-l', '--lr', action='store', dest='lr',
                    default=0.01,
                    type=float, help='Learning rate')
parser.add_argument('-o', '--out_dir', action='store', dest='out_dir',
                    required=True,
                    type=str, help='Output dir')
parser.add_argument('-v', '--verbose', help='increase output verbosity',
                    action='store_true')
parser.add_argument('-nt', '--notrain', help='No training',
                    action='store_true')
parser.add_argument('-w', '--wav', help='Use wav files instead of mfcc',
                    action='store_true')

def load_features(filename):
    if filename.endswith('.npy'):
        feats = np.load(filename)
    else:
        feats = np.loadtxt(filename)
    #feats=feats-np.mean(feats,axis=0)
    #feats=feats/np.std(feats,axis=0)
    return feats

def load_wav(filename):
    if filename.endswith('.wav'):
        fs, x = wavread(filename)
        if fs != 8000:
            x=resample(x,int(16000/fs*len(x)))
        return x
    return np.array([])

def save_model_weights(filename,model):
    np.save(filename,np.concatenate(model.get_weights()))

def load_model_weights(filename):
    weights=np.load(filename)
    weights=[weights[i] for i in range(weights.shape[0])]
    return weights

def prep_train(data,chunk_size=6000):
    train_x=[]
    train_y=[]

    for i in range(0,len(data)-chunk_size):
        end=min(len(data)-1,i+chunk_size)
        train_x.append(data[i:end])
        train_y.append(data[i+1:end+1])

    train_x=np.array(train_x)
    train_y=np.array(train_y)
    return train_x,train_y


if __name__ == '__main__':
    opt = parser.parse_args()

    if opt.verbose:
        print('----------------------------------------')
        print('            Creating model              ')
        print(' - input dim :', opt.embed_dim)
        print(' - hidden dim :', opt.hidden_dim)
        print(' - output dim :', opt.embed_dim)
        print('----------------------------------------')


    weights=None
    if opt.model is not None:
        if opt.verbose:
            print(' Initializing model with custom weights ')
            print(' - source :', os.path.basename(opt.model))
        weights=load_model_weights(opt.model)
    else:
        if opt.verbose:
            print(' Initializing model with random weights')
    
    model = build_model(opt.embed_dim,opt.hidden_dim,opt.embed_dim,weights)
    if opt.verbose:
        print(' Compiling model with ')
        print(' - optimizer : ' + opt.optim)
        print(' - loss function : ' + opt.loss)
    model.compile(
            optimizer=opt.optim,
            loss=opt.loss
            )
    
    if opt.verbose:
        print('----------------------------------------')
        print('       Running training on corpus       ')
        print('----------------------------------------')
    
    train_files=[]

    with open(opt.train_list, 'r') as f:
        train_files = [l for l in f.read().split('\n') if l.endswith('.npy')]
    
    for tf in train_files:
        if opt.verbose:
            print('         Loading training file          ')
            print(' - file :',tf)
            print('----------------------------------------')

        train_data=[]

        current_inputs = load_features(tf)
        if current_inputs.shape[1] != opt.embed_dim:
            print('Wrong input dim :', current_inputs.shape[1],
                  ', expected', opt.embed_dim,
                  'in file', f, ', skipping to next file.')
            continue
        
    
        train_x,train_y=prep_train(current_inputs,chunk_size=opt.chunk_size)
    
        model.fit(
                train_x,train_y,
                batch_size=1,
                nb_epoch=1,
                validation_split=0.0
                )

    
    if opt.verbose:
        print('----------------------------------------')
        print('       Running testing on corpus        ')
        print('----------------------------------------')
    
    test_files=[]

    with open(opt.test_list, 'r') as f:
        test_files = [l for l in f.read().split('\n') if l.endswith('.npy')]
    
    
    for i,f in enumerate(test_files):
        current_inputs = load_features(f)
        if current_inputs.shape[1] != opt.embed_dim:
            print('Wrong input dim :', current_inputs.shape[1],
                  ', expected', opt.embed_dim,
                  'in file', f, ', skipping to next file.')
            continue
        if opt.verbose:
            print('Running through file',f,'(',i,'/',len(test_files),')')
        loss=np.zeros(len(current_inputs))
        test_x=np.array([current_inputs[:-1]])
        test_y=current_inputs[1:]
        y=model.predict(test_x,batch_size=1,verbose=0)
        
        loss[1:]=np.sqrt(np.sum(np.square(y-test_y),axis=2))
        
        out_file = opt.out_dir + '/' + os.path.splitext(f.split('/')[-1])[0] + '.loss'
        np.savetxt(out_file,loss,fmt='%.3f')
    
    if opt.verbose:
        print('----------------------------------------')
        print('      Testing over, dumping model       ')
        print('----------------------------------------')

    save_model_weights(opt.out_dir+"/dump_model",model)
