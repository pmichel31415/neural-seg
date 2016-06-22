from __future__ import print_function, division

import numpy as np
import os
import argparse

from scipy.io.wavfile import read as wavread
from scipy.signal import resample

import kmodels

from keras.callbacks import EarlyStopping

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
parser.add_argument('-bs', '--batch_size', action='store', dest='batch_size',
                    default=10,
                    type=int, help='Batch size')
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
parser.add_argument('-mt', '--model_type', action='store', dest='model_type',
                    default='simple_rnn',
                    type=str, help='Model type to be used')
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
    np.savez(filename,list(model.get_weights()))

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

class StatefulDataGenerator:
    def __init__(self, data, span, batch_size,model=None):
        self.files = data
        np.random.shuffle(self.files)
        self.current_files=np.random.randint(0,len(data),size=(batch_size))
        self.current_indexes=np.zeros(batch_size)
        self.n=len(data)
        self.batch_size=batch_size
        self.model=model

    def __iter__(self):
        return self

    def next(self):
        for i in range(self.batch_size):
            if self.current_indexes[i]+2>=len(self.files[self.current_files[i]]):
                self.current_indexes[i]=0
                self.current_files[i]=(self.current_files[i]+1) % self.n
                if self.model is not None:
                    self.model.reset_states()
            else:
                self.current_indexes[i]+=1
        x=np.array([self.files[self.current_files[i]][self.current_indexes[i]].reshape(1,-1) for i in range(self.batch_size)])
        y=np.array([self.files[self.current_files[i]][self.current_indexes[i]+1].reshape(1,-1) for i in range(self.batch_size)])
        return (x,y)
    def get(self):
        return self.next()

class DataGenerator:
    def __init__(self, data, span, batch_size):
        self.data = data
        self.n=len(data)
        self.span=span
        self.batch_size=batch_size

    def __iter__(self):
        return self

    def next(self):
        batch_files=np.random.randint(0,self.n,size=(self.batch_size))
        batch_idxs=np.array([np.random.randint(0,len(self.data[f])-self.span-1) for f in batch_files])
        x=np.array([self.data[batch_files[i]][batch_idxs[i]:batch_idxs[i]+self.span] for i in range(self.batch_size)])
        y=np.array([self.data[batch_files[i]][batch_idxs[i]+1:batch_idxs[i]+self.span+1] for i in range(self.batch_size)])
        return (x,y)
    def get(self):
        return self.next()

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
    
    if hasattr(kmodels,'build_' + opt.model_type):
        build_model=getattr(kmodels,'build_' + opt.model_type)
    else:
        print('Unknown model type :/ you\'ll have to code it yourself')
        exit()
    model = build_model(opt.embed_dim,opt.hidden_dim,opt.embed_dim,weights,None,opt.batch_size)
    if opt.verbose:
        print(' Compiling model with ')
        print(' - optimizer : ' + opt.optim)
        print(' - loss function : ' + opt.loss)
    model.compile(
            optimizer=opt.optim,
            loss=opt.loss#kmodels.last_mse
            )
    
    if opt.verbose:
        print('----------------------------------------')
        print('       Running training on corpus       ')
        print('----------------------------------------')
    
    train_files=[]

    with open(opt.train_list, 'r') as f:
        train_files = [l for l in f.read().split('\n') if l.endswith('.npy')]
    train_data=[]
    valid_data=[]
    raw_data=[]
    for tf in train_files:
        if opt.verbose:
            print('         Loading training file          ')
            print(' - file :',tf)
            print('----------------------------------------')

        current_inputs = load_features(tf)
        if current_inputs.shape[1] != opt.embed_dim:
            print('Wrong input dim :', current_inputs.shape[1],
                  ', expected', opt.embed_dim,
                  'in file', f, ', skipping to next file.')
            continue
        raw_data.append(current_inputs)
    np.random.shuffle(raw_data)
    valid_split=int(0.1*len(train_files))
    valid_data=raw_data[:valid_split]
    train_data=raw_data[valid_split:]

    print(len(train_data))
    batch_generator=StatefulDataGenerator(train_data,opt.span,opt.batch_size,model)
    validation_generator=StatefulDataGenerator(valid_data,opt.span*2,opt.batch_size)
    print(next(batch_generator)[0].shape,next(batch_generator)[1].shape)
    model.fit_generator(
        batch_generator,
        sum(len(f)-opt.span for f in train_data)//2, # Sample per epoch
        1000, # epoch
        validation_data=validation_generator,
        nb_val_samples=10,
        callbacks=[EarlyStopping(monitor='val_loss', patience=0, verbose=1, mode='auto')],
        verbose=2
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
        y=np.zeros((len(current_inputs),opt.embed_dim))
        loss=np.zeros(len(current_inputs))
        test_x=np.array(current_inputs[:-1])
        test_x=test_x.reshape(len(test_x),1,-1)
        print(test_x.shape)
        test_y=current_inputs[1:]

        res=model.predict(test_x,batch_size=1)
        #y=model.predict_on_batch(test_x)
        y[1:]=res.reshape(len(res),-1)
        print(y.shape) 
        
        cosine=np.sum(test_y*y[1:],axis=-1)/(np.linalg.norm(test_y,ord=2,axis=-1)*np.linalg.norm(y[1:],ord=2,axis=-1)+0.0001)
        
        #loss[1:]=1-cosine
        loss[1:]=np.sqrt(np.mean(np.square(y[1:]-test_y),axis=-1))
        
        print(y.shape,test_y.shape,loss.shape)
        out_file = opt.out_dir + '/' + os.path.splitext(f.split('/')[-1])[0] + '_loss.npy'
        np.save( opt.out_dir + '/' + os.path.splitext(f.split('/')[-1])[0] + '.npy',y)
        np.save(out_file,loss)
    
    if opt.verbose:
        print('----------------------------------------')
        print('      Testing over, dumping model       ')
        print('----------------------------------------')

    save_model_weights(opt.out_dir+"/dump_model",model)
