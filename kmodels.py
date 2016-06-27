from __future__ import print_function, division

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.noise import GaussianDropout
from keras.layers.wrappers import TimeDistributed

import theano
import theano.tensor as T

def build_feed_forward(dx,dh,do,length,weights=None):
    model=Sequential()
    model.add(TimeDistributed(Dense(
        dh,
        activation='sigmoid'
        ),
        input_shape=(1,dx)))
    model.add(TimeDistributed(Dense(do)))
    if weights is not None:
        model.set_weights(weights)
    return model


def build_simple_rnn(dx,dh,do,length,weights=None):
    model=Sequential()
    model.add(SimpleRNN(
        dh,
        input_dim=dx,
        return_sequences=True
        ))
    model.add(TimeDistributed(Dense(do)))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_simple_rnn_stateful(dx,dh,do,length,weights=None,batch_size=1):
    model=Sequential()
    model.add(SimpleRNN(
        dh,
        batch_input_shape=(batch_size,1,dx),
        return_sequences=True,
        stateful=True
        ))
    model.add(TimeDistributed(Dense(do)))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_stacked_rnn(dx,dh,do,length,weights=None):
    model=Sequential()
    model.add(SimpleRNN(
        dh,
        input_dim=dx,
        return_sequences=True
        ))
    model.add(SimpleRNN(
        do,
        input_dim=dh,
        return_sequences=True,
        ))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_lstm(dx,dh,do,length,weights=None):
    model=Sequential()
    model.add(LSTM(
        dh,
        input_dim=dx,
        return_sequences=True
        ))
    model.add(TimeDistributed(Dense(do)))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_lstm_stateful(dx,dh,do,length,weights=None,batch_size=1):
    model=Sequential()
    model.add(LSTM(
        dh,
        batch_input_shape=(batch_size,1,dx),
        return_sequences=True,
        stateful=True
        ))
    model.add(TimeDistributed(Dense(do)))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_train_lstm_softmax(dx,dh,do,span=1,weights=None,batch_size=2):
    model=Sequential()
    model.add(LSTM(
        dh,
        input_dim=dx,
        return_sequences=False
        ))
    model.add(Dense(do))
    model.add(Activation('softmax'))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_test_lstm_softmax(dx,dh,do,weights=None):
    model=Sequential()
    model.add(LSTM(
        dh,
        input_dim=dx,
        return_sequences=True
        ))
    model.add(TimeDistributed(Dense(do)))
    model.add(TimeDistributed(Activation('softmax')))
    if weights is not None:
        model.set_weights(weights)
    return model


def build_lstm_stateful_softmax(dx,dh,do,length=1,weights=None,batch_size=1):
    model=Sequential()
    model.add(LSTM(
        dh,
        batch_input_shape=(batch_size,length,dx),
        return_sequences=False,
        stateful=True
        ))
    model.add(Dense(do))
    model.add(Activation('softmax'))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_stacked_lstm_dropout_stateful_softmax(dx,dh,do,length=1,weights=None,batch_size=1):
    model=Sequential()
    model.add(LSTM(
        dh,
        batch_input_shape=(batch_size,length,dx),
        return_sequences=True
        ))
    model.add(Dropout(0.2))
    model.add(LSTM(
        dh,
        batch_input_shape=(batch_size,length,dh),
        return_sequences=False
        ))
    model.add(Dense(do))
    model.add(Activation('softmax'))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_train_stacked_lstm_dropout_softmax(dx,dh,do,span=1,weights=None,batch_size=2):
    model=Sequential()
    model.add(LSTM(
        dh,
        input_dim=dx,
        return_sequences=True
        ))
    model.add(Dropout(0.2))
    model.add(LSTM(
        dh,
        input_dim=dh,
        return_sequences=False
        ))
    model.add(Dense(do))
    model.add(Activation('softmax'))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_stacked_lstm_dropout_softmax(dx,dh,do,weights=None):
    model=Sequential()
    model.add(LSTM(
        dh,
        input_dim=dx,
        return_sequences=True
        ))
    model.add(Dropout(0.2))
    model.add(LSTM(
        dh,
        input_dim=dh,
        return_sequences=True
        ))
    model.add(TimeDistributed(Dense(do)))
    model.add(TimeDistributed(Activation('softmax')))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_stacked_lstm_dropout(dx,dh,do,length,weights=None):
    model=Sequential()
    model.add(LSTM(
        dh,
        input_dim=dx,
        return_sequences=True
        ))
    model.add(Dropout(0.2))
    model.add(LSTM(
        do,
        input_dim=dh,
        return_sequences=True
        ))
    model.add(TimeDistributed(Dense(do)))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_stacked_lstm(dx,dh,do,length,weights=None):
    model=Sequential()
    model.add(LSTM(
        dh,
        input_dim=dx,
        return_sequences=True
        ))
    model.add(LSTM(
        do,
        input_dim=dh,
        return_sequences=True
        ))
    model.add(TimeDistributed(Dense(do)))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_stacked_lstm_stateful(dx,dh,do,length,weights=None,batch_size=5):
    model=Sequential()
    model.add(LSTM(
        dh,
        batch_input_shape=(batch_size,1,dx),
        return_sequences=True,
        stateful=True
        ))
    model.add(LSTM(
        do,
        batch_input_shape=(batch_size,1,dh),
        return_sequences=True,
        stateful=True
        ))
    model.add(TimeDistributed(Dense(do)))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_stacked_lstm_stateful_dropout(dx,dh,do,length,weights=None,batch_size=5):
    model=Sequential()
    model.add(LSTM(
        dh,
        batch_input_shape=(batch_size,1,dx),
        return_sequences=True,
        stateful=True
        ))
    model.add(Dropout(0.2))
    model.add(LSTM(
        do,
        batch_input_shape=(batch_size,1,dh),
        return_sequences=True,
        stateful=True
        ))
    model.add(TimeDistributed(Dense(do)))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_stacked_lstm_regularized(dx,dh,do,length,weights=None):
    model=Sequential()
    model.add(LSTM(
        dh,
        input_dim=dx,
        return_sequences=True,
        W_regularizer='l2',
        U_regularizer='l2',
        b_regularizer='l2'
        ))
    model.add(LSTM(
        do,
        input_dim=dh,
        return_sequences=True,
        activation='linear',
        W_regularizer='l2',
        U_regularizer='l2',
        b_regularizer='l2'
        ))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_stacked_lstm_regularized_dropout(dx,dh,do,length,weights=None):
    model=Sequential()
    model.add(LSTM(
        dh,
        input_dim=dx,
        return_sequences=True,
        W_regularizer='l2',
        U_regularizer='l2',
        b_regularizer='l2'
        ))
    model.add(Dropout(0.2))
    model.add(LSTM(
        do,
        input_dim=dh,
        return_sequences=True,
        activation='linear',
        W_regularizer='l2',
        U_regularizer='l2',
        b_regularizer='l2'
        ))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_stacked_lstm_regularized_dropout_batchnorm(dx,dh,do,length,weights=None):
    model=Sequential()
    model.add(LSTM(
        dh,
        input_dim=dx,
        return_sequences=True,
        W_regularizer='l2',
        U_regularizer='l2',
        b_regularizer='l2'
        ))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(LSTM(
        do,
        input_dim=dh,
        return_sequences=True,
        activation='linear',
        W_regularizer='l2',
        U_regularizer='l2',
        b_regularizer='l2'
        ))
    if weights is not None:
        model.set_weights(weights)
    return model

def build_overkill_stacked_lstm_regularized_dropout(dx,dh,do,length,weights=None):
    model=Sequential()
    model.add(LSTM(
        dh,
        input_dim=dx,
        return_sequences=True,
        W_regularizer='l2',
        U_regularizer='l2',
        b_regularizer='l2'
        ))
    model.add(Dropout(0.2))
    model.add(LSTM(
        512,
        input_dim=dh,
        return_sequences=True,
        W_regularizer='l2',
        U_regularizer='l2',
        b_regularizer='l2'
        ))
    model.add(Dropout(0.2))
    model.add(LSTM(
        do,
        input_dim=512,
        return_sequences=True,
        activation='linear',
        W_regularizer='l2',
        U_regularizer='l2',
        b_regularizer='l2'
        ))
    if weights is not None:
        model.set_weights(weights)
    return model

def last_mse(y_true,y_pred):
    yt=y_true[:,-1,:]
    yp=y_pred[:,-1,:]

    se=T.mean(T.square(yt-yp),axis=-1)

    return se 

def build_softmax_rnn(dx,dh,do,length,weights=None):
    model=Sequential()
    model.add(SimpleRNN(
        dh,
        input_dim=dx,
        return_sequences=True
        ))
    model.add(TimeDistributed(Dense(do),activation='softmax'))
    
    if weights is not None:
        model.set_weights(weights)
    return model

# Set model
build_model=build_lstm

if __name__ == '__main__':
    dummy_params= 10,15,5,0,None
    build_lstm(*dummy_params)
    build_simple_rnn(*dummy_params)
    build_softmax_rnn(*dummy_params)
    build_stacked_lstm_dropout(*dummy_params)

