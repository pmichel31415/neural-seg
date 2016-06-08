from __future__ import print_function, division

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.noise import GaussianDropout
from keras.layers.wrappers import TimeDistributed



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
        activation='linear'
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
        return_sequences=True,
        activation='linear'
        ))
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
        return_sequences=True,
        activation='linear'
        ))
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

