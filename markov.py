from __future__ import print_function, division

import numpy as np
import os
import argparse

train_dir = '../../resources/TIMIT/kmeans_states_8_train'
test_dir = '../../resources/TIMIT/kmeans_states_8_test'
out_dir = 'output/markov'

if not os.path.exists(out_dir):
        os.mkdir(out_dir)

k = 6
stats = np.zeros((8, k, 8))
prob=np.zeros(8)

for f in os.listdir(train_dir):
    if f.endswith('.npy'):
        x = np.load(train_dir+'/'+f)
        x = np.argmax(x, axis=-1)
        for i in range(k, len(x)):
            stats[x[i], range(k), x[i-k:i]] += 1
            prob[x[i]]+=1


nstats = stats/(prob.reshape(1, 1, -1))/k
for f in os.listdir(test_dir):
    if f.endswith('.npy'):
        x = np.load(test_dir+'/'+f)
        x = np.argmax(x, axis=-1)
        y = np.ones(len(x))
        for i in range(k, len(x)):
            y[i] = np.sum(nstats[x[i], range(k), x[i-k:i]])
        np.save(out_dir+'/'+f[:-4]+'_loss.npy',- np.log(y))
