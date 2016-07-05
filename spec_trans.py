from __future__ import print_function, division

import numpy as np
import argparse
import os
from scipy.signal import argrelmax, argrelmin

parser = argparse.ArgumentParser(
    description='Differential speech segmentation')

parser.add_argument('-i', '--input_data', action='store', dest='input_list',
                    required=True,
                    type=str, help='File listing input files locations')
parser.add_argument('-o', '--out_dir', action='store', dest='out_dir',
                    required=True,
                    type=str, help='Output dir')
parser.add_argument('-s', '--span', action='store', dest='span',
                    default=2,
                    type=int, help='Span of the difference calculation')
parser.add_argument('-r', '--sampling_rate', action='store', dest='rate',
                    default=100.0,
                    type=float, help='Input sampling rate')
parser.add_argument('-v', '--verbose', help='increase output verbosity',
                    action='store_true')

def load_features(filename):
    if filename.endswith('.npy'):
        feats = np.load(filename)
    else:
        feats = np.loadtxt(filename)
    feats=feats[:,:10]
    feats=feats-np.mean(feats,axis=0)
    feats=feats/np.std(feats,axis=0)
    return feats

def calculate_diff(mfcc,I):
    N=len(mfcc)
    diff=np.zeros(N)
    diff[I:-I]=np.mean(np.square(sum(mfcc[n+I:N-I+n]*n for n in range(-I,I+1))/np.sum(np.arange(-I,I+1)**2)),axis=1)
    #diff[I:-I]=np.mean(np.square(sum(mfcc[n+I:N-I+n]*np.arange(n+I,N-I+N) for n in range(-I,I+1))/sum(np.arange(n+I,N-I+N))))
    return diff

def upper_valley(b,v):
    i=0
    while i<len(v)-1 and v[i]<b:
        i+=1
    return i

def check_valleys(diff,i):
    left=True
    right=True
    li=i-1
    ri=i+1
    while li>=0:
        if li-1<0 or diff[li-1] > diff[li]: #then this is a valley
            left = abs(diff[i]-diff[li])>=0.1*diff[i]
            break
        li=li-1
    
    while ri<len(diff):
        if ri+1==len(diff) or diff[ri+1] > diff[ri]: #then this is a valley
            right = abs(diff[ri]-diff[i])>=0.1*diff[i]
            break
        ri=ri+1
    return left and right

def post_process(diff):
    maxpeak=np.max(diff)
    threshold = maxpeak * 0.01
    valley_threshold=maxpeak * 0.1
    potential_boundaries=argrelmax(diff)[0]
    valleys=argrelmin(diff)[0]
    boundaries=[]
    for i,pb in enumerate(potential_boundaries):
        if pb==0 or pb == len(diff):
            boundaries.append(pb)
            continue

        if diff[pb]-diff[pb-1] < threshold or diff[pb]-diff[pb+1] < threshold:
            continue
        if not check_valleys(diff,pb):
            continue
        #j=upper_valley(pb,valleys)
        #if j>0 and valleys[j]>pb and valleys[j-1]<pb:
        #    if pb-valleys[j] < valley_threshold or pb-valleys[j-1] < valley_threshold:
        #        continue
        boundaries.append(pb)

    return np.array(boundaries)



if __name__=='__main__':

    opt = parser.parse_args()

    input_files=[]

    with open(opt.input_list, 'r') as f:
        input_files = [l for l in f.read().split('\n') if l.endswith('.npy')]
    
    
    for i,f in enumerate(input_files):
        if opt.verbose:
            print('Processing file', f,'(',i+1,'/',len(input_files),')')
        current_inputs = load_features(f)
        diff=calculate_diff(current_inputs,opt.span)
        np.save(opt.out_dir + '/' + os.path.splitext(f.split('/')[-1])[0] + '_loss.npy',diff)
        boundaries=post_process(diff) / opt.rate
        out_file = opt.out_dir + '/' + os.path.splitext(f.split('/')[-1])[0] + '.syldet'
        np.savetxt(out_file,boundaries,fmt='%.2f')
