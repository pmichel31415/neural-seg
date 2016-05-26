#
# Copyright 2016  ENS LSCP (Author: Paul Michel)
#


import numpy as np
from scipy.signal import convolve, argrelmax
import argparse

parser = argparse.ArgumentParser(
    description='Peakdet on rnn error for syllable detection')

parser.add_argument('-i', action='store', dest='input_file',
                    required=True, type=str, help='Input csv file')
parser.add_argument('-vad', action='store', dest='vad_file',
                    default='', type=str, help='Input vad file')
parser.add_argument('-t','--times', action='store', dest='time_file',
                    default=None, type=str, help='Input time file')
parser.add_argument('-c', action='store', dest='clip',
                    default=0, type=float, help='Clip limit')
parser.add_argument('-r', action='store', dest='rate',
                    default=100.0,
                    type=float, help='Input sampling rate (Hz)')
parser.add_argument('-k', action='store', dest='ker_len',
                    default=7.0,
                    type=float, help='Convolution kernel length')
parser.add_argument('-o', action='store', dest='output_file',
                    required=True, type=str, help='Output file (boundaries)')
parser.add_argument('-v', '--verbose', help='increase output verbosity',
                    action='store_true')

if __name__ == '__main__':
    opt = parser.parse_args()
    # Load error signal
    x = np.loadtxt(opt.input_file)
    x = x.reshape(x.size)
    
    times=np.arange(len(x))/opt.rate
    if opt.time_file is not None:
        times=np.loadtxt(opt.time_file)

    
    kernel = np.ones((opt.ker_len)) / opt.ker_len
    x_smoothed = convolve(x, kernel)

    # Optionally take VAD into account
    if opt.vad_file!='':
        vad=np.loadtxt(opt.vad_file).astype(int)
        for i in range(len(vad)):
            # Set error to 0 when there isn't any voice activity
            x_smoothed[vad[i,0]:vad[i,1]]=0

        if opt.verbose:
            print('zeroed',int(np.sum(vad[:,1]-vad[:,0])),'frames')

    boundaries = argrelmax(x_smoothed)[0]
    boundaries = np.append(boundaries, len(x)-1)
    boundaries = np.insert(boundaries, 0, 0)
    boundaries = times[boundaries]
    
    # Optionaly clip all boundaries that are too close apart
    if opt.clip>0:
        y = [boundaries[0]]
        i = 0
        for j in range(1,len(boundaries)):
            if boundaries[j]-boundaries[i] >= opt.clip:
                boundaries[i:j]=np.mean(boundaries[i:j])
                i=j
            j+=1

        for bound in boundaries:
            if bound!=y[-1]:
                y.append(bound)
        if opt.verbose:
            print('Clipped',len(boundaries)-len(y),'boundaries')
        boundaries=np.array(y)

    np.savetxt(opt.output_file, boundaries, fmt="%.3f")
