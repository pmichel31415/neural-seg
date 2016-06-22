#
# Copyright 2016  ENS LSCP (Author: Paul Michel)
#


import numpy as np
from scipy.signal import convolve, argrelmax
from scipy.fftpack import rfft,rfftfreq,irfft
from peakdet import detect_peaks
import argparse

parser = argparse.ArgumentParser(
    description='Peakdet on rnn error for syllable detection')

parser.add_argument('-i', action='store', dest='input_file',
                    required=True, type=str, help='Input csv file')
parser.add_argument('-vad', action='store', dest='vad_file',
                    default='', type=str, help='Input vad file')
parser.add_argument('-t','--times', action='store', dest='time_file',
                    default=None, type=str, help='Input time file')
parser.add_argument('-n','--num', action='store', dest='num',
                    default=5, type=float, help='Num')
parser.add_argument('-c', action='store', dest='clip',
                    default=0, type=float, help='Clip limit')
parser.add_argument('-r', action='store', dest='rate',
                    default=100.0,
                    type=float, help='Input sampling rate (Hz)')
parser.add_argument('-k', action='store', dest='ker_len',
                    default=7.0,
                    type=float, help='Convolution kernel length')
parser.add_argument('-m', '--method', action='store', dest='method',
                    default='fourier', type=str, help='Input vad file')
parser.add_argument('-o', action='store', dest='output_file',
                    required=True, type=str, help='Output file (boundaries)')
parser.add_argument('-v', '--verbose', help='increase output verbosity',
                    action='store_true')

def check_valleys(x,i,threshold=1):
    left=True
    right=True
    li=i-1
    ri=i+1
    while li>=0:
        if li-1<0 or x[li-1] > x[li]: #then this is a valley
            left = abs(x[i]-x[li])>=threshold*np.mean(x)
            break
        li=li-1
    
    #while ri<len(x):
    #    if ri+1==len(x) or x[ri+1] > x[ri]: #then this is a valley
    #        right = abs(x[ri]-x[i])>=0.1*x[i]
    #        break
    #    ri=ri+1
    return left #and right

def cliffs(x):
    potential_boundaries=argrelmax(x)[0]
    ret=[]
    for i,pb in enumerate(potential_boundaries):
        li=i-1
        left=abs(x[i]-x[0])
        while li>=0:
            if li-1<0 or x[li-1] > x[li]: #then this is a valley
                left = abs(x[i]-x[li])
            break
            li=li-1
        ret.append([pb,left])
    return ret

def greedy_detect(x,times,num=5):
    diffs=np.array(cliffs(x))
    diffs=diffs[diffs[:,1].argsort()]
    lim = int(len(x)/num)
    diffs=np.sort(diffs[-lim:,0]).astype(int)
    return times[diffs]

def baseline_like_detect(x,times,threshold=1):
    potential_boundaries=argrelmax(x)[0]
    boundaries=[]
    for i,pb in enumerate(potential_boundaries):
        if pb==0 or pb == len(x):
            boundaries.append(pb)
            continue

        #if x[pb]-x[pb-1] < x[pb]*0.1 or x[pb]-x[pb+1] < x[pb]*0.1:
        #    continue
        if not check_valleys(x,pb,threshold):
            continue
        #j=upper_valley(pb,valleys)
        #if j>0 and valleys[j]>pb and valleys[j-1]<pb:
        #    if pb-valleys[j] < valley_threshold or pb-valleys[j-1] < valley_threshold:
        #        continue
        boundaries.append(pb)

    return times[boundaries]

def manual_detect(x,times,ker_len,clip,rate):

    kernel = np.ones((int(ker_len))) / ker_len
    x_smoothed = convolve(x, kernel)

    boundaries = argrelmax(x_smoothed)[0]
    boundaries = np.append(boundaries, len(x)-1)
    boundaries = np.insert(boundaries, 0, 0)
    boundaries = times[boundaries]
    
    # Optionaly clip all boundaries that are too close apart
    if clip>0:
        y = [boundaries[0]]
        i = 0
        for j in range(1,len(boundaries)):
            if boundaries[j]-boundaries[i] >= clip:
                boundaries[i:j]=np.mean(boundaries[i:j])
                i=j
            j+=1

        for bound in boundaries:
            if bound!=y[-1]:
                y.append(bound)
        boundaries=np.array(y)
    
    return boundaries

def fourier_detect(x,times,rate):
    fr=rfftfreq(len(times),1/rate)
    y=rfft(x)
    y[fr>int(1/0.05)]=0
    x_smoothed = irfft(y)
    return times[argrelmax(x_smoothed)[0]]

def auto_detect(x,times,ker_len):
    
    kernel = np.ones((int(ker_len))) / ker_len
    x_smoothed = convolve(x, kernel)
    boundaries=detect_peaks(x_smoothed,mph=np.max(x_smoothed)*0.4,mpd=2,)
    boundaries=times[boundaries]
    
    return boundaries

if __name__ == '__main__':
    opt = parser.parse_args()
    # Load error signal
    x = np.load(opt.input_file)
    x = x.reshape(x.size)
    
    times=np.arange(len(x))/opt.rate
    if opt.time_file is not None:
        times=np.loadtxt(opt.time_file)
    
    if opt.method=='fourier':
        boundaries=fourier_detect(x,times,opt.rate)
    elif opt.method=='auto':
        boundaries=auto_detect(x,times,opt.ker_len)    
    elif opt.method=='manual':
        boundaries=manual_detect(x,times,opt.ker_len,opt.clip,opt.rate)
    elif opt.method=='baseline':
        boundaries=baseline_like_detect(x,times,threshold=opt.num)
    elif opt.method=='greedy':
        boundaries=greedy_detect(x,times,opt.num)
    elif opt.method=='none':
        boundaries=times[argrelmax(x)[0]]
    else:
        boundaries=fourier_detect(x,times,opt.rate)

    np.savetxt(opt.output_file, boundaries, fmt="%.3f")


