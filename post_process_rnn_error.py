#
# Copyright 2016  ENS LSCP (Author: Paul Michel)
#
from __future__ import print_function, division

import numpy as np
import os
from scipy.signal import convolve, argrelmax
from scipy.fftpack import rfft, rfftfreq, irfft
from peakdet import detect_peaks


def check_valleys(x, i, threshold=1):
    left = True
    right = True
    li = i-1
    ri = i+1
    while li >= 0:
        if li-1 < 0 or x[li-1] > x[li]:  # then this is a valley
            left = abs(x[i]-x[li]) >= threshold
            break
        li = li-1

    # while ri<len(x):
    #    if ri+1==len(x) or x[ri+1] > x[ri]: #then this is a valley
    #        right = abs(x[ri]-x[i])>=0.1*x[i]
    #        break
    #    ri=ri+1
    return left  # and right


def cliffs(x):
    potential_boundaries = argrelmax(x)[0]
    ret = []
    for i, pb in enumerate(potential_boundaries):
        li = i-1
        left = abs(x[i]-x[0])
        while li >= 0:
            if li-1 < 0 or x[li-1] > x[li]:  # then this is a valley
                left = abs(x[i]-x[li])
            break
            li = li-1
        ret.append([pb, left])
    return ret


def greedy_detect(x, times, num=5):
    diffs = np.array(cliffs(x))
    diffs = diffs[diffs[:, 1].argsort()]
    lim = int(len(x)/num)
    diffs = np.sort(diffs[-lim:, 0]).astype(int)
    return times[diffs]


def baseline_like_detect(x, times, threshold=1, min_threshold=1):
    #x = 1-np.exp(-x)
    potential_boundaries = argrelmax(x)[0]
    boundaries = []
    mean = np.mean(x[potential_boundaries])
    for i, pb in enumerate(potential_boundaries):
        if pb == 0 or pb == len(x):
            boundaries.append(pb)
            continue

        if x[pb] < min_threshold*mean:
            continue
        if not check_valleys(x, pb, threshold):
            continue
        # j=upper_valley(pb,valleys)
        # if j>0 and valleys[j]>pb and valleys[j-1]<pb:
        #    if pb-valleys[j] < valley_threshold or pb-valleys[j-1] < valley_threshold:
        #        continue
        boundaries.append(pb)

    return times[boundaries]


def manual_detect(x, times, ker_len, clip, rate):

    kernel = np.ones((int(ker_len))) / ker_len
    x_smoothed = convolve(x, kernel)

    boundaries = argrelmax(x_smoothed)[0]
    boundaries = np.append(boundaries, len(x)-1)
    boundaries = np.insert(boundaries, 0, 0)
    boundaries = times[boundaries]

    # Optionaly clip all boundaries that are too close apart
    if clip > 0:
        y = [boundaries[0]]
        i = 0
        for j in range(1, len(boundaries)):
            if boundaries[j]-boundaries[i] >= clip:
                boundaries[i:j] = np.mean(boundaries[i:j])
                i = j
            j += 1

        for bound in boundaries:
            if bound != y[-1]:
                y.append(bound)
        boundaries = np.array(y)

    return boundaries


def fourier_detect(x, times, rate):
    fr = rfftfreq(len(times), 1/rate)
    y = rfft(x)
    y[fr > int(1/0.05)] = 0
    x_smoothed = irfft(y)
    return times[argrelmax(x_smoothed)[0]]


def auto_detect(x, times, ker_len):

    kernel = np.ones((int(ker_len))) / ker_len
    x_smoothed = convolve(x, kernel)
    boundaries = detect_peaks(x_smoothed, mph=np.max(x_smoothed)*0.4, mpd=2,)
    boundaries = times[boundaries]

    return boundaries


def post_process_file(
    input_file,
    output_file,
    method='baseline',
    time_file=None,
    rate=100.0,
    ker_len=3,
    clip=0.03,
    threshold=0.5,
    min_threshold=1
):
    # Load error signal
    x = np.load(input_file)
    x = x.reshape(x.size)

    # Flatten beginning

    x[:7]=0

    times = np.arange(len(x))/rate
    if time_file is not None:
        times = np.loadtxt(time_file)

    if method == 'fourier':
        boundaries = fourier_detect(x, times, rate)
    elif method == 'auto':
        boundaries = auto_detect(x, times, ker_len)
    elif method == 'manual':
        boundaries = manual_detect(x, times, ker_len, clip, rate)
    elif method == 'baseline':
        boundaries = baseline_like_detect(
            x,
            times,
            threshold=threshold,
            min_threshold=min_threshold
        )
    elif method == 'greedy':
        boundaries = greedy_detect(x, times, threshold)
    elif method == 'none':
        boundaries = times[argrelmax(x)[0]]
    else:
        boundaries = fourier_detect(x, times, rate)
    boundaries=list(boundaries)
    if not (len(x)-1)/rate in boundaries:
        boundaries.append((len(x)-1)/rate)
    if not 0 in boundaries:
        boundaries=[0]+boundaries
    np.savetxt(output_file, boundaries, fmt="%.2f")


def run(
    input_dir,
    output_dir,
    method='baseline',
    time_dir=None,
    rate=100.0,
    ker_len=3,
    clip=0.03,
    threshold=0.5,
    min_threshold=1
):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for f in os.listdir(input_dir):
        if f.endswith('_loss.npy'):

            ifile = input_dir+'/'+f
            ofile = output_dir+'/'+f[:-9]+'.syldet'

            if time_dir is not None:
                tfile = time_dir + f[:-9]+'times'
            else:
                tfile = None

            post_process_file(
                ifile,
                ofile,
                method=method,
                time_file=tfile,
                rate=rate,
                ker_len=ker_len,
                clip=clip,
                threshold=threshold,
                min_threshold=min_threshold
            )
