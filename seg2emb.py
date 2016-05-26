from __future__ import print_function, division

import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description='Phone segmentation to new embedding')

parser.add_argument('-s', action='store', dest='seg_file',
                    required=True,
                    type=str, help='Segmentation file')
parser.add_argument('-i', action='store', dest='input_file',
                    required=True,
                    type=str, help='Input file (features)')
parser.add_argument('-d', action='store', dest='div',
                    default=3,
                    type=int,
                    help='Number of parts in which to divide each phones')
parser.add_argument('-r', action='store', dest='rate',
                    default=100,
                    type=float,
                    help='Sampling rate of the input')
parser.add_argument('-o', action='store', dest='out_file',
                    required=True,
                    type=str, help='Output file')
parser.add_argument('-t', action='store', dest='time_file',
                    required=True,
                    type=str, help='Time file')
parser.add_argument('-v', '--verbose', help='increase output verbosity',
                    action='store_true')

if __name__ == '__main__':

    opt = parser.parse_args()

    x = np.load(opt.input_file)
    seg = np.loadtxt(opt.seg_file)

    ret = np.zeros((len(seg) - 1, x.shape[1] * 3))
    times = np.zeros(len(seg)-1)
    for i in range(len(seg)-1):
        # The segmentatons is supposed to have been post processed to that
        # all segments smaller than 30ms have been clipped
        sub = x[int(opt.rate*seg[i]):int(opt.rate*seg[i+1])]
        if len(sub)==0:
            ret[i]=np.zeros(3*x.shape[1])
            continue
        while len(sub)<3:
            print(sub.shape)
            sub=np.vstack((sub,sub[len(sub)-1].reshape(1,sub.shape[1])))

        a, b = len(sub) // 3, (2 * len(sub)) // 3 
        part1 = np.mean(sub[:a], axis=0)
        part2 = np.mean(sub[a:b], axis=0)
        part3 = np.mean(sub[b:], axis=0)

        ret[i] = np.concatenate((part1, part2, part3))
        times[i]=seg[i]
    np.save(opt.out_file, ret)
    np.save(opt.time_file, times)
