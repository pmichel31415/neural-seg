from __future__ import print_function, division

import numpy as np
import load
import match
import argparse

parser = argparse.ArgumentParser(
    description='Speech segmentation evaluation tool')

parser.add_argument('-g', action='store', dest='gold_file',
                    default='test/new_english.phn',
                    type=str, help='Gold file')
parser.add_argument('-b', action='store', dest='bounds_file',
                    default='test/500ms_gold.classes',
                    type=str, help='Boundaries file')
parser.add_argument('-t', action='store', dest='gap',
                    default=0.02,
                    type=float, help='Gap threshold (left and right) in s')
parser.add_argument('-o', action='store', dest='out_file',
                    default='',
                    type=str, help='Output file')
parser.add_argument('-v', '--verbose', help='increase output verbosity',
                    action='store_true')

if __name__ == '__main__':

    args = parser.parse_args()

    gold = load.load_seg(args.gold_file)
    bounds = load.load_seg(args.bounds_file)

    matches, deletions, insertions = match.match_eval(bounds, gold,args.gap)

    n_bounds = len(bounds)
    n_gold = len(gold)
    n_matches = len(matches)
    n_deletions = len(deletions)
    n_insertions = len(insertions)

    precision = n_matches / n_bounds
    recall = n_matches / n_gold

    F_1 = 2 * precision * recall / (precision + recall)

    if args.verbose:
        print('precision : %.1f' % (100 * precision), '%')
        print('recall : %.1f' % (100 * recall), '%')
        print('F1-score : %.1f' % (100 * F_1), '%')
        print('insertions : %.1f' % (100 * n_insertions/n_bounds), '%')
        print('deletions : %.1f' % (100 * n_deletions/n_gold), '%')
    else:
        print(precision,recall,F_1)

    if args.out_file != '':
        x=np.array(insertions)/gold[-1]
        np.savetxt(args.out_file,x,fmt="%.3f")

    