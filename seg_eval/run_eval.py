from __future__ import print_function, division

import os
import numpy as np
import load
import match


def remove_silences(gold, bounds, gap):
    inf, sup = gold[1], gold[-2]
    bounds = [b for b in bounds if b >= inf-gap and b <= sup+gap]
    gold = gold[1:-1]
    return gold, bounds


def eval_file(
    bounds_file,
    gold_file,
    gap=0.02,
    remove_trailing_silences=False
):

    gold = load.load_seg(gold_file)
    bounds = load.load_seg(bounds_file)
    
    gold=(gold*100).astype(int)/100

    if remove_trailing_silences:
        gold, bounds = remove_silences(gold, bounds, gap)
    # if not isinstance(bounds,list) or len(bounds)==0:
    #     return [0,0,len(gold)]
    n_bounds = len(bounds)
    n_gold = len(gold)

    matches, deletions, insertions = match.match_eval(bounds, gold, gap)
    #np.savetxt(bounds_file[:-7]+'_matches.syldet', matches, fmt='%.3f')
    n_matches = len(matches)
    n_deletions = len(deletions)
    n_insertions = len(insertions)

    return [n_matches, n_bounds, n_gold]


def run(
    bounds_dir,
    gold_dir,
    out_file,
    gap=0.02,
    summary='',
    remove_trailing_silences=False
):
    results = []
    for f in os.listdir(bounds_dir):
        if not f.endswith('.syldet'):
            continue

        input_file = bounds_dir + '/' + f
        gold_file = gold_dir + '/' + f

        if not os.path.exists(gold_file):
            continue

        results.append(
            eval_file(input_file, gold_file, gap, remove_trailing_silences))

    results = np.array(results).sum(axis=0)
    print(gap)
    print(results)
    precision = results[0]/(results[1]+0.00000000001)
    recall = results[0]/results[2]
    F_1 = 2*precision*recall/(precision+recall)

    # mean_prec = np.mean(precision)
    # mean_recall = np.mean(recall)
    # mean_F1 = np.mean(F_1)

    with open(out_file, 'w+') as f:
        f.write(summary)
        f.write('Results :')
        f.write('\n Precision %.3f' % precision)
        f.write('\n Recall %.3f' % recall)
        f.write('\n F-score %.3f' % F_1)
        f.write('\n')
