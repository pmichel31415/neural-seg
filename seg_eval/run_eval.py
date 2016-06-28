from __future__ import print_function, division

import os
import numpy as np
import load
import match

def eval_file(bounds_file, gold_file, gap=0.02):

    gold = load.load_seg(gold_file)
    bounds = load.load_seg(bounds_file)

    matches, deletions, insertions = match.match_eval(bounds, gold, gap)

    n_bounds = len(bounds)
    n_gold = len(gold)
    n_matches = len(matches)
    n_deletions = len(deletions)
    n_insertions = len(insertions)

    precision = n_matches / n_bounds
    recall = n_matches / n_gold

    F_1 = 2 * precision * recall / (precision + recall)

    return [n_matches, n_bounds, n_gold]


def run(bounds_dir, gold_dir, out_file, gap=0.02, summary=''):
    results = []
    for f in os.listdir(bounds_dir):
        if not f.endswith('.syldet'):
            continue

        input_file = bounds_dir + '/' + f
        gold_file = gold_dir + '/' + f

        if not os.path.exists(gold_file):
            continue

        results.append(eval_file(input_file, gold_file, gap))

    results = np.array(results).T

    precision = results[0]/results[1]
    recall = results[0]/results[2]
    F_1 = 2*precision*recall/(precision+recall)

    mean_prec = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_F1 = np.mean(F_1)

    with open(out_file, 'w+') as f:
        f.write(summary)
        f.write('Results :')
        f.write('\n Precision %.3f' % mean_prec)
        f.write('\n Recall %.3f' % mean_recall)
        f.write('\n F-score %.3f' % mean_F1)
        f.write('\n')
