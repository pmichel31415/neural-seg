import numpy as np 

def match_eval(bounds,gold,gap_threshold=0.02):
    """
    Evaluates number of insertions, mismatches and deletions in two 
    boundaries sequences
    :param bounds: array of times (s)
    :param gold: gold times (s)
    :returns: tuple (matches,deletions,insertions)
    """

    bounds.sort()
    gold.sort()

    n_gold = len(gold)
    n_bounds = len(bounds)

    i_gold,i_bounds=0,0

    matches =[]
    deletions = []
    insertions =[]

    while i_gold<n_gold or i_bounds<n_bounds:
        if i_gold<n_gold and i_bounds<n_bounds:
            diff = bounds[i_bounds]-gold[i_gold]
        else:
            diff=np.Inf
        if np.abs(diff)<=gap_threshold:
            matches.append(bounds[i_bounds])
            i_bounds+=1
            i_gold+=1
        elif diff < -gap_threshold or i_gold==n_gold:
            insertions.append(bounds[i_bounds])
            i_bounds+=1
        elif diff>gap_threshold or i_bounds==n_bounds:
            deletions.append(gold[i_gold])
            i_gold+=1

    return matches,deletions,insertions
