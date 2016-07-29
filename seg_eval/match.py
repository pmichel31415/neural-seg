import numpy as np
from scipy.spatial.distance import pdist

epsilon = 0.00000001  # for float precision


def pdist2(a, b):
    D = np.abs(a.reshape(-1, 1)-b.reshape(1, -1))
    return D

def match_eval(bounds, gold, gap_threshold=0.02):
    """
    Evaluates number of insertions, mismatches and deletions in two 
    boundaries sequences
    :param bounds: array of times (s)
    :param gold: gold times (s)
    :returns: tuple (matches,deletions,insertions)
    """


    bounds = np.array(bounds)
    gold = np.array(gold)
    bounds = bounds.reshape(-1, 1)
    gold = gold.reshape(-1, 1)

    bounds = bounds[bounds >= 0]
    gold = gold[gold >= 0]

    n_gold = len(gold)
    n_bounds = len(bounds)

    i_gold, i_bounds = 0, 0

    matches = []
    deletions = []
    insertions = []

    left_margin=[gap_threshold+epsilon if i==0 or gold[i]-gold[i-1]+epsilon>=2*gap_threshold else (gold[i]-gold[i-1])/2+epsilon for i in range(len(gold))]
    right_margin=[gap_threshold+epsilon if i==len(gold)-1 or gold[i+1]-gold[i]+epsilon>=2*gap_threshold else (gold[i+1]-gold[i])/2+epsilon for i in range(len(gold))]
    found_gold=set()
    for i,g in enumerate(gold):
        for j,b in enumerate(bounds):
            if (b<=g and g-b<=left_margin[i]) or (g<=b and b-g<=right_margin[i]):
                found_gold.add(g)
    found_gold=list(found_gold)
    found_gold.sort()
    return np.array(found_gold),None,None

    # D = pdist2(bounds, gold)
    # # aD=np.argsort(D,axis=0)[0,:]
    # D = np.sort(D, axis=0)
    # # print(np.sum(D[0, :] <= (gap_threshold+epsilon)),np.sum(D[]))
    # return np.arange(D.shape[1])[D[0, :] <= (gap_threshold+epsilon)], None, None
    # meandist{level}(signal,j) = mean(D(1,:)/16000)
    # recall{level}(signal,j) = np.sum(D[0,:] <= allowed_deviation)/len(gold)
    # ins{level}(signal,j) = length(bounds)/length(refbounds)
    # precision{level}(signal,j) = sum(D(1,:)/16000 <= allowed_deviation)/length(bounds)
    # rval{level}(signal,j) = rvalue(recall{level}(signal,j)*100,ins{level}(signal,j)*100-100)
    # totbounds{level}(signal,j) = length(bounds)
    # while i_gold<n_gold or i_bounds<n_bounds:
    #     if i_gold<n_gold and i_bounds<n_bounds:
    #         diff = bounds[i_bounds]-gold[i_gold]
    #     else:
    #         diff=np.Inf
    #     if np.abs(diff)<=gap_threshold:
    #         matches.append(gold[i_gold])
    #         i_bounds+=1
    #         i_gold+=1
    #     elif diff < -gap_threshold or i_gold==n_gold:
    #         insertions.append(bounds[i_bounds])
    #         i_bounds+=1
    #     elif diff>gap_threshold or i_bounds==n_bounds:
    #         deletions.append(gold[i_gold])
    #         i_gold+=1

    # return matches,deletions,insertions
