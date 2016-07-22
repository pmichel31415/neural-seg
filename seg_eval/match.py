import numpy as np 
from scipy.spatial.distance import pdist

def pdist2(a,b):
    D=np.abs(a.reshape(-1,1)-b.reshape(1,-1))
    return D

def match_eval(bounds,gold,gap_threshold=0.02):
    """
    Evaluates number of insertions, mismatches and deletions in two 
    boundaries sequences
    :param bounds: array of times (s)
    :param gold: gold times (s)
    :returns: tuple (matches,deletions,insertions)
    """

    bounds=np.array(bounds)
    gold=np.array(gold)
    bounds=bounds.reshape(-1,1)
    gold=gold.reshape(-1,1)

    bounds=bounds[bounds>=0]
    gold=gold[gold>=0]

    n_gold = len(gold)
    n_bounds = len(bounds)

    i_gold,i_bounds=0,0

    matches =[]
    deletions = []
    insertions =[]


    D = pdist2(bounds,gold)
    D = np.sort(D,axis=0)
    return np.arange(D.shape[1])[D[0,:]<= gap_threshold],None,None
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
