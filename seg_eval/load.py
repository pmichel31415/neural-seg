import numpy as np
import subprocess

def load_seg(filename):
    """
    :returns: array containing the boundaries (s)
    """
    if filename.endswith('.txt') or filename.endswith('.csv') or filename.endswith('.syldet'):
        segs = np.loadtxt(filename)
    elif filename.endswith('.t7'):
        filename=filename[:-3]
        subprocess.call('th','t72npy.lua','-i',filename+'.t7','-o',filename+'.npy')
        segs = np.load(filename+'.npy')
    elif filename.endswith('.npy'):
        segs = np.load(filename)
    else:
        print('ERROR : Unrecognized file type ".'+filename.split('.')[-1]+'"')
        exit()
    return segs