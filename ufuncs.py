import bz2
import pickle
import _pickle as cPickle

import numpy as np
from scipy.special import comb as ncr


def dot_prod(matrix_1=None, matrix_2=None):
    """
    Computes the element-wise multiplication of an MxN matrix with a list of 'n'
    other equi-dimensional matrices (n,M,N); and outputs a list of scalar values 
    of sum over all the entries of each of the resulting 'n' matrices.

    Parameters
    ----------
    matrix_1 : float array of shape (M,N)
    matrix_2 : float array of shape (<nof_matrices>,M,N). If only one MxN matrix 
    is passed, it should be of shape (1,M,N).

    Returns
    -------
    list of float, scalar values
    
    NOTE
    ----
    This function also excepts Boolean (or binary) arrays as inputs.

    """
    
    mult_res = np.multiply(matrix_1, matrix_2, dtype=np.float64)
    
    result = []
    for i in range(len(mult_res)):
        result.append(np.sum(mult_res[i], dtype=np.float64))
    
    return np.array(result, dtype=np.float64)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def false_match_prob(k=None,tot_neurons=None,connSynapses_perdend=None,nmda_threshold=None):
    """
    Calculator of Probability for a False Match of SDRs

    Parameters
    ----------
    k : int; no. of winning columns per symbol.
    tot_neurons : int; total number of neurons in the network (= MxN).
    connSynapses_perdend : int; number of connected synapses per dendrite.
    nmda_threshold : int; threshold for NMDA spiking.

    Returns
    -------
    prob : float; Probability for a False Match of SDRs

    """
    summ = 0
    for i in range(nmda_threshold, connSynapses_perdend+1):
        summ += ncr(connSynapses_perdend,i, exact=True)*ncr(tot_neurons-connSynapses_perdend,k-i, exact=True)
    
    prob = summ/ncr(tot_neurons,k, exact=True)
    
    return prob



def full_pickle(filename=None, data=None):
    '''
    Saves the 'data' with the 'filename' as pickle    

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    f = open(filename+'.pickle', 'wb')
    pickle.dump(data, f)
    f.close()
    
    

def unpickle(filename=None):
    '''
    Loads and returns a pickled object.

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    '''
    
    f = open(filename+'.pickle', 'rb')
    data = pickle.load(f)
    f.close()
    
    return data



def compress_pickle(filename=None, data=None):
    '''
    Pickle a file and then compress it into BZ2 file. 

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    with bz2.BZ2File(filename+'.pbz2', 'wb') as f: 
        cPickle.dump(data, f)
        


def decompress_pickle(filename):
    '''
    Load any compressed pickle file.

    Parameters
    ----------
    filename : TYPE
        DESCRIPTION.

    Returns
    -------
    data : TYPE
        DESCRIPTION.

    '''
    
    data = bz2.BZ2File(filename+'.pbz2', 'rb')
    data = cPickle.load(data)
    
    return data


#_______________________________NOTES_________________________________________

# 1. http://www.linfo.org/bzip2.html for details on bzip2 file compression