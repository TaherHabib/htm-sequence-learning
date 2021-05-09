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
    matrix_2 : float array of shape (<nof_matrices>,M,N).

    Returns
    -------
    list of float, scalar values
    
    NOTE
    ----
    This function also accepts Boolean (or binary) arrays as inputs.

    """
    
    M, N = np.shape(matrix_1) 
    
    # when only 1 array of MxN is passed in 'matrix_2', reshape it to (1,M,N)
    # so that the function output is appropriate.
    if np.shape(matrix_2) == (M, N):
        matrix_2 = np.reshape(matrix_2, (1, M, N))
        
    mult_res = np.multiply(matrix_1, matrix_2, dtype=np.float64)
    
    result = []
    for i in range(len(mult_res)):
        result.append(np.sum(mult_res[i], dtype=np.float64))
    
    return np.array(result, dtype=np.float64)



def get_idx_nonZeroElements(matrix=None):
    """
    Return a list of tuples (row_idx, col_idx) of all the non-zero elements in 
    a given 'matrix'

    Parameters
    ----------
    matrix : any int/float matrix. The default is None.

    Returns
    -------
    list_idx_nonZeroElements : list of 2-tuples.

    """
    
    list_idx_nonZeroElements = []
    
    list_row_idx_nonZero = np.where(matrix)[0]
    list_col_idx_nonZero = np.where(matrix)[1]
    
    for i in range(len(list_col_idx_nonZero)):
        list_idx_nonZeroElements.append((list_row_idx_nonZero[i] , list_col_idx_nonZero[i]))
    
    return list_idx_nonZeroElements

def reberstring_length(rg_io):
    
    return len(rg_io[0])


def get_graph_allPossibleTransitions(graph=None):
    
    all_possibleTransitions = []
    
    for i, subsequent_1 in enumerate(graph[0][1]):
        transition_to_1 = graph[0][0][i]
    
        for j, subsequent_2 in enumerate(graph[transition_to_1][1]):
            all_possibleTransitions.append('A'+subsequent_1+subsequent_2)
    
    for startNode in graph:
        if startNode[1] == 'Z':
            continue
        
        for i, antecedent in enumerate(startNode[1]):
            if len(startNode[1]) == 1:
                transition_to_1 = startNode[0]
            else:
                transition_to_1 = startNode[0][i]
                
            
            for j, subsequent_1 in enumerate(graph[transition_to_1][1]): 
                
                if graph[transition_to_1][1] == 'Z':
                    all_possibleTransitions.append(antecedent+subsequent_1)
                    break
                elif len(graph[transition_to_1][1]) == 1:
                    transition_to_2 = graph[transition_to_1][0]
                else:
                    transition_to_2 = graph[transition_to_1][0][j]
                
                for subsequent_2 in graph[transition_to_2][1]:
                    all_possibleTransitions.append(antecedent+subsequent_1+subsequent_2)        
                
    return all_possibleTransitions


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

    '''
    
    f = open(filename+'.pickle', 'wb')
    pickle.dump(data, f)
    f.close()
    
    

def unpickle(filename=None):
    '''
    Loads and returns a pickled object.

    '''
    
    f = open(filename+'.pickle', 'rb')
    data = pickle.load(f)
    f.close()
    
    return data



def compress_pickle(filename=None, data=None):
    '''
    Pickle a file and then compress it into BZ2 file. 

    '''
    
    with bz2.BZ2File(filename+'.pbz2', 'wb') as f: 
        cPickle.dump(data, f)
        


def decompress_pickle(filename=None):
    '''
    Load any compressed pickle file.

    '''
    
    data = bz2.BZ2File(filename+'.pbz2', 'rb')
    data = cPickle.load(data)
    
    return data


#_______________________________NOTES_________________________________________

# 1. http://www.linfo.org/bzip2.html for details on bzip2 file compression
