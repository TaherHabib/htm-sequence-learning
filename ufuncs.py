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

def false_match_prob(k,tot_neurons,connSynapses_perdend,nmda_threshold):
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