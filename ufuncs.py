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



# Calculation of Probability for a False Match of SDRs------------------------
def false_match_prob(a,n,s,th):
    summ = 0
    for i in range(th, s+1):
        summ += ncr(s,i, exact=True)*ncr(n-s,a-i, exact=True)
    
    prob = summ/ncr(n,a, exact=True)
    return prob

# =============================================================================
# a = 25 # value of 'k'
# n = 1400 # total cells in network
# s = 32 # number of connected synapses on a dendritic segment
# th = 15  # NMDA threshold 
#         
# false_match_prob(a,n,s,th)
# =============================================================================
