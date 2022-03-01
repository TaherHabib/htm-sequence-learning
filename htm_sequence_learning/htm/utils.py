"""
MODULE DESCRIPTION:
-------------------
Utility function used in HTM model

"""

import numpy as np
import random
from scipy.special import comb as ncr


def get_num_columns(columns_per_char=None, chars=None):
    return columns_per_char * len(chars)


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
        list_idx_nonZeroElements.append((list_row_idx_nonZero[i], list_col_idx_nonZero[i]))

    return list_idx_nonZeroElements


def false_match_prob(k=None, tot_neurons=None, connSynapses_perdend=None, nmda_threshold=None):
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
    for i in range(nmda_threshold, connSynapses_perdend + 1):
        summ += ncr(connSynapses_perdend, i, exact=True) * ncr(tot_neurons - connSynapses_perdend, k - i, exact=True)

    prob = summ / ncr(tot_neurons, k, exact=True)

    return prob


def A_winner_cells(rg=None, M=None):
    A_winner_cells = np.zeros([M, rg.N], dtype=np.int8)
    random.seed(1)
    A_winnercells_i = random.choices(np.arange(M), k=rg.k)
    for i in range(rg.k):
        A_winner_cells[A_winnercells_i[i], rg.df_CharsToMinicols['A'][i]] = 1
    return A_winner_cells