"""
MODULE DESCRIPTION:
-------------------

This module runs the main experiments.

OTHER NOTES:
------------
Issue 001:
    When a column bursts, but no (matching) dendrite with connections to the previous timestep's activity
    are found AND when all HTM cells in a given minicolumn run out of their capacity to grow any new
    dendrite (given by 'maxDendritesPerCell').

Issue 002:
    When a dendrite has more synapses than its capacity given by 'maxSynapsesPerDendrite'.

Issue 003:
    When multiple matching dendrites are found in a bursting column.

Issue 004:
    To be read in the same context as Issue 001. See htm_net.py.

Issue 005:
    This issue reports a fundamental flaw in the learning of SDRs. If the total number of cells with
    permanence reinforcement on any one of their dendrites at any given timestep during execcution
    falls below the set NMDA threshold of the network, issue 005 is reported at the output terminal.
    It breaks the execution of the program for the current reber string and starts execution from the
    next reber string in the input stream.
    In the current implementation of HTM, this issue is generally found to be in 5% of the total
    number of reber strings in the inputstream.
"""

import numpy as np
import pandas as pd
import copy


def run_experiment():
    # DataFrame to store results for each string in 'list_in_strings'
    df_res = pd.DataFrame(columns=('reber_string', 'htm_states', 'htm_preds', 'htm_predDendrites',
                                   'htm_winner_cells', 'nof_dendrites', 'issue'))

    # 'htm_states' and 'htm_preds' store MxN binary state and prediction matrix of HTM network at each timestep
    # (each letter), for each input reber string, respectively.

    # 'htm_predDendrites' stores MxN matrix of responsible active dendrites for each of the MxN neuron's prediction
    # in HTM network at each timestep (each letter), for each input reber string.

    # 'htm_networks' stores MxN matrix of HTM cells at each timestep, for each input reber string. This storage
    # is mainly to have an access to the evolution of the synaptic permanence values of each cell in the
    # network with time.



