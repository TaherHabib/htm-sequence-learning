"""
MODULE DESCRIPTION:
-------------------

This module runs the main experiments. I takes... #TODO

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


def run_experiment(data=None,
                   htm_network=None,
                   A_winner_cells=None,
                   z_onehot=None,
                   verbosity=1):

    # DataFrame to store results for each string in the inputstream
    df_res = pd.DataFrame(columns=['reber_string', 'htm_states', 'htm_preds', 'htm_pred_dendrites',
                                   'htm_winner_cells', 'num_net_dendrites', 'issue'])

    # 'htm_states' and 'htm_preds' store MxN binary state and prediction matrix of HTM network at each time step
    # (each letter in the reber string), for each input reber string, respectively.

    # 'htm_pred_dendrites' stores MxN matrix of responsible active dendrites for each of the MxN neuron's prediction
    # in HTM network at each time step (each letter), for each input reber string.

    # 'htm_networks' stores MxN matrix of HTM cells at each time step, for each input reber string. This storage
    # is mainly to have access to the evolution of the synaptic permanence values of each cell in the
    # network with time.

    for string_idx in range(len(data)):

        htm_states = []
        htm_preds = []
        htm_pred_dendrites = []
        htm_winner_cells = []

        in_string = data[string_idx][1]
        in_string_alpha = data[string_idx][0]
        curr_pred = np.zeros([htm_network.M, htm_network.N], dtype=np.int8)

        if verbosity > 0:
            print('\n', color.BOLD, 'String_idx: ', string_idx, color.END, 'Input String: ', in_string_alpha)

        # 'len(in_string) is actually one less than the actual length of the string,
        # due to the final ommission of 'Z'.
        for step in range(len(in_string)):

            if verbosity > 1:
                print(color.BLUE, color.BOLD, 'Running for step: ', step, ' for: ', in_string_alpha[step],
                      color.END, color.END)

            # in_string[step] is a binary 1xN vector (np.array) with 'k' 1s.
            curr_state, curr_pred, curr_pred_dend = htm_network.get_net_state(prev_pred=curr_pred,
                                                                              curr_input=in_string[step])
            if step == 0:

                winnerCells = A_winner_cells

                htm_states.append(curr_state)
                htm_preds.append(curr_pred)
                htm_pred_dendrites.append(curr_pred_dend)
                htm_winner_cells.append(winnerCells)

                # No learning can occur for 'A' and its prediction.
                continue

            else:

                htm_network.update_net_dendriteDutyCycle()

                # HEBBIAN LEARNING & SYNAPTIC PERMANENCE UPDATE
                # Here, the network is learning to predict for symbol that is currrently in 'in_string[step]'
                winnerCells = htm_network.update_net_synaptic_permanences(curr_state=curr_state,
                                                                              prev_state=htm_states[step - 1],
                                                                              prev_pred=htm_preds[step - 1],
                                                                              prev_predDendrites=htm_pred_dendrites[
                                                                                  step - 1],
                                                                              prev_winnerCells=htm_winner_cells[
                                                                                  step - 1]
                                                                              )
                # __________________________________________________________________
                # The following checks for a fundamental flaw in implementation of
                # the learning of SDRs
                # REPORT ERROR AND BREAK THE EXECUTION for the current string!
                if np.count_nonzero(winnerCells) < htm_network.nmda_threshold:
                    print(color.RED, color.BOLD, '::::::::::::::::::::::::::::::::::Issue 005 in step: ', step, \
                          '::::::::::::::::::::::::::::::::::', color.END, color.END)
                    issue = '005'
                    break

                else:
                    issue = 'nan'

                htm_states.append(curr_state)
                htm_preds.append(curr_pred)
                htm_pred_dendrites.append(curr_pred_dend)
                htm_winner_cells.append(winnerCells)
                # htm_networks.append(self.htm_network.get_NETWORK(char_minicols=in_string[step]))

            # LEARNING TO PREDICT 'Z' at the penultimate step
            if step == len(in_string) - 1:

                if verbosity > 1:
                    print(color.BLUE, color.BOLD, 'Running for step: ', step + 1, ' for: Z', color.END, color.END)

                curr_state, _, _ = htm_network.get_net_state(prev_pred=curr_pred,
                                                             curr_input=z_onehot)
                htm_states.append(curr_state)
                # Since there won't be any predictions occurring at the timestep of 'Z', as input,
                # 'curr_pred' and 'curr_pred_dend' need not be appended at all. Also, NONE of the cells
                # in the network would be reinforcing their pre-synapses with the cells responsible
                # for 'Z'. In other words, the output of 'dot_prod(net_state,cell_connSynapses)' in
                # 'get_onestep_prediction()' will always be all zero, at this step!

                htm_network.update_net_dendriteDutyCycle()

                _ = htm_network.update_net_synaptic_permanences(curr_state=curr_state,
                                                               prev_state=htm_states[step],
                                                               prev_pred=htm_preds[step],
                                                               prev_predDendrites=htm_pred_dendrites[step],
                                                               prev_winnerCells=htm_winner_cells[step]
                                                               )
                # htm_networks.append(self.htm_network.get_NETWORK(char_minicols=self.z_minicols))

        count_dend = 0
        for i in range(htm_network.M):
            for j in range(htm_network.N):
                count_dend += htm_network.net_arch[i, j].get_cell_numDendrites()

        df_res.loc[string_idx] = [in_string_alpha,
                                  np.array(htm_states),
                                  np.array(htm_preds),
                                  np.array(htm_pred_dendrites),
                                  np.array(htm_winner_cells),
                                  count_dend,
                                  issue]

        # np.array(htm_states) is numpy array of shape: (<len(in_string)>+1,M,N)
        # np.array(htm_preds) is numpy array of shape: (<len(in_string)>,M,N)
        # np.array(htm_preds_dend) is numpy array of shape: (<len(in_string)>,M,N)
        # np.array(htm_winner_cells) is numpy array of shape: (<len(in_string)>,M,N)

    dict_results = {
        'results': df_res,
        'final_net': htm_network.net_arch
    }

    return dict_results


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
