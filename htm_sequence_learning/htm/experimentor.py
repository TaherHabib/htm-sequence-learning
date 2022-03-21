import numpy as np
import pandas as pd
from htm_sequence_learning.htm.utils import color


def run_experiment(data=None,
                   htm_network=None,
                   A_winner_cells=None,
                   z_onehot=None,
                   normalize_permanence=False,
                   prune_dendrites=False,
                   verbosity=1):

    string_step_lookup = []
    total_len_inputstream = 0

    # DataFrame to store results for each string in the inputstream
    df_res = pd.DataFrame(columns=['reber_string', 'htm_states', 'htm_preds', 'htm_pred_dendrites',
                                   'htm_winner_cells', 'num_net_dendrites', 'issue'])

    # 'htm_states' and 'htm_preds' store MxN binary state and prediction matrix of HTM network at each time step
    # (each letter in the reber string), for each input reber string, respectively.

    # 'htm_pred_dendrites' stores MxN matrix of responsible active dendrites for each of the MxN neuron's prediction
    # in HTM network at each time step (each letter), for each input reber string.

    # 'htm_networks' stores MxN matrix of HTM cells at each time step, for each input reber string. This storage is
    # mainly to have access to the evolution of the synaptic permanence values of each cell in the network with time.

    for string_idx in range(len(data)):

        htm_states = []
        htm_preds = []
        htm_pred_dendrites = []
        htm_winner_cells = []
        issue = 'none'

        in_string = data[string_idx][1]
        in_string_alpha = data[string_idx][0]
        curr_pred = np.zeros([htm_network.M, htm_network.N], dtype=np.int8)

        if verbosity > 0:
            print('\n', color.BOLD, 'String_idx: ', string_idx, color.END, 'Input String: ', in_string_alpha)

        # 'len(in_string) is actually one less than the actual length of the string, due to the final omission of 'Z'.
        for step in range(len(in_string)):
            string_step_lookup.append((in_string_alpha, step))
            total_len_inputstream += 1
            if verbosity > 1:
                print(color.BLUE, color.BOLD, 'Running for step: ', step, ' for: ', in_string_alpha[step],
                      color.END, color.END)

            # in_string[step] is a binary 1xN vector (np.array) with 'k' 1s.
            curr_state, curr_pred, curr_pred_dend = htm_network.get_net_state(prev_pred=curr_pred,
                                                                              curr_input=in_string[step])
            if step == 0:
                winner_cells = A_winner_cells
                htm_states.append(curr_state)
                htm_preds.append(curr_pred)
                htm_pred_dendrites.append(curr_pred_dend)
                htm_winner_cells.append(winner_cells)
                # No learning can occur for 'A' and its prediction.
                continue

            else:
                if prune_dendrites:
                    htm_network.update_net_dendrite_dutycycle()

                if normalize_permanence:
                    htm_network.normalize_net_permanence()

                # HEBBIAN LEARNING & SYNAPTIC PERMANENCE UPDATE
                # Here, the network is learning to predict for symbol that is currently in 'in_string[step]'
                winner_cells = htm_network.update_net_synaptic_permanences(curr_state=curr_state,
                                                                           prev_state=htm_states[step - 1],
                                                                           prev_pred=htm_preds[step - 1],
                                                                           prev_pred_dendrites=htm_pred_dendrites[step - 1],
                                                                           prev_winner_cells=htm_winner_cells[step - 1])
                # ____________________________________________________________________
                # The following checks for a fundamental flaw in the learning of SDRs
                # REPORT ERROR AND BREAK THE EXECUTION for the current string!
                if np.count_nonzero(winner_cells) < htm_network.nmda_threshold:
                    print(color.RED, color.BOLD, '::::::::::::::::::::::::::::::::::Issue 005 in step: ', step,
                          '::::::::::::::::::::::::::::::::::', color.END, color.END)
                    issue = '005'
                    break

                htm_states.append(curr_state)
                htm_preds.append(curr_pred)
                htm_pred_dendrites.append(curr_pred_dend)
                htm_winner_cells.append(winner_cells)
                # htm_networks.append(self.htm_network.get_NETWORK(char_minicols=in_string[step]))

            # LEARNING TO PREDICT 'Z' at the penultimate step
            if step == len(in_string) - 1:
                if verbosity > 1:
                    print(color.BLUE, color.BOLD, 'Running for step: ', step + 1, ' for: Z', color.END, color.END)
                curr_state, _, _ = htm_network.get_net_state(prev_pred=curr_pred, curr_input=z_onehot)
                htm_states.append(curr_state)
                # Since there won't be any predictions occurring at the timestep of 'Z', as input, 'curr_pred' and
                # 'curr_pred_dend' need not be appended at all. Also, NONE of the cells in the network would be
                # reinforcing their pre-synapses with the cells responsible for 'Z'. In other words, the output of
                # 'dot_prod(net_state,cell_connSynapses)' in 'get_onestep_prediction()' will be all zero, at this step!

                if prune_dendrites:
                    htm_network.update_net_dendrite_dutycycle()

                if normalize_permanence:
                    htm_network.normalize_net_permanence()

                _ = htm_network.update_net_synaptic_permanences(curr_state=curr_state,
                                                                prev_state=htm_states[step],
                                                                prev_pred=htm_preds[step],
                                                                prev_pred_dendrites=htm_pred_dendrites[step],
                                                                prev_winner_cells=htm_winner_cells[step])
                # htm_networks.append(self.htm_network.get_NETWORK(char_minicols=self.z_minicols))
                if normalize_permanence:
                    htm_network.normalize_net_permanence()

        count_total_dendrites = 0
        for i in range(htm_network.M):
            for j in range(htm_network.N):
                count_total_dendrites += htm_network.net_architecture[i, j].get_cell_numDendrites()
        df_res.loc[string_idx] = [in_string_alpha,
                                  np.array(htm_states),  # numpy array of shape: (<len(in_string)>+1,M,N)
                                  np.array(htm_preds),  # numpy array of shape: (<len(in_string)>,M,N)
                                  np.array(htm_pred_dendrites),  # numpy array of shape: (<len(in_string)>,M,N)
                                  np.array(htm_winner_cells),  # numpy array of shape: (<len(in_string)>,M,N)
                                  count_total_dendrites,
                                  issue]

    # df_res.set_index('reber_string', inplace=True)

    results_ = {
        'total_len_inputstream': total_len_inputstream,
        'string_step_lookup': np.array(string_step_lookup),
        'df_results': df_res,
        'final_network': htm_network
    }

    return results_
