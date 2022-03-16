"""
MODULE DESCRIPTION:
-------------------

This module runs the main experiments. I takes... #TODO

OTHER NOTES:
------------
Issue 001:
    When a column bursts, but no (matching) dendrite with connections to the previous time step's activity are found
    AND when all HTM cells in a given minicolumn run out of their capacity to grow any new dendrite (given by
    'maxDendritesPerCell').

Issue 002:
    When a dendrite has more synapses than its capacity given by 'maxSynapsesPerDendrite'.

Issue 003:
    When multiple matching dendrites are found in a bursting column.

Issue 004:
    To be read in the same context as Issue 001. See htm_net.py.

Issue 005:
    This issue reports a fundamental flaw in the learning of SDRs. If the total number of cells with permanence
    reinforcement on any one of their dendrites at any given time step during execution falls below the set NMDA
    threshold of the network, issue 005 is reported at the output terminal. It breaks the execution of the program for
    the current reber string and starts execution from the next reber string in the input stream. In the current
    implementation of HTM, this issue is generally found to be in 5% of the total number of reber strings in the
    inputstream.
"""

import os
import argparse
import logging
import json
from pickle_utils import full_pickle
import numpy as np
import pandas as pd
from pathlib import Path
from htm_sequence_learning.reber_grammar.utils import get_graph_from_dataset
from htm_sequence_learning.htm.utils import A_winner_cells
from htm_sequence_learning.reber_grammar.ReberGrammar import Reber_Grammar
from htm_sequence_learning.htm.htm_net import HTM_NET

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
logger.addHandler(handler)

ROOT = os.path.abspath(Path(__file__).parent.parent)
model_config_path = os.path.join(ROOT, 'configs', 'htm')
datastream_path = os.path.join(ROOT, 'data', 'reber_strings_dataset')
results_save_path = os.path.join(ROOT, 'data', 'experiment_results')
if not os.path.exists(results_save_path):
    os.makedirs(results_save_path)


default_config = 'default_config.json'
default_dataset = 'graph2_mix_numStrings2000_ergFalse.npy'

parser = argparse.ArgumentParser(description='Train an HTM model on generated Reber Grammar Strings')
parser.add_argument('-cd', '--default_config', dest='run_default_config', action='store', nargs='?', const=True,
                    default=True, help='')
parser.add_argument('-dd', '--default_dataset', dest='run_default_dataset', action='store', nargs='?', const=True,
                    default=True, help='')
parser.add_argument('-c', '--config', dest='config_json', action='store', default=None, type=str,
                    help='')
parser.add_argument('-d', '--dataset', dest='dataset_npy', action='store', default=None, type=str,
                    help='')
parser.add_argument('-n', '--normalize_permanence', dest='normalize_permanence', action='store', nargs='?',
                    const=True, default=False, help='')
parser.add_argument('-p', '--prune_dendrites', dest='prune_dendrites', action='store', nargs='?',
                    const=True, default=False, help='')
parser.add_argument('-s', '--save_results', dest='save_results', action='store', nargs='?', const=True, default=False,
                    help='')
parser.add_argument('-v', '--verbosity', dest='verbosity_level', action='store', default=1, choices=[0, 1, 2],
                    type=int, help='')


def run_experiment(data=None,
                   htm_network=None,
                   A_winner_cells=None,
                   z_onehot=None,
                   normalize_permanence=False,
                   prune_dendrites=False,
                   verbosity=1):
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
        issue = 'nan'

        in_string = data[string_idx][1]
        in_string_alpha = data[string_idx][0]
        curr_pred = np.zeros([htm_network.M, htm_network.N], dtype=np.int8)

        if verbosity > 0:
            print('\n', color.BOLD, 'String_idx: ', string_idx, color.END, 'Input String: ', in_string_alpha)

        # 'len(in_string) is actually one less than the actual length of the string, due to the final omission of 'Z'.
        for step in range(len(in_string)):
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
    dict_results = {
        'df_results': df_res.set_index('reber_string', inplace=True),
        'final_network': htm_network.net_architecture
    }

    return dict_results


if __name__ == '__main__':

    args = parser.parse_args()

    # Setting up model configuration
    if args.config_json is not None:
        logger.info('Building HTM network with configurations from: {}'.format(args.config_json))
        with open(os.path.join(model_config_path, args.config_json.replace('.json', '') + '.json'), 'r') as config:
            model_params = json.load(config)
        model_name = args.config_json.replace('.json', '')
    else:
        logger.info('Building HTM network with default configurations from: {}'.format(default_config))
        with open(os.path.join(model_config_path, default_config), 'r') as config:
            model_params = json.load(config)
        model_name = default_config.replace('.json', '')

    # Setting up Reber strings dataset
    if args.dataset_npy is not None:
        logger.info('Using dataset: {}'.format(args.dataset_npy))
        with open(os.path.join(datastream_path, args.dataset_npy.replace('.npy', '') + '.npy'), 'rb') as datafile:
            rg_inputoutput = np.load(datafile, allow_pickle=True)
        graph_idx = get_graph_from_dataset(args.dataset_npy)
        dataset_name = args.dataset_npy.replace('.npy', '')
    else:
        logger.info('Using default dataset: {}'.format(default_dataset))
        with open(os.path.join(datastream_path, default_dataset), 'rb') as datafile:
            rg_inputoutput = np.load(datafile, allow_pickle=True)
        graph_idx = get_graph_from_dataset(default_dataset)
        dataset_name = default_dataset.replace('.npy', '')

    logger.info('Defining Grammar object for later downstream processing tasks')
    grammar = Reber_Grammar(columns_per_char=model_params['columns_per_char'], graph_idx=graph_idx)

    A_winner_cells = A_winner_cells(rg=grammar, M=model_params['cells_per_column'])  # Get 'A' winner cells
    z_onehot = grammar.CharToOnehot('Z')  # Get Onehot for 'Z'

    logger.info('Initializing HTM Network')
    htm_network = HTM_NET.from_json(model_params=model_params, verbosity=args.verbosity_level)

    logger.info('Running the model...')
    dict_results = run_experiment(data=rg_inputoutput,
                                  htm_network=htm_network,
                                  A_winner_cells=A_winner_cells,
                                  z_onehot=z_onehot,
                                  normalize_permanence=args.normalize_permanence,
                                  prune_dendrites=args.prune_dendrites,
                                  verbosity=args.verbosity_level)
    # Saving to disk
    if args.save_results:
        logger.info('Saving results to disk...')
        results_file_name = 'model{}_dataset{}_normalize{}_prune{}'.format(model_name,
                                                                           dataset_name,
                                                                           args.normalize_permanence,
                                                                           args.prune_dendrites)
        dict_results['df_results'].to_hdf(path_or_buf=os.path.join(results_save_path, results_file_name + '.hdf5'),
                                          mode='w', key=results_save_path, complevel=7)
        full_pickle(filename=os.path.join(results_save_path, results_file_name+'_network'),
                    data=dict_results['final_network'])
        full_pickle(filename=os.path.join(results_save_path, results_file_name + '_grammar'),
                    data=grammar)

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
