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
import random
import numpy as np
from pathlib import Path

from htm_sequence_learning.reber_grammar.utils import get_graph_from_dataset
from htm_sequence_learning.htm.utils import A_winner_cells
from htm_sequence_learning.reber_grammar.ReberGrammar import Reber_Grammar
from htm_sequence_learning.htm.htm_net import HTM_NET
from htm_sequence_learning.htm.experimentor import run_experiment

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
default_dataset = 'graph2_mix_numStrings2000_ergFalse_sortFalse.npy'

parser = argparse.ArgumentParser(description='Train an HTM model on generated Reber Grammar Strings')
parser.add_argument('-q', '--default_config', dest='run_default_config', action='store', nargs='?', const=True,
                    default=True, help='Whether to use the default configuration of the HTM model. Specifying only the '
                                       'short flag without any argument leads to a value of True.')
parser.add_argument('-t', '--default_dataset', dest='run_default_dataset', action='store', nargs='?', const=True,
                    default=True, help='Whether to use the default dataset to run on the given HTM model. Specifying '
                                       'only the short flag without any argument leads to a value of True.')
parser.add_argument('-c', '--config', dest='config_json', action='store', default=None, type=str,
                    help='A .json file containing the configuration for the HTM model. See configs/htm/default_config.json'
                         'for the format of this file.')
parser.add_argument('-d', '--dataset', dest='dataset_npy', action='store', default=None, type=str,
                    help='A .npy file containing the input stream on which to run the HTM model. See '
                         'examples/reber_grammar_generator.py for details on how to generate input stream for the HTM model.')
parser.add_argument('-s', '--save_results', dest='save_results', action='store', nargs='?', const=True, default=False,
                    help='Whether to save the results of the experiment or not. Specifying only the short flag without '
                         'any argument leads to a value of True.')
parser.add_argument('-n', '--normalize_permanence', dest='normalize_permanence', action='store', nargs='?', const=True,
                    default=False, help='Whether to normalize (in the range [0,1]) the permanence values of the '
                                        'synaptic connections between neurons. Specifying only the short flag without '
                                        'any argument leads to a value of True.')
parser.add_argument('-p', '--prune_dendrites', dest='prune_dendrites', action='store', nargs='?', const=True, default=False,
                    help='Whether to prune the (unused) dendrites from the network. Specifying only the short flag '
                         'without any argument leads to a value of True.')
parser.add_argument('-v', '--verbosity', dest='verbosity_level', action='store', default=1, choices=[0, 1, 2],
                    type=int, help='Verbosity of the output. Choices are: 0, 1, 2')
parser.add_argument('-r', '--nof_runs', dest='nof_runs', action='store', default=1, type=int,
                    help='Number of runs of the experiment.')


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

    # Checking is multiple runs are (wrongly) attempted with sorted inputstream
    data_sorted = 'True' in dataset_name.split('_')[-1]
    if args.nof_runs > 1 and data_sorted:
        raise ValueError('Multiple runs of the experiment are being attempted on a sorted (string length-wise) input'
                         'stream of reber strings. This is absurd since there will be little to no variation in the'
                         'final results of all the multiple runs of the experiment if the input stream is constant!')

    logger.info('Running the model...')
    list_total_len_inputstream = []
    list_string_step_lookups = []
    list_df_results = []
    list_final_network = []
    for run in range(args.nof_runs):
        print('\n')
        logger.info('For Trial: {}'.format(run))
        random.shuffle(rg_inputoutput)
        results_ = run_experiment(data=rg_inputoutput,
                                  htm_network=htm_network,
                                  A_winner_cells=A_winner_cells,
                                  z_onehot=z_onehot,
                                  normalize_permanence=args.normalize_permanence,
                                  prune_dendrites=args.prune_dendrites,
                                  verbosity=args.verbosity_level)

        list_total_len_inputstream.append(results_['total_len_inputstream'])
        list_string_step_lookups.append(results_['string_step_lookup'])
        list_df_results.append(results_['df_results'])
        list_final_network.append(results_['final_network'])

    experiment_results = {
        'total_len_inputstreams': np.array(list_total_len_inputstream),
        'string_step_lookups': np.array(list_string_step_lookups),
        'df_results': np.array(list_df_results, dtype=object),  # this step converts the dataframes stored in
        # 'list_df_results' into numpy arrays. Remember to convert these arrays back into dataframes when loading
        # the saved .npz file back into memory.
        'final_networks': np.array(list_final_network, dtype=object),
        'grammar': np.array([grammar], dtype=object),
    }
    # Saving to disk
    if args.save_results:
        logger.info('Saving results to disk...')
        results_file_name = 'MODEL{}_DATASET{}_NORMALIZE{}_PRUNE{}_RUNS{}'.format(model_name,
                                                                                  dataset_name,
                                                                                  args.normalize_permanence,
                                                                                  args.prune_dendrites,
                                                                                  args.nof_runs)
        np.savez_compressed(file=os.path.join(results_save_path, results_file_name + '.npz'), **experiment_results)

        # dict_results['df_results'].to_hdf(path_or_buf=os.path.join(results_save_path, results_file_name + '.hdf5'),
        #                                   mode='w', key=results_file_name, complevel=7)
        # full_pickle(filename=os.path.join(results_save_path, results_file_name+'_network'),
        #             data=dict_results['final_network'])
        # full_pickle(filename=os.path.join(results_save_path, results_file_name + '_grammar'),
        #             data=grammar)
