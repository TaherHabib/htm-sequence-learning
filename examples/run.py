import os
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from htm_sequence_learning.reber_grammar.utils import get_graph_from_dataset
from htm_sequence_learning.htm.utils import get_A_winner_cells
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

parser = argparse.ArgumentParser(description='Train an HTM model on generated Reber Grammar Strings')
parser.add_argument('-dc', '--default_config', dest='run_default_config', action='store', nargs='?', const=True,
                    default=True, help='')
parser.add_argument('-dd', '--default_dataset', dest='run_default_dataset', action='store', nargs='?', const=True,
                    default=True, help='')
parser.add_argument('-c', '--config', dest='config_json', action='store', default=None, type=str,
                    help='')
parser.add_argument('-d', '--dataset', dest='dataset_npy', action='store', default=None, type=str,
                    help='')
parser.add_argument('-v', '--verbosity', dest='verbosity_level', action='store', default=1, choices=[0, 1, 2, 3],
                    type=int, help='')

default_config = 'default_config.json'
default_dataset = 'graph1_numStrings2000_ergFalse.npy'

if __name__ == '__main__':

    args = parser.parse_args()

    # Setting up model configuration
    if args.config_json is not None:
        logger.info('Building HTM network with configurations from: {}'.format(args.config_json))
        with open(os.path.join(model_config_path, args.config_json.replace('.json', '') + '.json'), 'r') as config:
            model_params = json.load(config)
    else:
        logger.info('Building HTM network with default configurations from: {}'.format(default_config))
        with open(os.path.join(model_config_path, default_config), 'r') as config:
            model_params = json.load(config)

    # Setting up Reber strings dataset
    if args.dataset_npy is not None:
        logger.info('Using dataset: {}'.format(args.dataset_npy))
        with open(os.path.join(datastream_path, args.dataset_npy.replace('.npy', '') + '.npy'), 'r') as data:
            dataset = np.load(data)
        graph_idx = get_graph_from_dataset(args.dataset_npy)
    else:
        logger.info('Using default dataset: {}'.format(default_dataset))
        with open(os.path.join(datastream_path, default_dataset), 'r') as data:
            dataset = np.load(data)
        graph_idx = get_graph_from_dataset(default_dataset)

    # Defining Grammar object for later downstream processing tasks
    grammar = Reber_Grammar(columns_per_char=model_params['columns_per_char'], graph_idx=graph_idx)
    # Get 'A' winner cells
    A_winner_cells = get_A_winner_cells(rg=grammar, M=model_params['cells_per_column'])
    # Get Onehot for 'Z'
    z_onehot = grammar.CharToOnehot('Z')

    # Initializing HTM Network
    htm_network = HTM_NET.from_json(model_params=model_params, verbosity=args.verbosity_level)

    # Running the model
    # TODO: call the experimentor module here to run the experiment