import os
import argparse
import sys
import logging
import json
import numpy

from utils import settings

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
logger.addHandler(handler)

project_root = settings.get_project_root()

parser = argparse.ArgumentParser(description='Train an HTM model on generated Reber Grammar Strings')
parser.add_argument('-dc', '--default_config', dest='run_default_config', action='store', nargs='?', const=True,
                    default=True, help='')
parser.add_argument('-dd', '--default_dataset', dest='run_default_dataset', action='store', nargs='?', const=True,
                    default=True, help='')
parser.add_argument('-c', '--config', dest='config_json', action='store', default=None, help='')
parser.add_argument('-d', '--dataset', dest='dataset_npy', action='store', default=None, help='')
parser.add_argument('-v', '--verbosity', dest='verbosity_level', action='store', default=1, choices=[0, 1, 2, 3],
                    type=int, help='')

default_config = 'default_config.json'
default_dataset = 'reber_mix_5000strings.npy'

if __name__ == '__main__':

    args = parser.parse_args()

    # Setting up model configuration
    if args.config_json is not None:
        logger.info('Building HTM network with configurations from: {}'.format(args.config_json))
        with open(os.path.join(project_root, 'configs', args.config_json), 'r') as config:
            model_params = json.load(config)
    else:
        logger.info('Building HTM network with default configurations from: {}'.format(default_config))
        with open(os.path.join(project_root, 'configs', default_config), 'r') as config:
            model_params = json.load(config)

    # Setting up Reber strings dataset
    if args.dataset_npy is not None:
        logger.info('Using dataset: {}'.format(args.dataset_npy))
        with open(os.path.join(project_root, 'data', 'reber_strings_dataset', args.dataset_npy), 'r') as data:
            dataset = numpy.load(data)
    else:
        logger.info('Using default dataset: {}'.format(default_dataset))
        with open(os.path.join(project_root, 'data', 'reber_strings_dataset', default_dataset), 'r') as data:
            dataset = numpy.load(data)

    # Running the model
    model_params = {
        'num_chars': 7,
        'columns_per_char': 32,
        'num_columns': 32 * 7,
        'cells_per_column': 16,
        'max_dendrites_per_cell': 128,
        'max_synapses_per_dendrite': 49,  # (=cols_per_char + (nmda-1))
        'nmda_threshold': 18,
        'permanence_threshold': 0.50,
        'learning_threshold': 11,
        'permanence_init': 0.25,
        'permanence_init_sd': 0.002,
        'permanence_decrement': 0.1,
        'permanence_increment': 0.2,  # 2*perm_decrement
        'permanence_decay': 0.02,  # 0.2*perm_decrement
        'max_dendrite_dormancy': 8 * 30
    }
