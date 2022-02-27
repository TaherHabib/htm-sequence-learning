"""
MODULE DESCRIPTION:
-------------------

This module is used to generate Simple Reber Grammar strings, as inputs to the HTM network
alongwith the corresponding output strings to evaluate the network's performance.

- The interface is defined using the Reber_Grammar() class in ReberGrammar.py.
- Five different (Markovian) transition graphs are available to choose from.

"""

import os
import argparse
import logging
import json
from pathlib import Path
import numpy as np
from htm_sequence_learning.reber_grammar.ReberGrammar import Reber_Grammar

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
logger.addHandler(handler)

ROOT = os.path.abspath(Path(__file__).parent.parent)
data_save_path = os.path.join(ROOT, 'data', 'reber_strings_dataset')

parser = argparse.ArgumentParser(description='Generator for Reber Grammar Strings')
parser.add_argument('graph_idx', action='store', nargs='?', default=1, type=int, help='')
parser.add_argument('reber_strings', action='store', type=str, help='')
parser.add_argument('num_strings', action='store', type=int, help='')
parser.add_argument('save_to_disk', action='store', nargs='?', default=True, help='')
parser.add_argument('-cc', '--columns_per_char', dest='columns_per_char', action='store', default=32, type=int,
                    help='')
parser.add_argument('-m', '--max_length', dest='max_string_length', action='store', default=1000, type=int,
                    help='')
parser.add_argument('-e', '--do_erg', dest='do_erg', action='store', nargs='?', const=True, default=False,
                    help='')
parser.add_argument('-s', '--sort', dest='sort_by_length', action='store', nargs='?', const=True, default=False,
                    help='')

if __name__ == '__main__':

    args = parser.parse_args()

    # Setting up Reber Grammar Generator configuration
    logger.info('Setting up Reber_Grammar() object.')
    rg = Reber_Grammar(columns_per_char=args.columns_per_char, graph_idx=args.graph_idx)

    if args.reber_strings == 'mix':
        reber_strings = args.reber_strings
        if not os.path.exists(data_save_path):
            os.mkdir(data_save_path)
    elif '.json' in args.reber_strings:
        with open(os.path.join(ROOT, 'configs', 'rebergrammar', args.reber_strings), 'r') as strings:
            reber_strings = json.load(strings)
        if not os.path.exists(data_save_path):
            os.mkdir(data_save_path)
    else:
        raise ValueError('Please provide a valid file format (.json only) containing a dictionary with input-output'
                         'reber strings to be generated for the input data stream. OR, simply provide the string '
                         '\'mix\' to generate a random mixed bag set of reber strings')

    logger.info('Generating Input Strings and Predictions...')
    rg_inputoutput = rg.generate_InputOutput(do_erg=args.do_erg,
                                             num_strings=args.num_strings,
                                             reber_strings=reber_strings,
                                             max_string_length=args.max_string_length,
                                             sort=args.sort_by_length)
    # Saving to disk
    if args.save_to_disk:
        logger.info('Saving to disk...')
        if args.reber_strings == 'mix_reber_strings':
            file_name = 'graph{}_numStrings{}_erg{}'.format(args.graph_idx, args.num_strings, args.do_erg)
            np.save(arr=rg_inputoutput, file=os.path.join(data_save_path, file_name + '.npy'))
        else:
            file_name = 'graph{}_{}_numStrings{}_erg{}'.format(args.graph_idx, args.reber_strings.replace('.json', ''),
                                                               args.num_strings, args.do_erg)
            np.save(arr=rg_inputoutput, file=os.path.join(data_save_path, file_name + '.npy'))
        logger.info('Generated Reber Strings inputs-outputs saved successfully!')
