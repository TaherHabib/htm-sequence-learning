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
if not os.path.exists(data_save_path):
    os.makedirs(data_save_path)

parser = argparse.ArgumentParser(description='Generator for Reber Grammar Strings')
parser.add_argument('graph_idx', action='store', nargs='?', default=2, type=int,
                    help='Index of the transition graph to use to generate Reber strings for the input stream. See '
                         'htm_sequence_learning/reber_grammar/graph_configs.py on details of different available '
                         'transition graphs. While this is a position argument, it is still optional to specify it, in'
                         'which case the standard transition graph (graph_idx = 2) is used as a default.')
parser.add_argument('reber_strings', action='store', type=str,
                    help='Specify whether to generate a mix set of Reber Strings (\'mix\') or generate input stream '
                         'using pre-defined input-output Reber Strings (a .json file). For format of this JSON file, '
                         'see configs/rebergrammar directory.')
parser.add_argument('num_strings', action='store', type=int, help='Total number of reber strings to generate.')
parser.add_argument('save_to_disk', action='store', nargs='?', default=True,
                    help='Whether to save the generated Reber Strings input stream. The file is saved in '
                         'data/reber_strings_dataset directory. While this is a position argument, it is still optional'
                         ' to specify it, in which case default is True.')
parser.add_argument('-cc', '--columns_per_char', dest='columns_per_char', action='store', default=32, type=int,
                    help='Number of (HTM network) columns to use per character (APVSXTZ). Default value is at 32. '
                         'This, along with \'graph_idx\', is an input to the Reber_Grammar() class object.')
parser.add_argument('-m', '--max_length', dest='max_string_length', action='store', default=1000, type=int,
                    help='Maximum allowable length of a Reber String in the input stream. Default is set at 1000.')
parser.add_argument('-e', '--do_erg', dest='do_erg', action='store', nargs='?', const=True, default=False,
                    help='Whether to generate Embedded Reber Strings or not. Specifying only the short flag without '
                    'any argument leads to a value of True.')
parser.add_argument('-s', '--sort', dest='sort_by_length', action='store', nargs='?', const=True, default=False,
                    help='Whether to sort the Reber Strings with length or not. Specifying only the short flag without '
                         'any argument leads to a value of True.')

if __name__ == '__main__':

    args = parser.parse_args()

    # Setting up Reber Grammar Generator configuration
    logger.info('Setting up Reber_Grammar() object.')
    rg = Reber_Grammar(columns_per_char=args.columns_per_char, graph_idx=args.graph_idx)

    if args.reber_strings == 'mix':
        reber_strings = args.reber_strings
    elif '.json' in args.reber_strings:
        with open(os.path.join(ROOT, 'configs', 'rebergrammar', args.reber_strings), 'r') as strings:
            reber_strings = json.load(strings)
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
        if args.reber_strings == 'mix':
            file_name = 'graph{}_{}_numStrings{}_erg{}_sort{}'.format(args.graph_idx,
                                                                      args.reber_strings,
                                                                      args.num_strings,
                                                                      args.do_erg,
                                                                      args.sort_by_length)
            np.save(arr=rg_inputoutput, file=os.path.join(data_save_path, file_name + '.npy'))
        else:
            file_name = 'graph{}_{}_numStrings{}_erg{}_sort{}'.format(args.graph_idx,
                                                                      args.reber_strings.replace('.json', ''),
                                                                      args.num_strings,
                                                                      args.do_erg,
                                                                      args.sort_by_length)
            np.save(arr=rg_inputoutput, file=os.path.join(data_save_path, file_name + '.npy'))
        logger.info('Generated Reber Strings inputs-outputs saved successfully!')
