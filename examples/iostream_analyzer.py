import json
import os
import argparse
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from htm_sequence_learning.reber_grammar.utils import get_graph_from_dataset, count_allTransitions

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s : %(levelname)s :- %(message)s'))
logger.addHandler(handler)

ROOT = os.path.abspath(Path(__file__).parent.parent)
data_path = os.path.join(ROOT, 'data', '../data/reber_strings_dataset')

parser = argparse.ArgumentParser(description='Simple Analysis of the input stream containing Reber Grammar Strings')
parser.add_argument('rg_iostream', action='store', default=None, type=str, help='')
parser.add_argument('-p', '--print_all', dest='print_all', action='store', nargs='?', const=True, default=False,
                    help='')
parser.add_argument('-s', '--savefigs', dest='save_figures', action='store', nargs='?', const=True, default=False,
                    help='')


def compute_input_stats(graph_idx=None, rg_inputoutput=None, print_all=False, save_figures=False):
    # N = len(rg_inputoutput[0][1][0])  # Taking the first reber string's ([0]) one-hot encodings ([1]) for the
    # # first letter 'A' ([0]) and finding its length gives us the value of N.

    in_reber_strings = [rg_inputoutput[i][0] for i in range(len(rg_inputoutput))]
    dict_count_allTransitions = count_allTransitions(graph_idx=graph_idx,
                                                     in_reber_strings=in_reber_strings)
    # NOF TOTAL CHARACTERS in the input stream
    total_len_inputstream = 0
    for string in in_reber_strings:
        total_len_inputstream += len(string)

    len_reber_strings = []
    for ex in in_reber_strings:
        len_reber_strings.append(len(ex))

    logger.info('The stream consists of a total of {} strings. \n\
    With: \n\
    Number of characters in total = {}. \n\
    Mean length of string = {}. \n\
    Median length of string = {}. \n\
    \n Unique Strings = {}. \n\
    Number of Unique Strings = {}.'.format(len(in_reber_strings),
                                           total_len_inputstream,
                                           np.mean(len_reber_strings),
                                           np.median(len_reber_strings),
                                           np.unique(in_reber_strings),
                                           len(np.unique(in_reber_strings))
                                           )
                )

    if print_all:
        for i, string in enumerate(in_reber_strings):
            print(i, string)

    # Plotting distribution of lengths of sample reber strings in the inputstream
    plt.figure()
    plt.hist(len_reber_strings)
    plt.show()

    # Plotting distribution of possible transitions (trigrams) in sample reber strings in the inputstream


if __name__ == '__main__':
    args = parser.parse_args()
    graph_idx = get_graph_from_dataset(args.rg_iostream)

    with open(os.path.join(data_path, args.rg_iostream), 'r') as dataset:
        rg_inputoutput = json.load(dataset)

    compute_input_stats(graph_idx=graph_idx,
                        rg_inputoutput=rg_inputoutput,
                        print_all=args.print_all,
                        save_figures=args.save_figures)
