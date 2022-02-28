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
data_path = os.path.join(ROOT, 'data', 'reber_strings_dataset')
fig_path = os.path.join(ROOT, 'figures', 'stats')
if not os.path.exists(fig_path):
    os.mkdir(fig_path)


parser = argparse.ArgumentParser(description='Simple Analysis of the input stream containing Reber Grammar Strings')
parser.add_argument('rg_iostream', action='store', default=None, type=str, help='')
parser.add_argument('-p', '--print_all', dest='print_all', action='store', nargs='?', const=True, default=False,
                    help='')
parser.add_argument('-s', '--savefigs', dest='save_figures', action='store', nargs='?', const=True, default=False,
                    help='')


def compute_input_stats(rg_iostream=None, print_all=False, save_figures=False):

    # N = len(rg_inputoutput[0][1][0])  # Taking the first reber string's ([0]) one-hot encodings ([1]) for the
    # # first letter 'A' ([0]) and finding its length gives us the value of N.

    graph_idx = get_graph_from_dataset(rg_iostream)

    logger.info('Reading input stream file...')
    with open(os.path.join(data_path, rg_iostream.replace('.npy', '') + '.npy'), 'rb') as stream:
        rg_inputoutput = np.load(stream, allow_pickle=True)
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

    logger.info('Plotting distribution of lengths of sample reber strings in the inputstream...')
    y, binEdges = np.histogram(len_reber_strings, bins=np.unique(len_reber_strings))
    plt.figure(figsize=(15, 8))
    plt.bar(binEdges[:-1], y, width=1, color='maroon')
    plt.errorbar(binEdges[:-1], y, yerr=np.sqrt(y), fmt='o', color='Black', elinewidth=3, capthick=2, alpha=0.7,
                 markersize=5, capsize=5)
    plt.xlabel('Length of Reber String', fontsize=18)
    plt.ylabel('Number of Occurrences \n in Inputstream', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.text(x=binEdges[-1] / 2, y=0.8 * y[0], s=f'Mean Length of Reber String: {np.mean(len_reber_strings)}',
             fontsize=20)
    plt.text(x=binEdges[-1] / 2, y=0.7 * y[0], s=f'Median Length of Reber String: {np.median(len_reber_strings)}',
             fontsize=20)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    if save_figures:
        fig_name = 'String_length_dist_{}'.format(rg_iostream.replace('.npy', ''))
        plt.savefig(fname=os.path.join(fig_path, fig_name), format='svg')
        logger.info('Figure saved in svg format at {}.svg.'.format(os.path.join(fig_path, fig_name)))
    plt.show()
    plt.close()

    logger.info('Plotting distribution of possible transitions (trigrams) in reber strings in the inputstream...')
    transitions = list(dict_count_allTransitions.keys())
    counts = list(dict_count_allTransitions.values())

    plt.figure(figsize=(15, 8))
    plt.bar(transitions, counts, color='maroon', width=0.5)
    plt.xlabel('Possible Transitions in Simple Reber Grammar', fontsize=18)
    plt.ylabel('Number of Occurrences in Input Stream', fontsize=18)
    plt.xticks(rotation=50, fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(True, linestyle="--", color='black', alpha=0.4)
    if save_figures:
        fig_name = 'Transitions_dist_{}'.format(rg_iostream.replace('.npy', ''))
        plt.savefig(fname=os.path.join(fig_path, fig_name), format='svg')
        logger.info('Figure saved in svg format at {}.svg.'.format(os.path.join(fig_path, fig_name)))
    plt.show()
    plt.close()

    if print_all:
        for i, string in enumerate(in_reber_strings):
            print(i, string)


if __name__ == '__main__':
    args = parser.parse_args()

    compute_input_stats(rg_iostream=args.rg_iostream,
                        print_all=args.print_all,
                        save_figures=args.save_figures)
