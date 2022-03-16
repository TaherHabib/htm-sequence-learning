"""
MODULE DESCRIPTION:
-------------------

This module is used to define Reber Grammar class, which makes the finite state graph for generation of
reber strings (as inputs to the HTM network) and also defines various methods required for/during generation
of the network's input stream.

- Five different (markovian) transition graphs are available to choose from.

"""

import numpy as np
import pandas as pd
import random

from .graph_configs import dict_reber_graphs, list_graphs, chars, embedded_chars
from .utils import reberstring_length, get_graph_allPossibleTransitions
from htm_sequence_learning.htm.utils import get_num_columns


class Reber_Grammar:

    def __init__(self,
                 columns_per_char=None,
                 graph_idx=1):
        """

        Parameters
        ----------
        columns_per_char
        graph_idx

        """

        self.chars = chars
        self.embedded_chars = embedded_chars
        self.graph = dict_reber_graphs[list_graphs[graph_idx-1]]
        self.k = columns_per_char
        self.N = get_num_columns(columns_per_char, chars)
        self.df_CharsToMinicols = self.get_charsToMinicols  # DataFrame with cols: 'A', 'T', 'P', etc. each with 'k'
        self.axis_MinicolsToChars = self.get_axis
        # minicolumn indices corresponding to that character.
        self.all_possibleTransitions = get_graph_allPossibleTransitions(self.graph)

    @property
    def get_charsToMinicols(self):

        df_CharsToMinicols = pd.DataFrame()
        minicolumns = np.arange(self.N)
        random.seed(1)
        random.shuffle(minicolumns)
        for i, sym in enumerate(chars):
            mc = minicolumns[i * self.k : (i + 1) * self.k]
            df_CharsToMinicols[sym] = np.sort(mc, kind='mergesort')

        return df_CharsToMinicols

    @property
    def get_axis(self):
        net_axis = np.empty(self.N, dtype=object)
        for ch in chars:
            net_axis[self.df_CharsToMinicols[ch]] = ch
        return net_axis

    def generateSequences(self,
                          min_length=5,
                          max_length=1000):
        """
        Function to generate one SRG string (alphabet form) and
        list of possible next transitions.

        Parameters
        ----------
        min_length : int, optional
            Minimum length of the SRG string. The default is 5.
        max_length : int, optional
            Maximum length of the SRG string. The default is 1000.

        Returns
        -------
        inchars : list of str; an array of chars generated from SRG. Ex: ['A', 'P', 'V', 'T', 'S', 'Z']
        outchars : list of tuples of str; an array of next possible transitions from each char in the first entry.
        Ex: [('T', 'P'), 'V', ('T', 'P', 'S'), ('T', 'P', 'S'), 'Z'] (say, from graph_1)

        """

        while True:
            inchars = ['A']
            node = 0
            outchars = []

            while node != 6:
                transitions = self.graph[node]
                i = np.random.randint(0, len(transitions[1]))
                inchars.append(transitions[1][i])
                outchars.append(transitions[1])
                if len(transitions[1]) == 1:
                    node = transitions[0]
                else:
                    node = transitions[0][i]

            if min_length <= len(inchars) <= max_length:
                return inchars, outchars
            else:
                continue

    def get_n_srg(self, n=1, min_length=5, max_length=1000):
        """
        Generate 'n' SRG string examples in one-hot encoding (wrt minicolumns).

        Parameters
        ----------
        n : int;
            No. of SRG examples to generate. The default is 1.
        min_length : int, optional
            Minimum length of the SRG string. The default is 5.
        max_length : int, optional
            Maximum length of the SRG string. The default is 1000.

        Returns
        -------
        srgIO : list; list of 'n' examples of SRG strings.
            Each entry in the list is a 3-tuple with:
                first_index: Input Simple Reber String.
                second_index: one-hot encoding of the simple reber string characters (wrt minicolumns)
                third_index: one-hot encoding of possible next transitions.

        NOTE
        ----
        get_n_srg(5)[0]: gives the first <get_one_srg() output>.
        get_n_srg(5)[0][0]: gives the first input Reber String  from the <get_n_srg() output>.
        get_n_srg(5)[0][1]: gives the input string's one-hot encoding of the first <get_n_srg() output>.
        get_n_srg(5)[2][2]: gives the prediction string's one-hot encoding of the third <get_n_srg() output>.
        get_n_srg(5)[2][1][0]: gives the one-hot encoding of 'A' of third <get_n_srg() output>.
        get_n_srg(5)[2][2][0]: gives the one-hot encoding of predictions of 'T' and 'P' after 'A' for the
                               third <get_n_srg() output>.

        """
        srgIO = []
        for i in range(n):

            inseq_chars, outseq_chars = self.generateSequences(min_length, max_length)
            inseq_onehot = []
            outseq_onehot = []
            for i_ch, o_ch in zip(inseq_chars, outseq_chars):
                inpt = np.zeros(self.N, dtype=np.int8)
                inpt[self.df_CharsToMinicols[i_ch]] = 1
                inseq_onehot.append(inpt)
                outpt = np.zeros(self.N, dtype=np.int8)
                for o in o_ch:
                    outpt[self.df_CharsToMinicols[o]] = 1
                outseq_onehot.append(outpt)

            srgIO.append((self.OnehotToWord(inseq_onehot),
                          np.array(inseq_onehot, dtype=np.int8),
                          np.array(outseq_onehot, dtype=np.int8)))
        return srgIO

    def get_n_erg(self, n=1, min_length=5, max_length=1000):
        """
        Generate 'n' ERG string examples in one-hot encoding (wrt minicolumns).

        Parameters
        ----------
        n : int; No. of SRG examples to generate. The default is 1.
        min_length : int, optional
            Minimum length of the SRG string. The default is 5.
        max_length : int, optional
            Maximum length of the SRG string. The default is 1000.

        Returns
        -------
        ergIO : list; list of 'n' examples of ERG strings.
            Each entry in the list is a 3-tuple with:
                first_index: Input Embedded Reber String.
                second_index: one-hot encoding of the embedded reber string characters (wrt minicolumns)
                third_index: one-hot encoding of possible next transitions.

        NOTE
        ----
        Similar to the note for get_n_srg() above.

        """
        ergIO = []

        for i in range(n):
            one_srg = self.get_n_srg(min_length=min_length, max_length=max_length)
            simple_in_onehot = one_srg[0][1]
            simple_out_onehot = one_srg[0][2]

            emb_in_onehot = list(simple_in_onehot[:])  # contains the one-hot encoded SRG, except the final 'Z'
            emb_out_onehot = list(simple_out_onehot[:])

            # Selecting one of 'T' or 'P' for embedding and computing its one-hot encoding
            embedded_char = embedded_chars[np.random.randint(0, len(embedded_chars))]
            emb_char_onehot = np.zeros(self.N, dtype=np.int8)
            emb_char_onehot[self.df_CharsToMinicols[embedded_char]] = 1

            # Entering the embedded char's one-hot encoding at the second position
            emb_in_onehot[1:1] = [emb_char_onehot]
            # Entering the embedded char's one-hot encoding at the last position
            emb_in_onehot.insert(len(emb_in_onehot), emb_char_onehot)

            # Entering the predictions of 'T' and 'P' at the second position
            emb_out_onehot[1:1] = [simple_out_onehot[0]]
            # Entering the prediction of embedded char at the second last position
            emb_out_onehot.insert(len(emb_out_onehot) - 1, emb_char_onehot)

            ergIO.append((self.OnehotToWord(emb_in_onehot),
                          np.array(emb_in_onehot, dtype=np.int8),
                          np.array(emb_out_onehot, dtype=np.int8)))

        return ergIO

    def generate_InputOutput(self, do_erg, num_strings, reber_strings,
                             max_string_length=1000, sort=False):

        if reber_strings != 'mix':
            rg_inputoutput = []
            for in_string, out_ in reber_strings.items():
                inseq_onehot = []
                outseq_onehot = []
                for in_ch, out_ch in zip(in_string, out_):
                    inpt = np.zeros(self.N, dtype=np.int8)
                    inpt[self.df_CharsToMinicols[in_ch]] = 1
                    inseq_onehot.append(inpt)

                    outpt = np.zeros(self.N, dtype=np.int8)
                    for o in out_ch:
                        outpt[self.df_CharsToMinicols[o]] = 1
                    outseq_onehot.append(outpt)

                for rep in range(int(np.floor(num_strings / len(reber_strings)))):
                    rg_inputoutput.append((in_string,
                                           np.array(inseq_onehot, dtype=np.int8),
                                           np.array(outseq_onehot, dtype=np.int8)))
            random.shuffle(rg_inputoutput)
            return np.array(rg_inputoutput, dtype=object)

        else:
            if do_erg:
                if sort:
                    rg_inputoutput = sorted(self.get_n_erg(n=num_strings, max_length=max_string_length),
                                            key=reberstring_length)
                else:
                    rg_inputoutput = self.get_n_erg(n=num_strings, max_length=max_string_length)
            else:
                if sort:
                    rg_inputoutput = sorted(self.get_n_srg(n=num_strings, max_length=max_string_length),
                                            key=reberstring_length)
                else:
                    rg_inputoutput = self.get_n_srg(n=num_strings, max_length=max_string_length)
            return np.array(rg_inputoutput, dtype=object)

    def OnehotToWord(self, sequence):
        """
        converts a sequence (one-hot) back to a RG string.

        Parameters
        ----------
        sequence : list of binary ndarray; Given Reber String in one-hot encoding.

        Returns
        -------
        reberString : list of str; Reber string in alphabet. Ex. 'APVPVZ'

        """

        reberString = ''
        for s in sequence:
            index = np.where(s == 1)[0][0]
            reberString += self.df_CharsToMinicols.columns[np.where(self.df_CharsToMinicols == index)[1][0]]
        reberString += 'Z'
        return reberString

    def CharToOnehot(self, char):
        """
        Returns Character's onehot encoding.

        Parameters
        ----------
        char : str (character)

        Returns
        -------
        charOnehot : onehot encoding of 'char' (in terms of the N minicolumns)
        """

        charOnehot = np.zeros(self.N, dtype=np.int8)
        charOnehot[self.df_CharsToMinicols[char.upper()]] = 1
        return charOnehot

    def in_grammar(self, word):
        """
        To check if a given character String is in the Reber Grammar.

        Parameters
        ----------
        word : string;

        Returns
        -------
        bool; True, if the string is a valid string of RG.

        """

        if word[0] != 'A':
            return False
        node = 0
        for c in word[1:]:
            transitions = self.graph[node]
            try:
                node = transitions[0][transitions[1].index(c)]
            except ValueError:
                return False
        return True

