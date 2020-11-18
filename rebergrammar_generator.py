'''
The following code allows us to generate random valid grammar strings:
(Source: https://www.deep-teaching.org/notebooks/sequence-learning/exercise-pytorch-char-rnn-reber-grammar)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


chars='ATSXPVZ'
emb_chars = 'TP'

graph = [[(1,5),('T','P')] , [(1,2),('S','X')], \
           [(3,5),('V','X')], [(6,),('Z')], \
           [(4,2,3),('T','P','S')], [(4,),('V')] ]


class Reber_Grammar():
    
    def __init__(self, numColumns, columnsPerChar):
        """
        Initializer function.
        
        Parameters
        ----------
        N : int
            DESCRIPTION.
        k : int
            DESCRIPTION.
        erg : BOOLEAN, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None.

        """
        
        self.chars = chars
        self.emb_chars = emb_chars
        self.graph = graph
        
        self.N = numColumns
        self.k = columnsPerChar
        
        self.df_CharsToMinicols = pd.DataFrame() # DataFrame with cols: 'A', 'T', 'P', etc. each with 'k'
                                                 # minicolumn indices corresponding to that character.
        minicolumns = np.arange(self.N)
        random.seed(1)
        random.shuffle(minicolumns)
        
        for i, sym in enumerate(self.chars):
            mc = minicolumns[i*self.k:(i+1)*self.k]
            self.df_CharsToMinicols[sym] = np.sort(mc, kind='mergesort')
        
        return None
        
        
    
    def generateSequences(self, minLength=5):
        """
        Function to generate one SRG string (alphabet form) and 
        list of possible next transitions.        
        
        Parameters
        ----------
        minLength : int, optional
            Minimum length of the SRG string. The default is 5.

        Returns
        -------
        inchars : list of str; an array of chars generated from SRG. Ex: ['A', 'P', 'V', 'T', 'S', 'Z']
        outchars : list of tuples of str; an array of next possible transitions from each char in the first entry. 
        Ex: [('T', 'P'), 'V', ('T', 'P', 'S'), ('T', 'P', 'S'), 'Z']

        """
        
        while True:
            inchars = ['A']
            node = 0
            outchars = []    
            
            while node != 6:
                transitions = graph[node]
                i = np.random.randint(0, len(transitions[0]))
                inchars.append(transitions[1][i])
                outchars.append(transitions[1])
                node = transitions[0][i]
            
            if len(inchars) > minLength:  
                return inchars, outchars
            
            
    def get_one_srg(self, minLength=5):
        """
        To generate one SRG string in one-hot vector form (with respect to the minicolumns)
        and its corresponding list of possible next transitions.
        
        Parameters
        ----------
        minLength : int, optional
            Minimum length of the SRG string. The default is 5.

        Returns
        -------
        inseq_oh : list of binary ndarray; (excluding the one-hot for 'Z' at the end)
        outseq_oh : list of binary ndarray; (including the one-hot for 'Z' at the end 
                                             for penultimate prediction)

        """
        
        inchars, outchars = self.generateSequences(minLength)
        
        
        inseq_oh = []
        outseq_oh = []
        
        for i_ch,o_ch in zip(inchars, outchars): 
            
            inpt = np.zeros(self.N, dtype=np.int8)
            inpt[self.df_CharsToMinicols[i_ch]] = 1     
            inseq_oh.append(inpt)
            
            outpt = np.zeros(self.N, dtype=np.int8)
            for o in o_ch:
                outpt[self.df_CharsToMinicols[o]] = 1
            outseq_oh.append(outpt)
        
        return np.array(inseq_oh, dtype=np.int8), np.array(outseq_oh, dtype=np.int8)
    
    
    def get_n_srg(self, n, minLength=5):
        """
        Generate 'n' SRG string examples in one-hot encoding (wrt minicolumns).

        Parameters
        ----------
        n : int; No. of SRG examples to generate.
        minLength : int, optional
            Minimum length of the SRG string. The default is 5.

        Returns
        -------
        examples : list; list of 'n' examples of SRG strings, along with list
        of possible next transitions. (i.e. list of 'n' get_one_srg() outputs)
        
        NOTE
        ----
        get_n_srg(5)[0]: gives the first <get_one_srg() output>.
        get_n_srg(5)[0][0]: gives the input string's one-hot encoding of the first <get_n_srg() output>.
        get_n_srg(5)[2][1]: gives the prediction string's one-hot encoding of the third <get_n_srg() output>.
        get_n_srg(5)[2][0][0]: gives the one-hot encoding of 'A' of third <get_n_srg() output>.
        get_n_srg(5)[2][1][0]: gives the one-hot encoding of predictions of 'T' and 'P' after 'A' for the 
                               third <get_n_srg() output>.
        
        """
        examples = []
        
        for i in range(n):
            examples.append(self.get_one_srg(minLength))
        
        return examples
    
    
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
            index = np.where(s==1)[0][0]
            reberString += self.df_CharsToMinicols.columns[np.where(self.df_CharsToMinicols==index)[1][0]]
        reberString+='Z'
        
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
        charOnehot[self.df_CharsToMinicols[char]] = 1
        
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
            transitions = graph[node]
            try:
                node = transitions[0][transitions[1].index(c)]
            except ValueError: # using exceptions for flow control in python is common
                return False
        
        return True  

    
    def histogram_lenReberStrings(self, rg_exs):
        """
        Plots histogram of the lengths of the generated reber grammar strings.

        Parameters
        ----------
        rg_exs : list of reber grammar strings.

        Returns
        -------
        None.

        """
        
        len_rg_exs = []

        for ex in rg_exs:
            len_rg_exs.append(len(ex[0])+1)
    
        plt.figure()
        plt.hist(len_rg_exs)
        plt.show()
        
        return
    
# ______________________________For ERG strings________________________________
    
    def get_one_erg(self, minLength=5):
        """
        To generate one ERG string in one-hot vector form (with respect to the minicolumns)
        and its corresponding list of possible next transitions.

        Parameters
        ----------
        minLength : int, optional
            Minimum length of the ERG string. The default is 5.

        Returns
        -------
        emb_in_oh : list of binary ndarray; (excluding the one-hot for 'Z' at the end).
        emb_out_oh : list of binary ndarray; (including the one-hot for 'Z' at the end 
                                             for penultimate prediction)

        """
        
        simple_in_oh, simple_out_oh = self.get_one_srg()
        
        emb_in_oh = simple_in_oh[:] # contains the one-hot encoded SRG, except the final 'Z'
        emb_out_oh = simple_out_oh[:]
        
        # Selecting one of 'T' or 'P' for embedding and computing its one-hot encoding
        emb_char = emb_chars[np.random.randint(0, len(emb_chars))]
        emb_char_oh = np.zeros(self.N, dtype=np.int8)
        emb_char_oh[self.df_CharsToMinicols[emb_char]] = 1     
    
        # Entering the embedded char's one-hot encoding at the second position    
        emb_in_oh[1:1] = [emb_char_oh] 
        # Entering the embedded char's one-hot encoding at the last position
        emb_in_oh.insert(len(emb_in_oh), emb_char_oh) 
        #### print('Embedded INPUT string:', OnehotToWord(emb_in))
        
        # Entering the predictions of 'T' and 'P' at the second position
        emb_out_oh[1:1] = [simple_out_oh[0]]
        # Entering the prediction of embedded char at the second last position
        emb_out_oh.insert(len(emb_out_oh)-1, emb_char_oh)
    
        return np.array(emb_in_oh, dtype=np.int8), np.array(emb_out_oh, dtype=np.int8)
    
    
    def get_n_erg(self, n, minLength=5):
        """
        Generate 'n' ERG string examples in one-hot encoding (wrt minicolumns).

        Parameters
        ----------
        n : int; No. of SRG examples to generate.
        minLength : int, optional
            Minimum length of the SRG string. The default is 5.

        Returns
        -------
        examples : list; list of 'n' examples of ERG strings, along with list
        of possible next transitions.
        
        NOTE
        ----
        Similar to the note for get_n_srg() above.

        """
        examples = []
        
        for i in range(n):
            examples.append(self.get_one_erg(minLength))
        
        return examples
    
    
    