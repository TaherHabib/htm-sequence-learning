'''
The following code allows us to generate random valid grammar strings:
(Source: https://www.deep-teaching.org/notebooks/sequence-learning/exercise-pytorch-char-rnn-reber-grammar)
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from ufuncs import reberstring_length, get_graph_allPossibleTransitions


chars='ATSXPVZ'
embedded_chars = 'TP'


graph_1 = [[(1,5),('T','P')] , [(1,2),('S','X')], \
           [(3,5),('V','X')], [(6),('Z')], \
           [(4,2,3),('T','P','S')], [(4),('V')]]
    
graph_2 = [[(1,5),('T','P')] , [(1,2),('S','P')], \
           [(3,5),('V','X')], [(6),('Z')], \
           [(4,3),('T','X')], [(1,4),('V','S')]] # the one used as a standard. Doesn't have a 
                                                    # uniform distribution of trigrams.

graph_3 = [[(1,5),('T','P')] , [(1,2),('S','P')], \
           [(3,5),('V','X')], [(6),('Z')], \
           [(4,3),('T','X')], [(2,4),('V','S')]]
    
graph_4 = [[(1,5),('T','P')] , [(4,2),('X','P')], \
           [(3,2),('V','S')], [(6),('Z')], \
           [(5,3),('T','X')], [(1,4),('V','S')]]
    
graph_5 = [[(1,5),('T','P')] , [(4,2),('X','P')], \
           [(3,1),('V','S')], [(6),('Z')], \
           [(5,3),('T','X')], [(2,4),('V','S')]] # symmetric one with uniform distribution 
                                                    # of trigrams.


    
class Reber_Grammar():
    
    def __init__(self, numColumns, columnsPerChar, graph='1'):
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
        self.embedded_chars = embedded_chars
        
        if graph=='1':
            self.graph = graph_1
        elif graph=='2':
            self.graph = graph_2
        elif graph=='3':
            self.graph = graph_3
        elif graph=='4':
            self.graph = graph_4
        else:
            self.graph = graph_5
        
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
            
        self.all_possibleTransitions = get_graph_allPossibleTransitions(self.graph)
        
        return
        
    
    def generateSequences(self, minLength=5, maxLength=None):
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
        Ex: [('T', 'P'), 'V', ('T', 'P', 'S'), ('T', 'P', 'S'), 'Z'] (from graph_1)

        """
        
        if maxLength is None:
            maxL = 1000
        else:
            maxL = maxLength
            
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
                
            if len(inchars) >= minLength and len(inchars) <= maxL:
                return inchars, outchars
            else:
                continue
        
    
    def get_n_srg(self, n=1, minLength=5, maxLength=None):
        """
        Generate 'n' SRG string examples in one-hot encoding (wrt minicolumns).

        Parameters
        ----------
        n : int; No. of SRG examples to generate. The default is 1.
        minLength : int, optional
            Minimum length of the SRG string. The default is 5.

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
            
            inseq_chars, outseq_chars = self.generateSequences(minLength, maxLength)
        
            inseq_onehot = []
            outseq_onehot = []
            
            for i_ch,o_ch in zip(inseq_chars, outseq_chars): 
                
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
    
# ______________________________For ERG strings________________________________
    
    def get_n_erg(self, n=1, minLength=5, maxLength=None):
        """
        Generate 'n' ERG string examples in one-hot encoding (wrt minicolumns).

        Parameters
        ----------
        n : int; No. of SRG examples to generate. The default is 1.
        minLength : int, optional
            Minimum length of the SRG string. The default is 5.

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
            
            one_srg = self.get_n_srg(minLength=minLength, maxLength=maxLength)
            simple_in_onehot = one_srg[0][1] 
            simple_out_onehot = one_srg[0][2]
        
            emb_in_onehot = list(simple_in_onehot[:]) # contains the one-hot encoded SRG, except the final 'Z'
            emb_out_onehot = list(simple_out_onehot[:])
            
            # Selecting one of 'T' or 'P' for embedding and computing its one-hot encoding
            embedded_char = embedded_chars[np.random.randint(0, len(embedded_chars))]
            emb_char_onehot = np.zeros(self.N, dtype=np.int8)
            emb_char_onehot[self.df_CharsToMinicols[embedded_char]] = 1     
        
            # Entering the embedded char's one-hot encoding at the second position    
            emb_in_onehot[1:1] = [emb_char_onehot] 
            # Entering the embedded char's one-hot encoding at the last position
            emb_in_onehot.insert(len(emb_in_onehot), emb_char_onehot) 
            #### print('Embedded INPUT string:', OnehotToWord(emb_in))
            
            # Entering the predictions of 'T' and 'P' at the second position
            emb_out_onehot[1:1] = [simple_out_onehot[0]]
            # Entering the prediction of embedded char at the second last position
            emb_out_onehot.insert(len(emb_out_onehot)-1, emb_char_onehot)
    
            
            ergIO.append((self.OnehotToWord(emb_in_onehot),
                         np.array(emb_in_onehot, dtype=np.int8), 
                         np.array(emb_out_onehot, dtype=np.int8)))
        
        return ergIO
        
        
    def generate_RGInputOutput(self, do_ERG, nof_strings, reber_strings, 
                               string_max_length=None, sort_by_length=False):
        
        if reber_strings != 'mix':
            
            rg_inputoutput=[]    
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
                
                for rep in range(int(np.floor(nof_strings/len(reber_strings)))):
                    rg_inputoutput.append((in_string,
                                           np.array(inseq_onehot, dtype=np.int8), 
                                           np.array(outseq_onehot, dtype=np.int8)))
            
            random.shuffle(rg_inputoutput)
            return np.array(rg_inputoutput, dtype=object)
        
        else:
            if do_ERG:
                     
                if sort_by_length:
                    rg_inputoutput = sorted(self.get_n_erg(n=nof_strings, maxLength=string_max_length), key=reberstring_length)
                else:
                    rg_inputoutput = self.get_n_erg(n=nof_strings, maxLength=string_max_length)
        
            else:
                              
                if sort_by_length:
                    rg_inputoutput = sorted(self.get_n_srg(n=nof_strings, maxLength=string_max_length), key=reberstring_length)
                else:
                    rg_inputoutput = self.get_n_srg(n=nof_strings, maxLength=string_max_length)
                    
                                 
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
            transitions = self.graph[node]
            try:
                node = transitions[0][transitions[1].index(c)]
            except ValueError: 
                return False
        
        return True  

    
    def count_allTransitions(self, in_reberStrings):
        
        dict_count_allTransitions = {}
        
        for transition in self.all_possibleTransitions:
            counter = 0
            end_at = len(transition)-1
            
            for string in in_reberStrings:
                for i in range(len(string)-end_at):
                    trans = ''
                    for j in range(len(transition)):
                        trans += string[i+j]
            
                    if trans == transition:
                        counter += 1
            
            dict_count_allTransitions[transition] = counter
        
        return dict_count_allTransitions
