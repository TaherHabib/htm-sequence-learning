"""
FILE DESCRIPTION:
------------------------

....


OTHER NOTES:
------------
Issue 001: 
    When a bursting column's capacity is found to be FULL, when growing a new dendrite in NO BEST MATCH condition.
    
Issue 002: 
    When a bursting column's capacity is found to be FULL, when growing a new dendrite in the condition where there was
    a best match found, but there were not enough capacity on the best matching dendrite to grow new synapses to the 
    previous timestep's winner cells.
    
Issue 003:
    When in a bursting column, a maximum overlap with the previous active cells of the network is found at two different
    places – either on different cells in a column, or on different dendrites of the same cell in that column.
    
"""

import numpy as np
import pandas as pd
import copy
import random

from htm_net_v3 import HTM_NET
from rebergrammar_generator_v3 import *


class Experimentor():
    
    def __init__(self, numColumns=None, cellsPerColumn=None, columnsPerChar=None, 
                 maxDendritesPerCell=None, maxSynapsesPerDendrite=None, 
                 nmdaThreshold=None, permThreshold=None, learningThreshold=None,
                 permInit=None, permInit_sd=None,
                 perm_decrement=None, perm_increment=None, perm_decay=None,
                 rg=None, rg_inputoutput=None,
                 htm_network=None,
                 verbose=2):
        
        self.nmdaThreshold = nmdaThreshold
        self.permThreshold = permThreshold
        
        self.M = cellsPerColumn
        self.N = numColumns 
        self.k = columnsPerChar
        
        self.rg = rg
        self.rg_inputoutput = rg_inputoutput
        self.nof_strings = len(rg_inputoutput)
        self.verbose = verbose
        
        # Computing maxDendriteDormancy
        self.maxDendriteDormancy = 8*30 
        
        # Onehot for 'Z'
        self.z_onehot = self.rg.CharToOnehot('Z')
        
        # Winner Cells for 'A'
        self.A_winner_cells = np.zeros([self.M, self.N], dtype=np.int8)
        random.seed(1)
        A_winnercells_i = random.choices(np.arange(self.M), k=self.k)
        for i in range(self.k):
            self.A_winner_cells[A_winnercells_i[i], self.rg.df_CharsToMinicols['A'][i]] = 1            
        
                                                                               
        if htm_network == None:
            # Initializing Network
            self.htm_network = HTM_NET(cellsPerColumn=self.M, numColumns=self.N, columnsPerChar=self.k, 
                                       maxDendritesPerCell=maxDendritesPerCell, 
                                       maxSynapsesPerDendrite=maxSynapsesPerDendrite, 
                                       nmdaThreshold=nmdaThreshold, permThreshold=permThreshold, 
                                       learningThreshold=learningThreshold, 
                                       permInit=permInit, permInit_sd=permInit_sd,
                                       perm_decrement=perm_decrement, perm_increment=perm_increment,
                                       perm_decay=perm_decay,
                                       dendriteDuty_UpperLimit=self.maxDendriteDormancy,
                                       verbose=verbose)
        else:
            self.htm_network = htm_network
            
        return
    
    
    def run_experiment(self):
        
        # DataFrame to store results for each string in 'list_in_strings'
        df_res = pd.DataFrame(columns=('reber_string', 'htm_states', 'htm_preds', 'htm_predDendrites', 
                                       'htm_winner_cells', 'nof_dendrites', 'issue'))
        
        # 'htm_states' and 'htm_preds' store MxN binary state and prediction matrix of HTM network at each timestep 
        # (each letter), for each input reber string, respectively.
        
        # 'htm_predDendrites' stores MxN matrix of responsible active dendrites for each of the MxN neuron's prediction
        # in HTM network at each timestep (each letter), for each input reber string.
        
        # 'htm_networks' stores MxN matrix of HTM cells at each timestep, for each input reber string. This storage 
        # is mainly to have an access to the evolution of the synaptic permanence values of each cell in the 
        # network with time. 
        
        for string_idx in range(self.nof_strings):
            
            htm_states=[]
            htm_preds=[]
            htm_predDendrites=[]
            htm_winnerCells=[]
            
            in_string = self.rg_inputoutput[string_idx][1]
            in_string_alpha = self.rg_inputoutput[string_idx][0]    
            curr_pred = np.zeros([self.M,self.N], dtype=np.int8)
            
            if self.verbose > 0:
                print('\n', color.BOLD, 'String_idx: ', string_idx, color.END, 'Input String: ', in_string_alpha)
            
            # 'len(in_string) is actually one less than the actual length of the string,
            # due to the final ommission of 'Z'.
            for step in range(len(in_string)):
                
                if self.verbose > 1:
                    print(color.BLUE,color.BOLD, 'Running for step: ', step, ' for: ', in_string_alpha[step], color.END, color.END)
                
                # in_string[step] is a binary 1xN vector (np.array) with 'k' 1s.
                curr_state, curr_pred, curr_pred_dend  = self.htm_network.get_net_state(prev_pred=curr_pred,
                                                                                        curr_input=in_string[step])
                if step == 0:
                
                    winnerCells = self.A_winner_cells
                    
                    htm_states.append(curr_state)
                    htm_preds.append(curr_pred)
                    htm_predDendrites.append(curr_pred_dend)
                    htm_winnerCells.append(winnerCells)
                    
                    # No learning can occur for 'A' and its prediction. 
                    continue
                    
                else:
                    
                    self.htm_network.update_net_dendriteDutyCycle()
                    
                    # HEBBIAN LEARNING & SYNAPTIC PERMANENCE UPDATE
                    # Here, the network is learning to predict for symbol that is currrently in 'in_string[step]'
                    winnerCells = self.htm_network.update_net_synapticPermanences(curr_state=curr_state,
                                                                                  prev_state=htm_states[step-1],
                                                                                  prev_pred=htm_preds[step-1], 
                                                                                  prev_predDendrites=htm_predDendrites[step-1],
                                                                                  prev_winnerCells=htm_winnerCells[step-1]
                                                                                  )
                    #__________________________________________________________________    
                    # The following checks for a fundamental flaw in implementation of 
                    # the learning of SDRs
                    # REPORT ERROR AND BREAK THE EXECUTION for the current string!
                    if np.count_nonzero(winnerCells) < self.nmdaThreshold:
                        print(color.RED, color.BOLD, '::::::::::::::::::::::::::::::::::Issue 005 in step: ', step, \
                              '::::::::::::::::::::::::::::::::::', color.END, color.END)
                        issue = '005'
                        break
                    
                    else:
                        issue = 'nan'
                    
                    htm_states.append(curr_state)
                    htm_preds.append(curr_pred)
                    htm_predDendrites.append(curr_pred_dend)
                    htm_winnerCells.append(winnerCells)
                    #htm_networks.append(self.htm_network.get_NETWORK(char_minicols=in_string[step]))
                    
                # LEARNING TO PREDICT 'Z' at the penultimate step
                if step == len(in_string)-1:
                    
                    if self.verbose > 1:
                        print(color.BLUE, color.BOLD, 'Running for step: ', step + 1, ' for: Z', color.END, color.END)
                    
                    curr_state, _, _ = self.htm_network.get_net_state(prev_pred=curr_pred,
                                                                      curr_input=self.z_onehot)
                    htm_states.append(curr_state)
                    # Since there won't be any predictions occurring at the timestep of 'Z', as input,
                    # 'curr_pred' and 'curr_pred_dend' need not be appended at all. Also, NONE of the cells
                    # in the network would be reinforcing their pre-synapses with the cells responsible
                    # for 'Z'. In other words, the output of 'dot_prod(net_state,cell_connSynapses)' in 
                    # 'get_onestep_prediction()' will always be all zero, at this step!
                
                    self.htm_network.update_net_dendriteDutyCycle()
                    
                    _ = self.htm_network.update_net_synapticPermanences(curr_state=curr_state,
                                                                        prev_state=htm_states[step],
                                                                        prev_pred=htm_preds[step], 
                                                                        prev_predDendrites=htm_predDendrites[step],
                                                                        prev_winnerCells=htm_winnerCells[step]
                                                                        )
                    #htm_networks.append(self.htm_network.get_NETWORK(char_minicols=self.z_minicols))
            
            count_dend = 0
            for i in range(self.M):
                for j in range(self.N):
                    count_dend += self.htm_network.net_arch[i,j].get_cell_numDendrites()

            df_res.loc[string_idx] = [in_string_alpha, 
                                      np.array(htm_states), 
                                      np.array(htm_preds), 
                                      np.array(htm_predDendrites),
                                      np.array(htm_winnerCells),
                                      count_dend,
                                      issue]
            
            # np.array(htm_states) is numpy array of shape: (<len(in_string)>+1,M,N)
            # np.array(htm_preds) is numpy array of shape: (<len(in_string)>,M,N)
            # np.array(htm_preds_dend) is numpy array of shape: (<len(in_string)>,M,N)
            # np.array(htm_winner_cells) is numpy array of shape: (<len(in_string)>,M,N)
            
        dict_results = {
            'results': df_res,
            'final_net': self.htm_network.net_arch
            }
        
        return dict_results
    

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

       
            
# ================================NOTES=======================================

#_________I_________
# In the original HTM, M=32, N=2048, k=40. 
# Total number of cells = 32*2048 =  65,536
# Initial (before SDR learning) activity level in the network = 40*32 = 1280 (~2%)
# After SDR learning, activity is expected to reduce to 40*1 = 30 (~0.05%)

# Max nof dendrites per neuron = 128
# Max nof synapses per dendrites = 128 (approx. max. 0.2% of the cells are connected 
# to a single dendrite)
# Thus, nof connected synapses per neuron = 128*128 = 16,384, which is 25% of 
# the totol network size.

# NMDA threshold = 15. 
# NMDA threshold/nof synapses per dendrites = 15/128 ~ 11.7%

# False match probability = 3.7848536276106233e-31

#________II__________
# In my HTM version, M=16, N=32*7, k=32.
# Total number of cells = 16*32*7 = 3584. 
# Initial (before SDR learning) activity level in the network = 32*16 = 512 (~14%)
# After SDR learning, activity is expected to reduce to 32*1 = 32 (~0.9%)

# Max nof dendrites per neuron = 128
# Max nof synapses per dendrites = 32 (approx. max. 2.3% of the cells are connected 
# to a single dendrite)
# Thus, nof connected synapses per neuron = 32*32 = 1024, which is ~73% of 
# the totol network size.

# NMDA threshold = 20.
# NMDA threshold/nof synapses per dendrites = 20/32 ~ 63%

# False match probability = 1.0291712295744538e-36
