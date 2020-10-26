import numpy as np
import pandas as pd


#from htm_cell import HTM_CELL
from htm_net_v2 import HTM_NET
from rebergrammar_generator import *


class Experimentor():
    
    def __init__(self, M=None, N=None, k=None, 
                 n_dendrites=None, n_synapses=None, nmda_th=None, perm_th=None, perm_init=None, 
                 perm_decrement=None, perm_increment=None, perm_decay=None, perm_boost=None,
                 activity_horizon=None, activity_th=None,
                 do_ERG=False, nof_strings=1000):
        
        self.n_dendrites = n_dendrites
        self.n_synapses = n_synapses 
        self.nmda_th = nmda_th
        self.perm_th = perm_th
        self.perm_init = perm_init
        self.activity_horizon = activity_horizon
        self.activity_th = activity_th
        
        self.M = M 
        self.N = N 
        self.k = k
        self.perm_decrement = perm_decrement
        self.perm_increment = perm_increment
        self.perm_decay = perm_decay
        self.perm_boost = perm_boost
        
        self.do_ERG = do_ERG
        self.nof_strings = nof_strings
    
        # Initializing Grammar
        self.rg = Reber_Grammar(N, k)
        self.df_CharsToMinicols = self.rg.df_CharsToMinicols
        self.z_minicols = np.zeros(self.N, dtype=np.int8)
        self.z_minicols[self.df_CharsToMinicols['Z']] = 1 
                    
        
        # Initializing Network
        self.htm_network = HTM_NET(M, N, k, 
                                   n_dendrites=n_dendrites, n_synapses=n_synapses, nmda_th=nmda_th, 
                                   perm_th=perm_th, perm_init=perm_init,
                                   perm_decrement=perm_decrement, perm_increment=perm_increment,
                                   perm_decay=perm_decay, perm_boost=perm_boost)
        
        
        # Generating Input and Prediction Strings
        if self.do_ERG:
            rg_inputoutput = self.rg.get_n_erg(self.nof_strings)
        else:
            rg_inputoutput = self.rg.get_n_srg(self.nof_strings)
        
        self.list_in_strings = np.array([rg_inputoutput[i][0] for i in range(self.nof_strings)], dtype=object)
        self.list_out_strings = np.array([rg_inputoutput[i][1] for i in range(self.nof_strings)], dtype=object)

        in_strings_alpha = []
        for string_oh in self.list_in_strings:
            string_alpha = self.rg.OnehotToWord(string_oh)
            in_strings_alpha.append(string_alpha)
        
        self.in_strings_alpha = np.array(in_strings_alpha, dtype=object)
        
        return
    
    
    
    def run_experiment(self):
        
        # DataFrame to store results for each string in 'list_in_strings'
        df_res = pd.DataFrame(columns=('reber_string', 'htm_states', 'htm_preds', 'htm_preds_dend', 'htm_network'))
        
        # 'htm_states' and 'htm_preds' store MxN binary state and prediction matrix of HTM network at each timestep 
        # (each letter), for each input reber string, respectively.
        
        # 'htm_preds_dend' stores MxN matrix of responsible active dendrites for each of the MxN neuron's prediction
        # in HTM network at each timestep (each letter), for each input reber string.
        
        # 'htm_networks' stores MxN matrix of HTM cells at each timestep, for each input reber string. This storage 
        # is mainly to have an access to the evolution of the synaptic permanence values of each cell in the 
        # network with time. 
        
        for string_idx in range(self.nof_strings):
    
            curr_state = np.zeros([self.M,self.N], dtype=np.int8)
            curr_pred = np.zeros([self.M,self.N], dtype=np.int8)
            
            htm_states=[]
            htm_preds=[]
            htm_preds_dend=[]
            htm_net_ = self.htm_network.get_NETWORK(char_minicols='all')
            
            in_string = self.list_in_strings[string_idx]
            in_string_alpha = self.in_strings_alpha[string_idx]    
            
            # 'len(in_string) is actually one less than the actual length of the string,
            # due to the final ommission of 'Z'.
            for step in range(len(in_string)):
                
                # in_string[step] is a binary 1xN vector (np.array) with 'k' 1s.
                curr_state, curr_pred, curr_pred_dend = self.htm_network.get_net_state(prev_pred=curr_pred,
                                                                                       curr_input=in_string[step])
                
                htm_states.append(curr_state)
                htm_preds.append(curr_pred)
                htm_preds_dend.append(curr_pred_dend)
                
                if step == 0:
                    
                    # No learning can occur for 'A' and its prediction. 
                    continue 
                
                else:
                    
                    # PRUNING Negative Permanence Values (set them to 0)
                    # PRUNING Positive Permanence Values (set them to 1)
                    self.htm_network.prune_net_Permanences()
                    
                    # HEBBIAN LEARNING & SYNAPTIC PERMANENCE UPDATE
                    # Here, the network is learning to predict for symbol that is currrently in 'in_string[step]'
                    multi_cell_MaxOverlap = self.htm_network.do_net_synaPermUpdate(curr_state=curr_state, 
                                                                                   prev_state=htm_states[step-1],
                                                                                   prev_pred=htm_preds[step-1], 
                                                                                   prev_pred_dend=htm_preds_dend[step-1], 
                                                                                   curr_input=in_string[step])
                    #htm_networks.append(self.htm_network.get_NETWORK(char_minicols=in_string[step]))
                    
                    if multi_cell_MaxOverlap == True:
                        print('Multi Cell MaxOverlap in String:', in_string_alpha, 'at:', in_string_alpha[step])
                    
                
                # LEARNING TO PREDICT 'Z' at the penultimate step
                if step == len(in_string)-1:
                    
                    curr_state, _, _ = self.htm_network.get_net_state(prev_pred=curr_pred,
                                                                      curr_input=self.z_minicols)
                    htm_states.append(curr_state)
                    # Since there won't be any predictions occurring at the timestep of 'Z', as input,
                    # 'curr_pred' and 'curr_pred_dend' need not be appended at all. Also, NONE of the cells
                    # in the network would be reinforcing their pre-synapses with the cells responsible
                    # for 'Z'. In other words, the output of 'dot_prod(net_state,cell_connSynapses)' in 
                    # 'get_onestep_prediction()' will always be all zero, at this step!
                
                    self.htm_network.prune_net_Permanences()
                    
                    multi_cell_MaxOverlap =self.htm_network.do_net_synaPermUpdate(curr_state=curr_state, 
                                                                                  prev_state=htm_states[step],
                                                                                  prev_pred=htm_preds[step], 
                                                                                  prev_pred_dend=htm_preds_dend[step], 
                                                                                  curr_input=self.z_minicols)
                    #htm_networks.append(self.htm_network.get_NETWORK(char_minicols=self.z_minicols))
                    
                    if multi_cell_MaxOverlap == True:
                        print('Multi Cell MaxOverlap in String:', in_string_alpha, 'at:', in_string_alpha[step])
                    
            
            df_res.loc[string_idx] = [in_string_alpha, 
                                      np.array(htm_states), 
                                      np.array(htm_preds), 
                                      np.array(htm_preds_dend), 
                                      htm_net_]
            
            # np.array(htm_states) is numpy array of shape: (<len(in_string)>+1,M,N)
            # np.array(htm_preds) is numpy array of shape: (<len(in_string)>,M,N)
            # np.array(htm_preds_dend) is numpy array of shape: (<len(in_string)>,M,N)
            
            
        dict_results = {
            'results': df_res,
            'chars_to_minicols': self.df_CharsToMinicols,
            'in_strings_onehot': self.list_in_strings,
            'out_strings_onehot': self.list_out_strings}
        
        return dict_results
    
        
            
        
            
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
# In my HTM version, M=8, N=175, k=25.
# Total number of cells = 8*175 = 1400
# Initial (before SDR learning) activity level in the network = 25*8 = 200 (~14%)
# After SDR learning, activity is expected to reduce to 25*1 = 25 (~1.8%)

# Max nof dendrites per neuron = 32
# Max nof synapses per dendrites = 32 (approx. max. 2.3% of the cells are connected 
# to a single dendrite)
# Thus, nof connected synapses per neuron = 32*32 = 1024, which is ~73% of 
# the totol network size.

# NMDA threshold = 15.
# NMDA threshold/nof synapses per dendrites = 15/32 ~ 47%

# False match probability = 1.492248791690464e-20



#________III_________
# np.where(b==np.amax(b)) always gives a 2-tuple as output, for all 'b' such that 
# 'b' is a list of size (m,n), containing 'm' numpy.arrays of length 'n' filled with
# numeric float integers.
# The first entry of the tuple contains the array index(indices) which contain the 
# maximum numeric entry in 'b', the second entry of the tuple contains the index(indices)
# within those arrays at which the max element could be found. 

# For instance, if 
# b = [array([2.25, 2.5 , 0.4 ]),
#      array([4.5, 5. , 0.8]),
#      array([11.25, 12.5 ,  2.  ]),
#      array([ 1. ,  2. , 12.5])]
# Then, np.where(b==np.amax(b)) will give: (array([2, 3]), array([1, 2])) as ouput. 



#_________IV__________

# On Deciding on the activity horizon and threshold for boosting cell activity
# 
