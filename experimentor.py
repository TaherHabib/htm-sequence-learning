"""
This file conducts the experiment on the HTM network on learning SRG and ERG 
tasks.

"""

import numpy as np
import pandas as pd


from htm_cell import HTM_CELL
from htm_net import HTM_NET
from rebergrammar_generator import *


# ============================PARAM SETTING===================================

# Network params
M = 8
N = 175
k = 25

perm_decrement = 0.05 # p-
perm_increment = 2*perm_decrement # p+
perm_decay = 0.1*perm_decrement # p--
perm_boost = 0.1*perm_decrement # p++

# Cell params
dendrites_percell = 32
connSynapses_perdend = 32
nmda_threshold = 15
permanence_threshold = 0.40
init_permanence = 0.25

len_activity_horizon = 
activity_threshold = 

# Task params
do_ERG = False


# ========================INITIALIZING========================================

rg = Reber_Grammar(N, k)
df_CharsToMinicols = rg.df_CharsToMinicols


htm_network = HTM_NET(M, N, k, 
                      n_dendrites=dendrites_percell, n_synapses=connSynapses_perdend, nmda_th=nmda_threshold, 
                      perm_th=permanence_threshold, perm_init=init_permanence,
                      perm_decrement=perm_decrement, perm_increment=perm_increment,
                      perm_decay=perm_decay, perm_boost=perm_boost)


# =================GENERATING INPUT AND PREDICTIONS STRINGS====================

# Generate Input strings and Predictions
nof_strings = 1

if do_ERG:
    rg_inputoutput = rg.get_n_erg(nof_strings)
else:
    rg_inputoutput = rg.get_n_srg(nof_strings)

list_in_strings = [rg_inputoutput[i][0] for i in range(nof_strings)]
list_out_strings = [rg_inputoutput[i][1] for i in range(nof_strings)]

in_strings_alpha = []
for string_oh in list_in_strings:
    string_alpha = rg.OnehotToWord(string_oh)
    in_strings_alpha.append(string_alpha)


# =======================STARTING EXPERIMENT===================================

# dict to store MxN binary state matrix of HTM network at each timestep, for 
# each input string as key.
dict_htm_states = {}

# dict to store MxN binary predition matrix of HTM network and the dendrites 
# responsible for those predictions at each timestep, for each input string 
# as key.
dict_htm_preds = {}
dict_htm_preds_dend = {}

# dict to store MxN matrix of HTM cells at each timestep, for each input
# string as key. This storage is mainly to have an access to the evolution of 
# the synaptic permanence values of each cell in the network with time. 
dict_htm_networks = {}

dict_htm_multicell_MaxOverlap = {}


for string_idx in range(nof_strings):
    
    key = in_strings_alpha[string_idx]    
    
    curr_state = np.zeros([M,N])
    curr_pred = np.zeros([M,N])
    curr_pred_dend = np.empty([M,N], dtype=object)
    curr_pred_dend[:] = np.nan
    
    htm_states=[]
    htm_preds=[]
    htm_preds_dend=[]
    htm_networks=[htm_network.get_NETWORK()]
    htm_multicell_MaxOverlap = []
    
    in_string = list_in_strings[string_idx]
    
    # 'len(in_string) is actually one less than the actual length of the string,
    # due to the final ommission of 'Z'.
    for step in range(len(in_string)):
        
        # in_string[step] is a binary 1xN vector (np.array) with 'k' 1s.
        curr_state, curr_pred, curr_pred_dend = htm_network.get_net_state(prev_pred=curr_pred,
                                                                          curr_input=in_string[step])
        
        htm_states.append(curr_state)
        htm_preds.append(curr_pred)
        htm_preds_dend.append(curr_pred_dend)
        
        if step == 0:
            
            # No learning can occur for 'A' and its prediction. 
            continue 
        
        else:
            
            # PRUNING Negative Permanence Values
            htm_network.prune_net_NegPermanences()
            
            # HEBBIAN LEARNING & SYNAPTIC PERMANENCE UPDATE
            # Here, the network is learning to predict for symbol that is currrently in 'in_string[step]'
            multi_cell_MaxOverlap = htm_network.do_net_synaPermUpdate(curr_state=curr_state, 
                                                                      prev_state=htm_states[step-1],
                                                                      prev_pred=htm_preds[step-1], 
                                                                      prev_pred_dend=htm_preds_dend[step-1], 
                                                                      curr_input=in_string[step])
            htm_networks.append(htm_network.get_NETWORK())
            htm_multicell_MaxOverlap.append(multi_cell_MaxOverlap)
            
        
        # LEARNING TO PREDICT 'Z' at the penultimate step
        if step == len(in_string)-1:
            
            z_minicols = np.zeros[N]
            z_minicols[df_CharsToMinicols['Z']] = 1 
            
            
            curr_state, _, _ = htm_network.get_net_state(prev_pred=curr_pred,
                                                         curr_input=z_minicols)
            htm_states.append(curr_state)
            # Since there won't be any predictions occurring at the timestep of 'Z', as input,
            # 'curr_pred' and 'curr_pred_dend' need not be appended at all. Also, NONE of the cells
            # in the network would be reinforcing their pre-synapses with the cells responsible
            # for 'Z'. In other words, the output of 'dot_prod(net_state,cell_connSynapses)' in 
            # 'get_onestep_prediction()' will always be all zero, at this step!
            
            multi_cell_MaxOverlap = htm_network.do_net_synaPermUpdate(curr_state=curr_state, 
                                                                      prev_state=htm_states[step],
                                                                      prev_pred=htm_preds[step], 
                                                                      prev_pred_dend=htm_preds_dend[step], 
                                                                      curr_input=z_minicols)
            htm_networks.append(htm_network.get_NETWORK())
            htm_multicell_MaxOverlap.append(multi_cell_MaxOverlap)
            
            
    
    dict_htm_states[key] = np.array(htm_states) # numpy array of shape: (<len(in_string)>+1,M,N)
    dict_htm_preds[key] = np.array(htm_preds) # numpy array of shape: (<len(in_string)>,M,N)
    dict_htm_preds_dend[key] = np.array(htm_preds_dend) # numpy array of shape: (<len(in_string)>,M,N)
    dict_htm_networks[key] = np.array(htm_networks) # numpy array of shape: (<len(in_string)>+1,M,N)
    dict_htm_multicell_MaxOverlap = np.array(htm_multicell_MaxOverlap) # numpy array of shape: 
                                                                       # (<len(in_string)>+1,)


# IMPORTANT       
# SOLVE THE PROBLEM OF COMPATIBILITY OF GET_NET_STATE()'S OUTPUT WITH THE OUTPUTS OF
# REBER GRAMMAR GENERATING FUNCTION. THERE IS NO FINAL 'Z' IN THE END OF EACH STRING (AS INPUT),
# BUT THERE IS A PREDICTION FOR 'Z'.
        
        
        
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


#----------------------------------------------------------------------
        # THIS IS WHERE PERFORMANCE MEASUREMENT WILL OCCUR !!!
        #
        # Take 'curr_pred' (a matrix of shape MxN) from the line above and 
        # compare it with 'list_out_strings[string_idx][step]' (also a matrix 
        # of shape MxN). 
        # 
        # NO WAIT!
        # 
        # I am anyway collecting the network predictions at each timestep in 
        # 'htm_preds'. I can then later on use this to measure the performance
        # of the network, after the network has run over a few thousand
        # strings.
        #----------------------------------------------------------------------
        
        