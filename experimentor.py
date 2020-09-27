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

# Cell params
dendrites_percell = 32
connSynapses_perdend = 32
nmda_threshold = 12
permanence_threshold = 0.4
init_permanence = 0.2

# Task params
do_ERG = False


# ========================INITIALIZING========================================

rg = Reber_Grammar(N, k)
df_CharsToMinicols = rg.df_CharsToMinicols


htm_network = HTM_NET(M, N, dendrites_percell, connSynapses_perdend, nmda_threshold, permanence_threshold,
                      init_permanence, k)

htm_init_state = np.zeros([M,N])


# =================GENERATING INPUT AND PREDICTIONS STRINGS====================

# Generate Input strings and Predictions
nof_strings = 1

if do_ERG:
    rg_inputoutput = rg.get_n_erg(nof_strings)
else:
    rg_inputoutput = rg.get_n_srg(nof_strings)

list_in_strings = [rg_inputoutput[i][0] for i in range(nof_strings)]
list_out_strings = [rg_inputoutput[i][1] for i in range(nof_strings)]


# =============================================================================
curr_state = htm_init_state

# array to store MxN binary state matrix of HTM network at each timestep
htm_states = []
# array to store MxN binary predition matrix of HTM network at each timestep
htm_preds = []


for string_idx in range(nof_strings):
    in_string = list_in_strings[string_idx]
    for step in range(len(in_string)):
        
        # in_string[step] is a binary 1xN vector with 'k' 1s.
        curr_pred, curr_state = htm_network.get_net_state(prev_state=curr_state, curr_input=in_string[step])
        htm_preds.append(curr_pred)
        htm_states.append(curr_state)
        
        
        
        
        
        
        
        
# ================================NOTES=======================================

#_________I__________
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

# NMDA threshold = 12.
# NMDA threshold/nof synapses per dendrites = 12/32 ~ 37.5%

# False match probability = 8.733769726186268e-15
