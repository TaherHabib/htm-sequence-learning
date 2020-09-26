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
                      init_permanence, k, do_ERG)

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
        
        
        
        
        
        
        
        
# =============================================================================
# 
# =============================================================================
