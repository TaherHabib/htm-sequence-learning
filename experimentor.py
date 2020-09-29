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


htm_network = HTM_NET(M, N, k, n_dendrites=dendrites_percell, n_synapses=connSynapses_perdend, 
                      nmda_th=nmda_threshold, perm_th=permanence_threshold, perm_init=init_permanence)

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

# array to store MxN binary state matrix of HTM network at each timestep
htm_states = []

# array to store MxN binary predition matrix of HTM network at each timestep
htm_preds = []

# array to store MxN matrix of HTM cells at each timestep. This storage is mainly 
# to have an access to the evolution of the synaptic permanence values of each 
# cell in the network with time. 
htm_networks = []

for string_idx in range(nof_strings):

    curr_state = htm_init_state
    curr_pred = htm_init_state
    in_string = list_in_strings[string_idx]

    for step in range(len(in_string)):
        
        # in_string[step] is a binary 1xN vector with 'k' 1s.
        curr_pred, curr_state = htm_network.get_net_state(prev_pred=curr_pred, prev_state=curr_state,
                                                          curr_input=in_string[step])
        
        htm_preds.append(curr_pred)
        htm_states.append(curr_state)
        htm_networks.append(htm_network.get_NETWORK())
        
        # PRUNING PERMANENCE VALUES that have become negative due to updating via
        # Hebbian rule, and setting them all to zero.
        htm_network.prune_net_NegPermanences()
        
        # HEBBIAN LEARNING & SYNAPTIC PERMANENCE UPDATE
        htm_network.do_net_synaPermUpdate(prev_input=in_string[step], prev_pred=curr_pred, prev_state=curr_state)
        

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

# NMDA threshold = 12.
# NMDA threshold/nof synapses per dendrites = 12/32 ~ 37.5%

# False match probability = 8.733769726186268e-15


#________III_________
