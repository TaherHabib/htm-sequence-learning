import numpy as np
import pandas as pd
import copy

class htm_cell():
    
    """
    
    """
    
    def __init__(self, n_dendrites=None, n_synapses=None, nmda_th=None, perm_th=None, perm_init=None, conn_sparsity=None):
        
        self.n_dendrites = n_dendrites
        self.n_synapses = n_synapses
        self.nmda_th = nmda_th
        self.perm_th = perm_th
        self.perm_init = perm_init
        self.conn_sparsity = conn_sparsity
        
        self.synaptic_struct_raw = # list of list containing binary values of raw synaptic connections
        self.synaptic_perm = # list of list containing permanence values of synaptic connections
        self.synaptic_conn = # list of list containing permanence values of synaptic connections
        