import numpy as np
import pandas as pd
import copy

class htm_cell():
    
    """
    
    """
    
    def __init__(self, M=None, N=None, n_dendrites=None, n_synapses=None, 
                 nmda_th=None, perm_th=None, perm_init=None):
        
        """
        
        """
        
        self.M = M # number of cells per minicolumn
        self.N = N # number of minicolumns
        
        self.n_dendrites = n_dendrites # number of dendritic segments on a single cell
        self.n_synapses = n_synapses # total number of CONNECTED synapses on a single dendritic segment
        
        self.nmda_th = nmda_th
        self.perm_th = perm_th
        self.perm_init = perm_init
        
        self.dendrites = [] # list containing the matrices of potential synapses (permanence values)
                            # for each dendrite of the HTM cell
        
        # Initializing potential synapses for each dendritic segment of the HTM cell
        for d in range(self.n_dendrites):
            dend = np.random.normal(loc=self.perm_init, scale=0.05, size=[self.M, self.N])
            self.dendrites.append(dend)
            
        
    def max_overlap_dendrite(self):
        
        """
        
        """
        
    def cell_output(self):
        
        """
        
        """
        
    def cell_predict(self):
        
        """
        
        """