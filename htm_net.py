import numpy as np
import pandas as pd
import copy
import htm_cell as CELL 

class htm_net(CELL):
    
    """
    
    """
    
    def __init__(self, M=None, N=None, n_dendrites=None, n_synapses=None, 
                 nmda_th=None, perm_th=None, perm_init=None, k=None):
        
        super().__init__(M, N, n_dendrites, n_synapses, nmda_th, perm_th, perm_init)
        
        self.k = k
        
        
        
    def get_net_dims(self):
        
        """

        """
        
        return (self.M, self.N)
        
        
    def compute_net_state(self):
        
        """
        
        """