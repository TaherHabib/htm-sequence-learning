import numpy as np
import pandas as pd
import copy

class htm_net():
    
    """
    
    """
    
    def __init__(self, M=8, N=125, k=25, cell=None):
        
        self.M = M
        self.N = N
        self.k = k
        self.htm_cell = cell
        
        
    def get_net_dims(self):
        
        """

        Returns
        -------
        None.

        """
        
        return (self.M, self.N)
        
        