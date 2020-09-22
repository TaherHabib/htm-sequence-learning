import numpy as np
import pandas as pd


from htm_cell import HTM_CELL



class HTM_NET():

    def __init__(self, M=None, N=None, n_dendrites=None, n_synapses=None, 
                 nmda_th=None, perm_th=None, perm_init=None, k=None):
        """

        Parameters
        ----------
        M : TYPE, optional
            DESCRIPTION. The default is None.
        N : TYPE, optional
            DESCRIPTION. The default is None.
        n_dendrites : TYPE, optional
            DESCRIPTION. The default is None.
        n_synapses : TYPE, optional
            DESCRIPTION. The default is None.
        nmda_th : TYPE, optional
            DESCRIPTION. The default is None.
        perm_th : TYPE, optional
            DESCRIPTION. The default is None.
        perm_init : TYPE, optional
            DESCRIPTION. The default is None.
        k : TYPE, optional
            DESCRIPTION. The default is None.
        
        Returns
        -------
        None.

        """
        
        self.M = M # 8
        self.N = N # 175 = k*M
        self.k = k # 25
        
        self.net_arch = np.empty([self.M, self.N], dtype=HTM_CELL)
        
        for i in range(self.M):
            for j in range(self.N):
                cell = HTM_CELL(M,N,n_dendrites,n_synapses,nmda_th,perm_th,perm_init)
                self.net_arch[i,j] = cell
    
        return
    
    
    def compute_predictions(self, prev_state=None):
        
        """
        """
        # ASSUMPTION: There will never be two dendrites on the same cell that
        # get activated to the same activity pattern in the population.
        
        curr_preds = np.zeros([self.M, self.N])
        
        return curr_preds
    
        
    def compute_net_state(self, prev_state=None, curr_input=None):
        
        """
        
        """
        
        # 'curr_preds' is MxN matrix holding predictions for current timetep
        curr_preds = self.compute_predictions(prev_state)
        
        return net_state
    
    
    def get_net_dims(self):
        """
        Returns
        -------
        tuple (int,int): (no. of cells per minicolumn, no. of minicolumns)
        
        """
        
        return (self.M, self.N)

        
     
    

# ==========================ROUGH==============================================

# self.net_dims = np.array([self.M, self.N])

# initializing each neuron of the network

# super().__init__(M, N, n_dendrites, n_synapses, nmda_th, perm_th, perm_init)

# =============================================================================
# minicolumns = np.arange(self.N)
# random.shuffle(minicolumns)
# for i in range(self.N//self.k):
#     mc = minicolumns[i*self.k:(i+1)*self.k]
# =============================================================================
       

# =============================================================================
