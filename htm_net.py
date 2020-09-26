import numpy as np
import pandas as pd


from htm_cell import HTM_CELL

# =============================================================================

def dot_prod(matrix_1=None, matrix_2=None):
    """
    Computes the element-wise multiplication over two equi-dimensional matrices,
    sums up the entries of the resulting matrix and returns a scalar value.

    Parameters
    ----------
    matrix_1 : float array of shape (MxN)
    matrix_2 : float array of shape (MxN)

    Returns
    -------
    float, scalar value

    """
    
    return np.sum(np.multiply(matrix_1, matrix_2))


# ========================DEFINING HTM NETWORK=================================

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
        
        # Initializing every cell in the network, i.e. setting up the dendrites for each cell.
        for i in range(self.M):
            for j in range(self.N):
                cell = HTM_CELL(M,N,n_dendrites,n_synapses,nmda_th,perm_th,perm_init)
                self.net_arch[i,j] = cell
    
        return
    
    
    def get_onestep_prediction(self, prev_state=None):
        """
        Computes the current step's predictions.

        Parameters
        ----------
        prev_state : binary array of shape (MxN), containing the activity of 
        cell population from previous time step.
        
        Returns
        -------
        curr_preds : binary array of shape (MxN), containing the predictions 
        for the current time step.

        """
        
        # ASSUMPTION: There will never be two dendrites on the same cell that
        # get activated to the same activity pattern in the population.
        
        curr_pred = np.zeros([self.M, self.N])
        
        for j in range(self.N):
            for i in range(self.M):
                cell = self.net_arch[i,j]
                cell_connSynapses = cell.get_connected_synapses() # is a list of 32 MxN binary matrices
                
                # 'cell_dendActivity' will be a boolean array 
                cell_dendActivity = [dot_prod(prev_state,cell_connSynapses[d])>cell.nmda_th for d in range(len(cell_connSynapses))]
                
                # if any denrite of the cell is active, then the cell becomes predictive.
                if any(cell_dendActivity):
                    curr_pred[i,j] = 1
                else:
                    curr_pred[i,j] = 0
        
        return curr_pred
    
    def get_LRD_prediction(self):
        """
        

        Returns
        -------
        None.

        """
        
        return
    
        
    def get_net_state(self, prev_state=None, curr_input=None):
        
        """
        
        """
        
        # 'curr_preds' is MxN binary matrix holding predictions for current timetep
        curr_pred = self.get_onestep_prediction(prev_state)
        
        net_state = []
        
        # Computing net state such that all minicolumns with current inputs are
        # fully activated.
        for m in range(self.M):
            net_state.append(curr_input)
        net_state = np.array(net_state) # MxN binary matrix
            
            
        net_state = curr_pred*curr_input
        
        return curr_pred, net_state
    
    
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
