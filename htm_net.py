import numpy as np
import pandas as pd


from htm_cell import HTM_CELL

# =======================DEFINING CUSTOM FUNCTIONS=============================

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

    def __init__(self, M=None, N=None, k=None, n_dendrites=None, n_synapses=None, 
                 nmda_th=None, perm_th=None, perm_init=None):
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
        Computes the current step's predictions. Disregarding the LRD mechanism.

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
                cell_connSynapses = cell.get_cell_connSynapses() # is a list of 32 MxN matrices, shape: (32,M,N)
                
                # 'cell_dendActivity' will be a boolean array of shape (<cell.n_dendrites>,)
                cell_dendActivity = [dot_prod(prev_state,cell_connSynapses[d])>cell.nmda_th 
                                     for d in range(len(cell_connSynapses))]
                
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
        Computes the current timestep's network activity and predictions, based
        on the previous timestep's state of the network and the current 
        timestep's input.

        Parameters
        ----------
        prev_state : MxN binary matrix of network's acitivity at the previous
        timestep.
        
        curr_input : binary vector of current input, shape (N,), with 'k' 1's.

        Returns
        -------
        curr_pred : binary MxN matrix of current timestep's predictions (input 
        chars for the next timestep).
    
        net_state : binary MxN matrix of network's activity at current timestep. 

        """
        
        # 'curr_preds' is MxN binary matrix holding predictions for current timetep
        curr_pred = self.get_onestep_prediction(prev_state)
        
        net_state = []
        
        # Computing net state such that all minicolumns with current inputs are
        # fully activated.
        for m in range(self.M):
            net_state.append(curr_input)
        net_state = np.array(net_state) # MxN binary matrix
        
        # 'net_state*curr_pred' gives MxN binary matrix of only those cells that
        # are predicted AND present in the current input. Adding 'net_state' to 
        # this gives binary MxN 'net_state' from line 144 above but with the 
        # predicted cells with value '2'. The next step is to find those columns
        # in 'net_state*curr_pred + net_state' with '2' as an entry and subtract 1.
        # The following 6 lines of code are computing eq. 1, pg. 6 in the proposal.
        
        # NOTE: Although the learning rules are designed to make the following
        # impossible, but even if it so happens that TWO DIFFERENT cells are predicted
        # in the same minicolumn at a particular time step, then the equation below
        # will make those cells become silent or active depending on whether that 
        # particular minicolumn is in the set of current timestep's input or not.
        # Hence, the equation is robust to such special cases.
        
        net_state = net_state*curr_pred + net_state
        
        for n in range(self.N):
            mc = net_state[:,n]
            if 2 in mc:
                net_state[:,n] = net_state[:,n] - 1
        
        return curr_pred, net_state
    
    
    def do_net_synaPermUpdate(self, prev_pred=None, prev_state=None):
        
        #_______________________CASE I__________________________
        # When winning column is not predicted (as would happen in the 
        # initial stage after initialization of the network)
        
        
        
        #_______________________CASE II__________________________
        
        
        
        return None
    
    
    def get_NETWORK(self):
        """
        Returns the network architecture – MxN matrix of HTM_CELLs

        Returns
        -------
        MxN matrix of HTM_CELLs
        
        """
        return self.net_arch
    
    
    def prune_net_NegSynaPermanences(self):
        
        for i in range(self.M):
            for j in range(self.N):
                cell = self.net_arch[i,j]
                cell.dendrites[cell.dendrites<0] = 0
                
        return
    

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
 

# array to store the MxN matrix – at each timestep – of each matrix P of 
# shape (<dendrites_percell>,M,N) which stores the permanence values of that cell
# htm_net_synaPerm = []


      

# =============================================================================
