import numpy as np
import pandas as pd


from htm_cell import HTM_CELL

# =======================DEFINING CUSTOM FUNCTIONS=============================

def dot_prod(matrix_1=None, matrix_2=None):
    """
    Computes the element-wise multiplication of an MxN matrix with a list of 'n'
    other equi-dimensional matrices (n,M,N); and outputs a list of scalar values 
    of sum over all the entries of each of the resulting 'n' matrices.

    Parameters
    ----------
    matrix_1 : float array of shape (M,N)
    matrix_2 : float array of shape (<nof_matrices>,M,N). If only one MxN matrix 
    is passed, it should be of shape (1,M,N).

    Returns
    -------
    list of float, scalar values
    
    NOTE
    ----
    This function also excepts Boolean (or binary) arrays as inputs.

    """
    
    mult_res = np.multiply(matrix_1, matrix_2)
    
    result = []
    for i in range(len(mult_res)):
        result.append(np.sum(mult_res[i]))
    
    return np.array(result)


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
    
    
    def get_onestep_prediction(self, net_state=None):
        """
        Computes the current step's predictions. Disregarding the LRD mechanism.

        Parameters
        ----------
        net_state : binary array of shape (MxN), containing the activity of 
        cell population from current time step.
        
        Returns
        -------
        pred : binary array of shape (MxN), containing the current timestep's 
        predictions (input chars for the next timestep).

        """
        
        # ASSUMPTION: There will never be two dendrites on the same cell that
        # get activated to the same activity pattern in the population.
        
        
        pred = np.zeros([self.M, self.N])
        
        for j in range(self.N):
            for i in range(self.M):
                cell = self.net_arch[i,j]
                cell_connSynapses = cell.get_cell_connSynapses() # is a boolean list of 32 MxN matrices, 
                                                                 # shape: (32,M,N)
                
                # 'cell_dendActivity' will be a boolean array of shape (<cell.n_dendrites>,)
                cell_dendActivity = dot_prod(net_state,cell_connSynapses)>cell.nmda_th
                
                # if any denrite of the cell is active, then the cell becomes predictive.
                if any(cell_dendActivity):
                    pred[i,j] = 1.0
                
        return pred
    
    
    def get_LRD_prediction(self):
        """
        

        Returns
        -------
        None.

        """
        
        return
    
        
    def get_net_state(self, prev_pred=None, curr_input=None):
        """
        Computes the current timestep's network activity and predictions, based
        on the previous timestep's state of the network and the current 
        timestep's input.

        Parameters
        ----------
        prev_pred : MxN binary matrix of network's prediction at the previous
        timestep.
        
        prev_state : MxN binary matrix of network's activity at the previous
        timestep.
        
        curr_input : binary vector of current input, shape (N,), with 'k' 1's.

        Returns
        -------
        curr_pred : binary MxN matrix of current timestep's predictions (input 
        chars for the next timestep).
    
        curr_state : binary MxN matrix of network's activity at current timestep. 

        """
        
        curr_state = []
        
        # Computing net state such that all minicolumns with current inputs are
        # fully activated.
        for m in range(self.M):
            curr_state.append(curr_input)
        curr_state = np.array(curr_state) # MxN binary matrix
        
        # 'curr_state*prev_pred' gives MxN binary matrix of only those cells that
        # are predicted AND present in the current input. Adding 'net_state' to 
        # this gives binary MxN 'net_state' from line 144 above but with the 
        # predicted cells with value '2'. The next step is to find those columns
        # in 'curr_state*prev_pred + curr_state' with '2' as an entry and subtract 1.
        # The following 6 lines of code are computing eq. 1, pg. 6 in the proposal.
        
        # NOTE: Although the learning rules are designed to make the following
        # impossible, but even if it so happens that TWO DIFFERENT cells are predicted
        # in the same minicolumn at a particular time step, then the equation below
        # will make those cells become silent or active depending on whether that 
        # particular minicolumn is in the set of current timestep's input or not.
        # Hence, the equation is robust to such special cases.
        
        curr_state = curr_state*prev_pred + curr_state
        
        winning_cols = list(np.where(curr_input)[0])
        
        for j in winning_cols:
            mc = curr_state[:,j]
            if 2 in mc:
                curr_state[:,j] = curr_state[:,j] - 1
                
        # 'curr_pred' is MxN binary matrix holding predictions for current timetep
        curr_pred = self.get_onestep_prediction(curr_state)
        
        return curr_pred, curr_state
    
    
    def do_net_synaPermUpdate(self, prev_input=None, prev_state=None):
        
        #_______________________CASE I__________________________
        # When winning column is not predicted (as would happen in the 
        # initial stage after initialization of the network)
        
        winning_cols = list(np.where(prev_input)[0])
        
        # Collect all columns that are unpredicted (minicols with more than one 1s)
        unpredicted_cols = []
        for j in winning_cols:
            if prev_state[:,j].sum() > 1:
                unpredicted_cols.append(j)
                
        for j in unpredicted_cols:
            overlap = [] # 'overlap' will eventually be a list of np.arrays.
                         # shape: (self.M, <cell.n_dendrites>)
            
            for i in range(self.M):
                cell_synapses = self.net_arch[i,j].dendrites
                
                # 'cell_dendFloat' will be a float array of shape (<cell.n_dendrites>,)
                cell_dendFloat = dot_prod(prev_state,cell_synapses) 
                overlap.append(cell_dendFloat)
                
            
        
        #_______________________CASE II__________________________
        
        
        
        return None
    
    
    def intrinsic_plasticity(self):
        
        return None
    
    
    def get_NETWORK(self):
        """
        Returns the network architecture – MxN matrix of HTM_CELLs

        Returns
        -------
        MxN matrix of HTM_CELLs
        
        """
        return self.net_arch
    
    
    def prune_net_NegPermanences(self):
        """
        

        Returns
        -------
        None.

        """
        
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
