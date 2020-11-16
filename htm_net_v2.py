
import numpy as np
import pandas as pd


from htm_cell import HTM_CELL
from ufuncs import dot_prod

# ========================DEFINING HTM NETWORK=================================

class HTM_NET():

    def __init__(self, numColumns=None, cellsPerColumn=None,
                 maxDendritesPerCell=None, maxSynapsesPerDendrite=None, 
                 nmdaThreshold=None, permThreshold=None, permInit=None, permInit_sd=None,
                 perm_decrement=None, perm_increment=None, perm_decay=None, perm_boost=None):
        
        self.M = cellsPerColumn # 8
        self.N = numColumns # 175
        
        self.perm_decrement = perm_decrement
        self.perm_increment = perm_increment
        self.perm_decay = perm_decay
        self.perm_boost = perm_boost
        
        
        # Initializing every cell in the network, i.e. setting up the dendrites for each cell.
        self.net_arch = np.empty([self.M, self.N], dtype=HTM_CELL)        
        for i in range(self.M):
            for j in range(self.N):
                self.net_arch[i,j] = HTM_CELL(self.M, self.N, maxDendritesPerCell, maxSynapsesPerDendrite,
                                              nmdaThreshold, permThreshold, permInit, permInit_sd)        
        return
    
    
    def get_net_oneStepPrediction(self, net_state=None):
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
        # get activated to the same activity pattern in the population. BUT 
        # this assumption may be broken (?). CHECK. (ANS: So far there has been 
        # no evidence that this assumption may break.)
        
        # MxN binary numpy array to store the predictive states of all cells.
        pred_state = np.zeros([self.M, self.N], dtype=np.int8)
        
        # List to store the index of the dendrites that led to the predictive 
        # states of the cell (i,j). Elements are 2-tuples with:
        # first entry: index of the cell, (i,j)
        # second entry: 1D numpy array of max. length <cell.n_dendrites>                    
        pred_dendrites = {} 
        
        for j in range(self.N):
            for i in range(self.M):
                cell_connectedSynapses = self.net_arch[i,j].get_cell_connectedSynapses() 
                
                if cell_connectedSynapses is not None:    
                    # 'cell_dendActivity' will be a boolean array of shape (<cell.n_dendrites>,)
                    cell_dendActivity = dot_prod(net_state,cell_connectedSynapses)>self.net_arch[i,j].nmda_th
                
                    # if any denrite of the cell is active, then the cell becomes predictive.
                    if any(cell_dendActivity):
                        pred_state[i,j] = 1
                        pred_dendrites[(i,j)] = np.where(cell_dendActivity)[0] # Dict 'value' is 1D numpy array 
                                                                               # of max. length <cell.n_dendrites>
                        
        return pred_state, pred_dendrites
    
    
    def get_net_LRDPrediction(self):
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
        curr_state = np.array(curr_state, dtype=np.int8) # MxN binary matrix
        
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
        # In other words, the equation is robust to such special cases.
        
        curr_state = curr_state*prev_pred + curr_state
        
        winning_cols = np.where(curr_input)[0]
        
        for j in winning_cols:
            if 2 in curr_state[:,j]:
                curr_state[:,j] -= 1 
                
        # 'curr_pred' is MxN binary matrix holding predictions for current timetep
        curr_pred, curr_pred_dend = self.get_net_oneStepPrediction(curr_state)
        
        return curr_state, curr_pred, curr_pred_dend
    
    
    def update_net_synapticPermanences(self,
                                       curr_state=None, prev_state=None, 
                                       prev_pred=None, prev_pred_dend=None,
                                       prev_winner_cells=None):
        
        #----------------------------------------------------------------------
        # From currently active columns, collect all columns that are bursting,
        # i.e. unpredicted (minicols with all 1s) and correctly and 
        # incorrectly predicted (minicols with more than one 1).
        #----------------------------------------------------------------------
        
        active_cols = np.unique(np.where(curr_state)[1]) # np.array of length <k>
        
        # 'predicted_cols' will be np.array of max. possible length <self.N>
        predicted_cols = np.unique(np.where(prev_pred)[1]) 
                
        bursting_cols = [col for col in active_cols if curr_state[:, col].sum() == self.M]
        
        correctlyPredicted_cols = [col for col in active_cols if col not in bursting_cols]
        
        incorrectlyPredicted_cols = [col for col in predicted_cols if col not in correctlyPredicted_cols]
        
        #_______________________CASE I________________________________________
        
        # When an active column is NOT PREDICTED, it will burst. 
        # This would happen in the initial stages of learning)
        # ---------------------------------------------------------------------
        
        if len(bursting_cols) != 0:
            winner_cells, multiCellMaxOverlap = self.net_burstColumns(bursting_cols, 
                                                                      prev_state,
                                                                      prev_winner_cells)
                
    
        #_______________________CASE II________________________________________
        
        # When winning column IS CORRECTLY PREDICTED (can have more than 1 
        # predicted cells)
        # ---------------------------------------------------------------------
        
        for j in correctlyPredicted_cols:
            
            # extract the i-indices of all the CORRECTLY predicted cells in the column
            cells_i = np.where(prev_pred[:,j])[0]
            
            # Reinforce the active dendrites for all of the predicted cells in the minicolumn.
            for i in cells_i:
                
                # for indices of all dendrites that led to cell's prediction.
                for d in prev_pred_dend[i,j]:
                    self.net_arch[i,j].dendrites[d] = self.net_arch[i,j].dendrites[d] + self.perm_increment*prev_state 
                    - self.perm_decrement*self.net_arch[i,j].dendrites[d]
            
        
        #_______________________CASE III_______________________________________
        
        # When a column IS WRONGLY PREDICTED (can have more than 1 predicted 
        # cells)
        # ---------------------------------------------------------------------
        
        for j in incorrectlyPredicted_cols:
            
            # extract the i-indices of all the WRONGLY predicted cells in the column
            cells_i = np.where(prev_pred[:,j])[0]
            
            # Punish the active dendrites for all of the wrongly predicted cells in the minicolumn.
            for i in cells_i:
                
                # for indices of all dendrites that led to cell's wrong prediction.
                for d in prev_pred_dend[i,j]:
                    self.net_arch[i,j].dendrites[d] = self.net_arch[i,j].dendrites[d] - self.perm_decay*prev_state
                    
        
        #_______________________BOOSTING_______________________________________
        
        # Boosting of all the synaptic permanence values of all the cells with 
        # low activity and/or predictivity.
        # ---------------------------------------------------------------------
        
        
        return multiCellMaxOverlap
    
    
    def net_burstColumns(self, bursting_cols, prev_state, prev_winner_cells):
        
        # the best matching segment would be selected based on...
        # 'winner_cells' are based on...
        
        multiCellMaxOverlap = False
        winner_cells = []
        
        
                
        for j in bursting_cols:
            
            overlap = [] # 'overlap' will eventually be a list of shape: (self.M, <cell.n_dendrites>)
            maxOverlapScore = -1
            
            for i in range(self.M):                
                
                # 'cell_dendFloat' will be a float64 array of Max. shape (<cell.n_dendrites>,)
                cell_dendOverlap = dot_prod(prev_state, self.net_arch[i,j].dendrites) 
                overlap.append(cell_dendOverlap)
                
                
                if cell_dendOverlap is None:
                    # Make a new segment
                    ...
                    
                
                    
                    
                
            
            # NOTE: Ideally, the maximum value in overlap should occur at a unique position.
            # Single cell's single dendrite. However, sometimes the maximum value in 'overlap' 
            # can be at two different places (extremely rarely):
            # 1. In two different cells, out of the M cells in the minicolumn.
            # 2. In the same cell, but with two different dendrites.
            
            max_overlap_cell = np.where(overlap==np.amax(overlap))[0]
            max_overlap_dendrite = np.where(overlap==np.amax(overlap))[1]
            
            if len(max_overlap_cell) > 1: # (The RARE CASE)
                
                multi_cell_MaxOverlap = True
                
                # 'MaxOverlap_cell_dend' is a MxN permanence value matrix.
                # In the case when there are more than 1 cells with a max overlap with 
                # 'prev_state', I take the first one (index [0]), reinforce it, and 
                # simply re-initialize the other cells/dendrites.
                MaxOverlap_cell_dend = self.net_arch[max_overlap_cell[0],j].dendrites[max_overlap_dendrite[0]]
                
                # Decrement all synapses by p- and increment active synapses by p+
                self.net_arch[max_overlap_cell[0],j].dendrites[max_overlap_dendrite[0]] = MaxOverlap_cell_dend + self.perm_increment*prev_state 
                - self.perm_decrement*MaxOverlap_cell_dend
                
                # Re-initializing other cells within Max. Overlap
                for d in range(1,len(max_overlap_cell)):
                    self.net_arch[max_overlap_cell[d],j].dendrites[max_overlap_dendrite[d]] = \
                        np.random.normal(loc=self.perm_init, scale=0.01, size=[self.M, self.N])                 
                    
            else:
                
                MaxOverlap_cell_dend = self.net_arch[max_overlap_cell[0],j].dendrites[max_overlap_dendrite[0]]
                
                # Increment active synapses by p+ and Decrement all synapses by p-
                self.net_arch[max_overlap_cell[0],j].dendrites[max_overlap_dendrite[0]] = MaxOverlap_cell_dend + self.perm_increment*prev_state 
                - self.perm_decrement*MaxOverlap_cell_dend
        
        return winner_cells, multiCellMaxOverlap
    
    
    def prune_net_permanences(self):
        """
        Prunes Negative Permanence Values (setting them to 0).
        Prunes Positive Permanence Values greater than 1 (setting them to 1).
                    
        Returns
        -------
        None.

        """
        
        for i in range(self.M):
            for j in range(self.N):
                dendrites = self.net_arch[i,j].get_cell_dendrites()
                
                if len(dendrites) != 0:
                    dendrites[dendrites<0] = 0.0
                    dendrites[dendrites>1] = 1.0
                    
                else:
                    continue
                
        return
    
    
    def net_IP(self):
        
        return None
    
    
    def get_NETWORK(self, char_minicols='all'):
        """
        Returns the network architecture – MxN matrix of HTM_CELLs

        Returns
        -------
        MxN matrix of HTM_CELLs
        
        """
        
        if char_minicols == 'all':
            return  self.net_arch
        
        else:
            return self.net_arch[:, np.where(char_minicols)[0]]
        

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
