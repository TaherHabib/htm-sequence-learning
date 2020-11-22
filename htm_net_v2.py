"""
FILE DESCRIPTION:
------------------------

....


OTHER NOTES:
------------
Issue 001: 
    When a bursting column's capacity is found to be FULL, when growing a new dendrite in NO BEST MATCH condition.
    
Issue 002: 
    When a bursting column's capacity is found to be FULL, when growing a new dendrite in the condition where there was
    a best match found, but there were not enough capacity on the best matching dendrite to grow new synapses to the 
    previous timestep's winner cells.
    
Issue 003:
    When in a bursting column, a maximum overlap with the previous active cells of the network is found at two different
    places – either on different cells in a column, or on different dendrites of the same cell in that column.
    
"""

import numpy as np
import pandas as pd
import random 

from htm_cell_v2 import HTM_CELL
from ufuncs import dot_prod, get_idx_nonZeroElements


class HTM_NET():

    def __init__(self, numColumns=None, cellsPerColumn=None,
                 maxDendritesPerCell=None, maxSynapsesPerDendrite=None, 
                 nmdaThreshold=None, permThreshold=None, permInit=None, permInit_sd=None,
                 perm_decrement=None, perm_increment=None, perm_decay=None, perm_boost=None):
        
        self.M = cellsPerColumn # 8
        self.N = numColumns # 175
        
        self.maxDendritesPerCell = maxDendritesPerCell
        self.maxSynapsesPerDendrite = maxSynapsesPerDendrite # max. number of CONNECTED synapses on a 
                                                             # single dendritic segment.        
        
        # Initializing every cell in the network, i.e. setting up the dendrites for each cell.
        self.net_arch = np.empty([self.M, self.N], dtype=HTM_CELL)        
        for i in range(self.M):
            for j in range(self.N):
                self.net_arch[i,j] = HTM_CELL(self.M, self.N, self.maxDendritesPerCell, self.maxSynapsesPerDendrite,
                                              nmdaThreshold, permThreshold, permInit, permInit_sd,
                                              perm_decrement, perm_increment, perm_decay, perm_boost)        
        self.net_deadCells = []
        
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
        
        # MxN binary numpy array to store the predictive states of all cells.
        pred_state = np.zeros([self.M, self.N], dtype=np.int8)
        
        # Dictionary to store the index of the dendrites that led to the predictive states of the cell (i,j).
        # key: index of the cell, (i,j)
        # value: 1D numpy array of max. possible length <maxDendritesPerCell>                    
        dict_predDendrites = {} 
        
        for j in range(self.N):
            for i in range(self.M):
                
                cell_predictivity, predDendrites = self.net_arch[i,j].get_cell_predictivity(net_state)
                
                if cell_predictivity:
                    pred_state[i,j] = 1
                    dict_predDendrites[(i,j)] = predDendrites
                    
        return pred_state, dict_predDendrites
    
    
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
        curr_pred, curr_predDendrites = self.get_net_oneStepPrediction(curr_state)
        
        return curr_state, curr_pred, curr_predDendrites
    
    
    def update_net_synapticPermanences(self,
                                       curr_state=None, prev_state=None, 
                                       prev_pred=None, prev_predDendrites=None,
                                       prev_winnerCells=None):
        
        #----------------------------------------------------------------------
        # From currently active columns, collect all columns that are bursting,
        # i.e. unpredicted (minicols with all 1s) and correctly and 
        # incorrectly predicted (minicols with more than one 1).
        #----------------------------------------------------------------------
        
        active_cols = np.unique(np.where(curr_state)[1]) # np.array of length <k>
        
        predicted_cols = np.unique(np.where(prev_pred)[1]) # np.array of max. possible length <self.N>
                
        bursting_cols = [col for col in active_cols if curr_state[:, col].sum() == self.M]
        
        correctlyPredicted_cols = [col for col in active_cols if col not in bursting_cols]
        
        otherPredicted_cols = [col for col in predicted_cols if col not in correctlyPredicted_cols]
        
        #_______________________CASE I_________________________________________

        # When an active column is NOT PREDICTED, it will burst. 
        # (This would happen in the initial stages of learning)
        # ---------------------------------------------------------------------
        
        if len(bursting_cols) != 0:
            burstColumns_winnerCells, multiCellMaxOverlap = self.case_net_burstColumns(burstCols=bursting_cols,
                                                                                       prev_state=prev_state,
                                                                                       prev_winnerCells=prev_winnerCells)
    
        #_______________________CASE II________________________________________
        
        # When winning column IS CORRECTLY PREDICTED (can have more than 1 
        # predicted cells)
        # ---------------------------------------------------------------------
        
        if len(correctlyPredicted_cols) != 0:
            reinforcePredictedColumns_winnerCells = self.case_net_reinforcePredictedColumns(corrPredictedCols=correctlyPredicted_cols,
                                                                                            prev_state=prev_state,
                                                                                            prev_pred=prev_pred,
                                                                                            prev_predDendrites=prev_predDendrites)
        
        #_______________________CASE III_______________________________________
        
        # When a column IS Otherwise PREDICTED (can have more than 1 predicted 
        # cells)
        # ---------------------------------------------------------------------
        
        if len(otherPredicted_cols) != 0:
            self.case_net_decayPredictedColumns(otherPredictedCols=otherPredicted_cols,
                                                prev_state=prev_state,
                                                prev_pred=prev_pred,
                                                prev_predDendrites=prev_predDendrites)
        
        
        winnerCells = np.zeros([self.M, self.N], dtype=np.int8)
        
        for cell_idx in (burstColumns_winnerCells + reinforcePredictedColumns_winnerCells):
            winnerCells[cell_idx] = 1
        
        
        return winnerCells, multiCellMaxOverlap
    
    
    def case_net_burstColumns(self, burstCols=None, prev_state=None, prev_winnerCells=None):
        """
        

        Parameters
        ----------
        burstCols : TYPE, optional
            DESCRIPTION. The default is None.
        prev_state : TYPE, optional
            DESCRIPTION. The default is None.
        prev_winnerCells : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        winner_cells : TYPE
            DESCRIPTION.
        multiCellMaxOverlap : TYPE
            DESCRIPTION.

        """
        
        multiCellMaxOverlap = False
        winner_cells = []
        
        for j in burstCols:

            maxOverlapScore = -100
            bestMatchingCellDendrite = (None, None)
            cells_numUnusedDendrites = [] # this will be an array of <M> elements with integer entries
                                          # between 0 and <maxDendritesPerCell>.
            #--------------------------------------------------------------------------------------------------------------------
            for i in range(self.M):
                
                cells_numUnusedDendrites.append(self.net_arch[i,j].get_cell_numUnusedDendrites())
                
                for dendrite_idx in range(self.maxDendritesPerCell):
                    if self.net_arch[i,j].dendrites[dendrite_idx] is None:
                        continue
                    else:
                        # 'prev_state' is a binary and 'dendrites[idx]' is a float (numpy) MxN matrix. Hence,
                        # 'cell_dendriteOverlap' will be a numpy array of length 1, with a single float value,
                        # greater than 0.
                        cell_dendriteOverlap = dot_prod(prev_state, self.net_arch[i,j].dendrites[dendrite_idx])
                        
                        if cell_dendriteOverlap >= maxOverlapScore:        
                            if cell_dendriteOverlap == maxOverlapScore:
                                multiCellMaxOverlap = True
                                print(f'Issue 003 encountered! For column: {j}')
                    
                            maxOverlapScore = cell_dendriteOverlap
                            bestMatchingCellDendrite = (i, dendrite_idx)                
            #--------------------------------------------------------------------------------------------------------------------
                
            #__________CASE: NO MATCHING DENDRITE IS FOUND_____________________
            
            # Grow a new one on least used cell in the column.
            if bestMatchingCellDendrite[0] is None:
                
                # Checking for the condition where all <maxDendritesPerCell> dendrtes
                # are used on every cell and still NO Match was found.
                if np.amax(cells_numUnusedDendrites) == 0:
                    print(f'Issue 001 encountered! For column: {j}')  
                else:
                    # Select randomly one cell from the set of all cells with equal number of UnusedDendrites
                    cellIdx_leastUsedCell = random.choice(np.where(cells_numUnusedDendrites == np.amax(cells_numUnusedDendrites))[0])
                    # Grow a new dendrite on this cell with connections to 'prev_WinnerCells'
                    self.net_arch[cellIdx_leastUsedCell, j].grow_cell_newDendrite(prev_winnerCells)               
                    
                    winner_cells.append((cellIdx_leastUsedCell, j))
                    
            #__________CASE: MATCHING DENDRITE FOUND___________________________    
            
            else:
                # Reinforce connections with active cells in 'prev_state' and punish all other 
                # synaptic connections.
                self.net_arch[bestMatchingCellDendrite[0], j].update_cell_dendritePermanences(dendrite_idx=bestMatchingCellDendrite[1],
                                                                                              prev_state=prev_state)
                winner_cells.append((bestMatchingCellDendrite[0], j))
                
                #________Growing New Synapses To 'prev_winnerCells'____________
                # 1. When there is capacity on dendrite to grow new synapses.
                if self.net_arch[bestMatchingCellDendrite[0], j].get_cell_numSynapsesOnDendrite(bestMatchingCellDendrite[1]) < self.maxSynapsesPerDendrite:
                    self.net_arch[bestMatchingCellDendrite[0], j].grow_cell_newSynapsesOnDendrite(dendrite_idx=bestMatchingCellDendrite[1],
                                                                                                  prev_winnerCells=prev_winnerCells)
                # 2. When the dendrite has NO capacity to grow new synapses.
                else:
                    # Check if there is a significant overlap between the reinforced
                    # synapses from above and 'prev_winnerCells'. If the overlap is less 
                    # than the cell's NMDA threshold, then grow a new dendrite, else do not!
                    list_reinforcedSynapses = get_idx_nonZeroElements(self.net_arch[bestMatchingCellDendrite[0], j].dendrites[bestMatchingCellDendrite[1]]*prev_state)
                    list_prev_winnerCells = get_idx_nonZeroElements(prev_winnerCells)
                    
                    num_overlapCellIdx = len(set(list_reinforcedSynapses).intersection(list_prev_winnerCells))
                    
                    if num_overlapCellIdx < self.net_arch[bestMatchingCellDendrite[0], j].nmdaThreshold:
                        if np.amax(cells_numUnusedDendrites) == 0:
                            print(f'Issue 002 encountered! For column: {j}')
                        else:
                            cellIdx_leastUsedCell = random.choice(np.where(cells_numUnusedDendrites == np.amax(cells_numUnusedDendrites))[0])
                            self.net_arch[cellIdx_leastUsedCell, j].grow_cell_newDendrite(prev_winnerCells)               
                                
                            winner_cells.remove((bestMatchingCellDendrite[0], j))
                            winner_cells.append((cellIdx_leastUsedCell, j))
        
        return winner_cells, multiCellMaxOverlap            
                    
        
    def case_net_reinforcePredictedColumns(self, corrPredictedCols=None, prev_state=None, 
                                           prev_pred=None, prev_predDendrites=None):
        """
        

        Parameters
        ----------
        corrPredictedCols : TYPE, optional
            DESCRIPTION. The default is None.
        prev_state : TYPE, optional
            DESCRIPTION. The default is None.
        prev_pred : TYPE, optional
            DESCRIPTION. The default is None.
        prev_predDendrites : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        winner_cells : TYPE
            DESCRIPTION.

        """
        
        winner_cells = []
        
        for j in corrPredictedCols:
            
            # extract the i-indices of all the CORRECTLY predicted cells in the column
            cells_i = np.where(prev_pred[:,j])[0]
            
            # Reinforce the active dendrites for all of the predicted cells in the minicolumn.
            for i in cells_i:
                
                winner_cells.append((i,j))
                
                # for indices of all dendrites that led to cell's prediction.
                for dendrite_idx in prev_predDendrites[(i,j)]:
                    
                    self.net_arch[i,j].update_cell_dendritePermanences(dendrite_idx=dendrite_idx,
                                                                       prev_state=prev_state)
        
        return winner_cells
    
    
    def case_net_decayPredictedColumns(self, otherPredictedCols=None, prev_state=None,
                                       prev_pred=None, prev_predDendrites=None):
        """
        

        Parameters
        ----------
        otherPredictedCols : TYPE, optional
            DESCRIPTION. The default is None.
        prev_state : TYPE, optional
            DESCRIPTION. The default is None.
        prev_pred : TYPE, optional
            DESCRIPTION. The default is None.
        prev_predDendrites : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """ 
        
        for j in otherPredictedCols:
            
            # extract the i-indices of all the OTHER predicted cells in the column
            cells_i = np.where(prev_pred[:,j])[0]
            
            # Punish the active dendrites for all of the OTHER predicted cells in the minicolumn.
            for i in cells_i:
                
                # for indices of all dendrites that led to cell's prediction.
                for dendrite_idx in prev_predDendrites[(i,j)]:
                    
                    self.net_arch[i,j].update_cell_dendritePermanences(dendrite_idx=dendrite_idx,
                                                                       prev_state=prev_state,
                                                                       decay_only=True)
        
        return

        
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
                for dendrite in self.net_arch[i,j].get_cell_allDendrites():
                    if dendrite is None:
                        continue
                    else:
                        dendrite[dendrite<0] = 0.0
                        dendrite[dendrite>1] = 1.0
        return
    
    
    def net_IP(self):
        
        return None
    
    
    def get_NETWORK(self, char_onehot='all'):
        """
        Returns the network architecture – MxN matrix of HTM_CELLs

        Returns
        -------
        MxN matrix of HTM_CELLs
        
        """
        
        if char_onehot == 'all':
            return  self.net_arch
        
        else:
            return self.net_arch[:, np.where(char_onehot)[0]]
        

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