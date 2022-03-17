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

Issue 004:
    
Issue 005:
    
    
"""

import numpy as np
import pandas as pd
import random 

from htm_cell_v1 import HTM_CELL
from ..htm.utils import dot_prod, get_idx_nonZeroElements


class HTM_NET():

    def __init__(self, numColumns=None, cellsPerColumn=None,
                 maxDendritesPerCell=None, maxSynapsesPerDendrite=None, 
                 nmdaThreshold=None, permThreshold=None, 
                 permInit=None, permInit_sd=None,
                 perm_decrement=None, perm_increment=None, 
                 perm_decay=None, perm_boost=None):
        
        self.M = cellsPerColumn # 8
        self.N = numColumns # 175
        
        self.maxDendritesPerCell = maxDendritesPerCell
        self.maxSynapsesPerDendrite = maxSynapsesPerDendrite 
        # max. number of CONNECTED synapses on a single dendritic segment.
        
        self.nmdaThreshold = nmdaThreshold
        self.learningThreshold = 7
        
        #______________________________________________________________________
        # Initializing every cell in the network, i.e. setting up the dendrites 
        # for each cell.
        
        self.net_arch = np.empty([self.M, self.N], dtype=HTM_CELL)        
        for i in range(self.M):
            for j in range(self.N):
                self.net_arch[i,j] = HTM_CELL(cellsPerColumn=self.M, numColumns=self.N, 
                                              maxDendritesPerCell=self.maxDendritesPerCell, maxSynapsesPerDendrite=self.maxSynapsesPerDendrite,
                                              nmdaThreshold=nmdaThreshold, permThreshold=permThreshold, 
                                              permInit=permInit, permInit_sd=permInit_sd,
                                              perm_decrement=perm_decrement, perm_increment=perm_increment, 
                                              perm_decay=perm_decay, perm_boost=perm_boost)
        
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
        
        pred_state = np.zeros([self.M, self.N], dtype=np.int8)
        # MxN binary numpy array to store the predictive states of all cells.
        
        dict_predDendrites = {} 
        # Dictionary to store the index of the dendrites that led to the 
        # predictive states of the cell (i,j).
        # key: index of the cell, (i,j)
        # value: 1D numpy array of max. possible length <maxDendritesPerCell> 
        # with integer entries between 0 and <maxDendritesPerCell>-1.
        
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
        
        #______________________________________________________________________
        # Computing net state such that all minicolumns with current inputs 
        # are fully activated.
        for m in range(self.M):
            curr_state.append(curr_input)
        curr_state = np.array(curr_state, dtype=np.int8) # MxN binary matrix
        
        # 'curr_state*prev_pred' gives MxN binary matrix of only those cells 
        # that are predicted AND present in the current input. Adding 
        # 'net_state' to this gives binary MxN 'net_state' from line 144 above
        # but with the predicted cells with value '2'. The next step is to find 
        # those columnsin 'curr_state*prev_pred + curr_state' with '2' as an 
        # entry and subtract 1. The following 6 lines of code are computing 
        # eq. 1, pg. 6 in the proposal.
        
        curr_state = curr_state*prev_pred + curr_state
        
        winning_cols = np.where(curr_input)[0]
        
        for j in winning_cols:
            if 2 in curr_state[:,j]:
                curr_state[:,j] -= 1 
        
        # NOTE: Although the learning rules are designed to make the following
        # impossible, but even if it so happens that TWO DIFFERENT cells are 
        # predicted in the same minicolumn at a particular time step, then the 
        # equation below will make those cells become silent or active 
        # depending on whether that particular minicolumn is in the set of 
        # current timestep's input or not. In other words, the equation is 
        # robust to such special cases.
        #______________________________________________________________________        
    
        curr_pred, curr_predDendrites = self.get_net_oneStepPrediction(curr_state)        
        # 'curr_pred' is a binary matrix with predictions for current timetep.
        
        return curr_state, curr_pred, curr_predDendrites
    
    
    def update_net_synapticPermanences(self,
                                       curr_state=None, prev_state=None, 
                                       prev_pred=None, prev_predDendrites=None,
                                       prev_winnerCells=None):
        
        #______________________________________________________________________
        # From currently active columns, collect all columns that are bursting,
        # i.e. unpredicted (minicols with all 1s) and correctly and 
        # incorrectly predicted (minicols with more than one 1).
        
        active_cols = np.unique(np.where(curr_state)[1]) 
        # np.array of length <k>
        
        predicted_cols = np.unique(np.where(prev_pred)[1]) 
        # np.array of max. possible length <self.N>
                
        bursting_cols = [col for col in active_cols if curr_state[:, col].sum() == self.M]
        
        correctlyPredicted_cols = [col for col in active_cols if col not in bursting_cols]
        
        otherPredicted_cols = [col for col in predicted_cols if col not in correctlyPredicted_cols]
        
        burstColumns_winnerCells = []
        reinforcePredictedColumns_winnerCells = []
        
        #_______________________CASE I_________________________________________

        # When an active column is NOT PREDICTED, it will burst. 
        # (This would happen in the initial stages of learning)
        # ---------------------------------------------------------------------
        
        if len(bursting_cols) != 0:
            burstColumns_winnerCells = self.case_net_burstColumns(burstCols=bursting_cols,
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
        
        
        return winnerCells
    
    
    def case_net_burstColumns(self, burstCols=None, prev_state=None, prev_winnerCells=None):
        
        winner_cells = []
        
        for j in burstCols:
        
            #__________________________________________________________________    
            # The following checks if two different contexts for a single letter
            # were predicted in the previous timestep. 
            # NOTE: Multiple predictions for the SAME letter are not possible!
            # REPORT ERROR AND BREAK THE LEARNING PROCESS!
            if np.count_nonzero(prev_winnerCells) > self.maxSynapsesPerDendrite:
                print(color.RED, color.BOLD, '::::::::::::::::::::::::::::::::::Issue 007 enct. in col: ', j, \
                      '::::::::::::::::::::::::::::::::::', color.END, color.END)
                break
            
            bestMatch_CellDendrite = (None, None)
            cells_numUnusedDendrites = [] 
            # this will be an array of <M> elements with integer entries
            # between 0 and <maxDendritesPerCell>.
            
            cells_overlapScorePerDendrite = [] 
            # this will be an array of <M> elements with entries as other 
            # arrays of length <maxDendritesPerCell>. Each entry in the nested
            # array will be either -1 or a float value >= 0.
            
            cells_overlapSynapsesPerDendrite = [] 
            # this will be an array of <M> elements with entries as other 
            # arrays of length <maxDendritesPerCell>. Each entry in the nested
            # array will be an integer >= 0.
            
  
            #------------------------------------------------------------------
            for i in range(self.M):
                
                cells_numUnusedDendrites.append(self.net_arch[i,j].get_cell_numUnusedDendrites())
                
                cell_overlapScorePerDendrite = []
                cell_overlapSynapsesPerDendrite = []
                
                for dendrite_idx in range(self.maxDendritesPerCell):
                    if self.net_arch[i,j].dendrites[dendrite_idx] is None:
                        dendriteOverlapScore = -1
                        dendriteOverlap_numSynapses = 0
                    else:
                        dendriteOverlapScore = dot_prod(prev_state, self.net_arch[i,j].dendrites[dendrite_idx])
                        dendriteOverlap_numSynapses = np.count_nonzero(prev_state*self.net_arch[i,j].dendrites[dendrite_idx])
                        # 'prev_state' is a binary and 'dendrites[idx]' is a 
                        # float (numpy) MxN matrix. Hence,'dendriteOverlapScore' 
                        # will be a numpy array of length 1, with a single 
                        # float value, greater than or equal to 0.
            
                    cell_overlapScorePerDendrite.append(dendriteOverlapScore)
                    cell_overlapSynapsesPerDendrite.append(dendriteOverlap_numSynapses)
                
                cells_overlapScorePerDendrite.append(cell_overlapScorePerDendrite)
                cells_overlapSynapsesPerDendrite.append(cell_overlapSynapsesPerDendrite)
            
            #__________________________________________________________________
            # Finding the best matching segment using 
            # 'cells_overlapScorePerDendrite'.
            
            if np.amax(cells_overlapSynapsesPerDendrite) < self.learningThreshold:
                bestMatch_CellDendrite = (None, None)
            
            else:
                bestMatch = np.where(cells_overlapScorePerDendrite == np.amax(cells_overlapScorePerDendrite))
            
                # REPORT ERROR!    
                if len(bestMatch[0]) > 1:
                    print(color.RED, 'Issue 003 enct. in col: ',j, \
                          '. Best Match cells: ', list(bestMatch[0]), \
                          '. Best Match dendrites: ', list(bestMatch[1]), \
                          '. Overlap score: ', np.amax(cells_overlapScorePerDendrite), color.END)
                else:
                    bestMatch_CellDendrite = (bestMatch[0][0], bestMatch[1][0])
                    bestMatch_Score = np.amax(cells_overlapScorePerDendrite)
                
            
            #__________CASE: NO MATCHING DENDRITE IS FOUND_____________________
            
            # Grow a new one on least used cell in the column.
            if bestMatch_CellDendrite[0] is None:
                
                # Checking for the condition where all <maxDendritesPerCell> 
                # dendrtes are used on every cell and still NO Match was found.
                # REPORT ERROR!
                if np.amax(cells_numUnusedDendrites) == 0:
                    print(color.RED, color.BOLD, 'Issue 001 enct. in col: ', j, color.END, color.END)
                else:
                    # Select randomly one cell from the set of all cells with 
                    # equal number of UnusedDendrites
                    cellIdx_leastUsedCell = random.choice(np.where(cells_numUnusedDendrites == np.amax(cells_numUnusedDendrites))[0])
                    # Grow a new dendrite on this cell with connections to 
                    # 'prev_WinnerCells'
                    newDendrite_idx = self.net_arch[cellIdx_leastUsedCell, j].grow_cell_newDendrite(prev_winnerCells)               
                    print(color.GREEN, 'NO MATCH FOUND. Growing ', \
                          self.net_arch[cellIdx_leastUsedCell, j].get_cell_numSynapsesOnDendrite(newDendrite_idx), \
                          ' new synapses on cell {', cellIdx_leastUsedCell, j,'}; dendrite: ', newDendrite_idx, color.END)
                          
                    winner_cells.append((cellIdx_leastUsedCell, j))
                    
            
            #__________CASE: MATCHING DENDRITE FOUND___________________________    
            
            else:
                # Reinforce connections with active cells in 'prev_state' and 
                # punish all other synaptic connections.
                # Due to the previous restriction of having at least 
                # <learningThreshold> matching synapses for a dendritic segment 
                # to be counted as 'bestMatch', AT LEAST <learningThreshold> 
                # synapses will be reinforced at any given timestep.
                reinforcedSynapses, tot_numSynapses = self.net_arch[bestMatch_CellDendrite[0], j].update_cell_dendritePermanences(dendrite_idx=bestMatch_CellDendrite[1],
                                                                                                                                  prev_state=prev_state)
                print(color.GREEN, 'MATCH FOUND. Cell {', bestMatch_CellDendrite[0],',',j,'}; dendrite: ',bestMatch_CellDendrite[1], \
                      '. OverlapScore: ', bestMatch_Score, \
                      '. Reinforcing ', np.count_nonzero(reinforcedSynapses),' synapses. Existing synapses= ', tot_numSynapses, color.END)
                
                winner_cells.append((bestMatch_CellDendrite[0], j))
                
                bestMatch_dendriteCapacity = self.maxSynapsesPerDendrite - self.net_arch[bestMatch_CellDendrite[0], j].get_cell_numSynapsesOnDendrite(bestMatch_CellDendrite[1])
                # REPORT ERROR!
                if bestMatch_dendriteCapacity < 0:
                    print(color.RED, color.BOLD, 'Issue 004 enct. in cell {', bestMatch_CellDendrite[0],',',j,'}; dendrite: ',bestMatch_CellDendrite[1], color.END, color.END)
                
                #________Growing New Synapses To 'prev_winnerCells'____________
                
                # Check if ... #TODO
                list_reinforcedSynapses = get_idx_nonZeroElements(reinforcedSynapses)
                # will have at least <learningThreshold> elements
                
                list_prev_winnerCells = get_idx_nonZeroElements(prev_winnerCells)
                
                intersection_reinforcedPrevWinners = set(list_reinforcedSynapses).intersection(list_prev_winnerCells)
                newSynapsesToGrow = set(list_prev_winnerCells) - intersection_reinforcedPrevWinners
                remaining_winnerCells = np.zeros(shape=[self.M,self.N])
                for i in newSynapsesToGrow:
                    remaining_winnerCells[i] = 1
                
                # Check if there are far TOO MANY 'newSynapsesToGrow'!
                # REPORT ERROR!
                if len(newSynapsesToGrow) > self.maxDendritesPerCell:
                    print(color.RED, color.BOLD, 'Issue 005 enct. in col: ', j, color.END, color.END)    
            
                #--------------------------------------------------------------
                # 1. When there is capacity on 'bestMatch' dendrite to grow
                # new synapses.
                if len(newSynapsesToGrow) <= bestMatch_dendriteCapacity:
                    total_newSynapses = self.net_arch[bestMatch_CellDendrite[0], j].grow_cell_newSynapsesOnDendrite(dendrite_idx=bestMatch_CellDendrite[1],
                                                                                                                    prev_winnerCells=prev_winnerCells)
                    print(color.YELLOW, 'Growing ', total_newSynapses,' new Synapses.', color.END)                    
                    
                #elif len(newSynapsesToGrow) < self.nmdaThreshold and len(list_prev_winnerCells) < self.maxSynapsesPerDendrite:
                    
                
                # 2. When the 'bestMatch' dendrite has NO capacity to grow new 
                # synapses.
                else:
                
                    # There are cases where due to the bursting of all columns 
                    # in previous timestep, the 'reinforcedSynapses'(bestMatch) 
                    # are different than the 'prev_WinnerCells' entirely i.e., 
                    # the intersection set above might be empty. Therefore, it 
                    # is required to check whether there are other existing 
                    # dendrites in the minicolumn that have all the synaptic 
                    # connections to 'newSynapsesToGrow'. 
                    # If yes, then these existing synaptic connections are 
                    # reinforced. 
                    # Otherwise, a new Dendrite is grown to store 
                    # 'remaining_winnerCells' pattern on a randomly chosen 
                    # 'leastUsedCell'.                   
                    cellDendrites_ = []    
                    for i in range(self.M):
                        for dendrite_idx in range(self.maxDendritesPerCell):
                            if self.net_arch[i,j].dendrites[dendrite_idx] is None:
                                continue
                            else:
                                existing_synapticPerm = get_idx_nonZeroElements(self.net_arch[i,j].dendrites[dendrite_idx])
                                if newSynapsesToGrow - newSynapsesToGrow.intersection(existing_synapticPerm) == set():
                                    cellDendrites_.append((i, dendrite_idx))
                    
                    if cellDendrites_ == []:
                        # REPORT ERROR!
                        if np.amax(cells_numUnusedDendrites) == 0:
                            print(color.RED, color.BOLD, 'Issue 002 enct. in col: ', j, color.END, color.END)
                        
                        else:
                            cellIdx_leastUsedCell = random.choice(np.where(cells_numUnusedDendrites == np.amax(cells_numUnusedDendrites))[0])
                            newDendrite_idx = self.net_arch[cellIdx_leastUsedCell, j].grow_cell_newDendrite(remaining_winnerCells)               
                            print(color.YELLOW,'NO DENDRITIC CAPACITY AND EXISTING DENDRITE. Growing ', \
                                  self.net_arch[cellIdx_leastUsedCell, j].get_cell_numSynapsesOnDendrite(newDendrite_idx), \
                                  ' new synapses on cell {', cellIdx_leastUsedCell, j,'}; dendrite: ', newDendrite_idx, color.END)
                            
                            winner_cells.remove((bestMatch_CellDendrite[0], j))
                            winner_cells.append((cellIdx_leastUsedCell, j))
                            
                    else:
                        random.seed()
                        cellDendrite_idx = random.choice(cellDendrites_)
                        reinforcedSynapses_, _ = self.net_arch[cellDendrite_idx[0], j].update_cell_dendritePermanences(dendrite_idx=cellDendrite_idx[1],
                                                                                                                       prev_state=remaining_winnerCells)
                        print(color.DARKCYAN, 'NO DENDRITIC CAPACITY BUT EXISTING DENDRITE FOUND. Cell {', cellDendrite_idx[0],',',j,'}; dendrite: ',cellDendrite_idx[1], \
                              '. Reinforcing ', np.count_nonzero(reinforcedSynapses_),' synapses.', color.END)
                        
                        winner_cells.remove((bestMatch_CellDendrite[0], j))
                        winner_cells.append((cellDendrite_idx[0], j))
                        
                
        return winner_cells          
                
        
    def case_net_reinforcePredictedColumns(self, corrPredictedCols=None, prev_state=None, 
                                           prev_pred=None, prev_predDendrites=None):
        
        winner_cells = []
        
        for j in corrPredictedCols:
            
            # extract i-indices of all CORRECTLY predicted cells in the column
            cells_i = np.where(prev_pred[:,j])[0]
            
            #__________________________________________________________________
            # Reinforce active dendrites for all of predicted cells in the 
            # minicolumn.
            for i in cells_i:
                
                winner_cells.append((i,j))
                # for indices of all dendrites that led to cell's prediction.
                for dendrite_idx in prev_predDendrites[(i,j)]:
                    
                    reinforcedSynapses, tot_numSynapses = self.net_arch[i,j].update_cell_dendritePermanences(dendrite_idx=dendrite_idx,
                                                                                                             prev_state=prev_state)
                
                    #print(color.DARKCYAN, 'CORRECT PRED. Cell {', i,',',j,'}; dendrite: ', dendrite_idx, \
                     #     '. Reinforcing ', np.count_nonzero(reinforcedSynapses),' synapses. Existing synapses= ', tot_numSynapses, color.END)
                
                    
        return winner_cells
    
    
    def case_net_decayPredictedColumns(self, otherPredictedCols=None, prev_state=None,
                                       prev_pred=None, prev_predDendrites=None):
        
        for j in otherPredictedCols:
            
            # extract i-indices of all OTHER predicted cells in the column
            cells_i = np.where(prev_pred[:,j])[0]
            
            #__________________________________________________________________
            # Punish active dendrites for all of OTHER predicted cells in the 
            # minicolumn.
            for i in cells_i:
                
                # for indices of all dendrites that led to cell's prediction.
                for dendrite_idx in prev_predDendrites[(i,j)]:
                    
                    _, _ = self.net_arch[i,j].update_cell_dendritePermanences(dendrite_idx=dendrite_idx,
                                                                              prev_state=prev_state,
                                                                              decay_only=True)                
                
        return

        
    def update_net_dendriteDutyCycle(self):
        
        for i in range(self.M):
            for j in range(self.N):
                if self.net_arch[i,j].get_cell_numDendrites != 0:
                    
                    for dendrite_idx in range(self.maxDendritesPerCell):
                        if self.net_arch[i,j].dendrites_dutyCycle[dendrite_idx] is not None:
                            self.net_arch[i,j].dendrites_dutyCycle[dendrite_idx] -= 1
                        else:
                            continue
                        
                    for dendrite_idx in range(self.maxDendritesPerCell):
                        if self.net_arch[i,j].dendrites_dutyCycle[dendrite_idx] <= 0:
                            self.net_arch[i,j].dendrites_dutyCycle[dendrite_idx] = None
                            self.net_arch[i,j].dendrites[dendrite_idx] = None            
                        else:
                            continue                  
                    
                else:
                    
                    continue
                
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

        

class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'



# ==========================ROUGH==============================================


# 1. When there is capacity on 'bestMatch' dendrite to grow
# =============================================================================
# # new synapses.
# if len(newSynapsesToGrow) <= bestMatch_dendriteCapacity:
#     total_newSynapses = self.net_arch[bestMatch_CellDendrite[0], j].grow_cell_newSynapsesOnDendrite(dendrite_idx=bestMatch_CellDendrite[1],
#                                                                                                     prev_winnerCells=prev_winnerCells)
#     print(color.YELLOW, 'Growing ', total_newSynapses,' new Synapses.', color.END)                    
#     
# # 2. When the 'bestMatch' dendrite has NO capacity to grow new 
# # synapses.
# else:
# 
#     # There are cases where due to the bursting of all columns 
#     # in previous timestep, the 'reinforcedSynapses'(bestMatch) 
#     # are different than the 'prev_WinnerCells' entirely i.e., 
#     # the intersection set above might be empty. Therefore, it 
#     # is required to check whether there are other existing 
#     # dendrites in the minicolumn that have all the synaptic 
#     # connections to 'newSynapsesToGrow'. 
#     # If yes, then these existing synaptic connections are 
#     # reinforced. 
#     # Otherwise, a new Dendrite is grown to store 
#     # 'remaining_winnerCells' pattern on a randomly chosen 
#     # 'leastUsedCell'.                   
#     cellDendrites_ = []    
#     for i in range(self.M):
#         for dendrite_idx in range(self.maxDendritesPerCell):
#             if self.net_arch[i,j].dendrites[dendrite_idx] is None:
#                 continue
#             else:
#                 existing_synapticPerm = get_idx_nonZeroElements(self.net_arch[i,j].dendrites[dendrite_idx])
#                 if newSynapsesToGrow - newSynapsesToGrow.intersection(existing_synapticPerm) == set():
#                     cellDendrites_.append((i, dendrite_idx))
#     
#     if cellDendrites_ == []:
#         # REPORT ERROR!
#         if np.amax(cells_numUnusedDendrites) == 0:
#             print(color.RED, color.BOLD, 'Issue 002 enct. in col: ', j, color.END, color.END)
#         
#         else:
#             cellIdx_leastUsedCell = random.choice(np.where(cells_numUnusedDendrites == np.amax(cells_numUnusedDendrites))[0])
#             newDendrite_idx = self.net_arch[cellIdx_leastUsedCell, j].grow_cell_newDendrite(remaining_winnerCells)               
#             print(color.YELLOW,'NO DENDRITIC CAPACITY AND EXISTING DENDRITE. Growing ', \
#                   self.net_arch[cellIdx_leastUsedCell, j].get_cell_numSynapsesOnDendrite(newDendrite_idx), \
#                   ' new synapses on cell {', cellIdx_leastUsedCell, j,'}; dendrite: ', newDendrite_idx, color.END)
#             
#             winner_cells.remove((bestMatch_CellDendrite[0], j))
#             winner_cells.append((cellIdx_leastUsedCell, j))
#             
#     else:
#         random.seed()
#         cellDendrite_idx = random.choice(cellDendrites_)
#         reinforcedSynapses_, _ = self.net_arch[cellDendrite_idx[0], j].update_cell_dendritePermanences(dendrite_idx=cellDendrite_idx[1],
#                                                                                                        prev_state=remaining_winnerCells)
#         print(color.DARKCYAN, 'NO DENDRITIC CAPACITY BUT EXISTING DENDRITE FOUND. Cell {', cellDendrite_idx[0],',',j,'}; dendrite: ',cellDendrite_idx[1], \
#               '. Reinforcing ', np.count_nonzero(reinforcedSynapses_),' synapses.', color.END)
#         
#         winner_cells.remove((bestMatch_CellDendrite[0], j))
#         winner_cells.append((cellDendrite_idx[0], j))
# 
# =============================================================================

# =============================================================================
#     def case_net_burstColumns(self, burstCols=None, prev_state=None, prev_winnerCells=None):
#         
#         winner_cells = []
#         
#         for j in burstCols:
# 
#             maxOverlapScore = -100
#             bestMatch_CellDendrite = (None, None)
#             cells_numUnusedDendrites = [] 
#             # this will be an array of <M> elements with integer entries
#             # between 0 and <maxDendritesPerCell>.
#             
#             #------------------------------------------------------------------
#             for i in range(self.M):
#                 
#                 cells_numUnusedDendrites.append(self.net_arch[i,j].get_cell_numUnusedDendrites())
#                 
#                 for dendrite_idx in range(self.maxDendritesPerCell):
#                     if self.net_arch[i,j].dendrites[dendrite_idx] is None:
#                         continue
#                     else:
#                         cell_dendriteOverlap = dot_prod(prev_state, self.net_arch[i,j].dendrites[dendrite_idx])
#                         # 'prev_state' is a binary and 'dendrites[idx]' is a float (numpy) MxN matrix. Hence,
#                         # 'cell_dendriteOverlap' will be a numpy array of length 1, with a single float value,
#                         # greater than or equal to 0.
#                         
#                         
#                         if cell_dendriteOverlap >= maxOverlapScore and cell_dendriteOverlap!=0:        
#                             if cell_dendriteOverlap == maxOverlapScore:
#                                 print(f'Issue 003 encountered! For cell: {i, j} with overlap: {cell_dendriteOverlap}')
#                     
#                             maxOverlapScore = cell_dendriteOverlap
#                             bestMatch_CellDendrite = (i, dendrite_idx)                
#                             
#                         else:
#                             continue
#                         
#             #------------------------------------------------------------------
#                 
#             #__________CASE: NO MATCHING DENDRITE IS FOUND_____________________
#             
#             # Grow a new one on least used cell in the column.
#             if bestMatch_CellDendrite[0] is None:
#                 
#                 # Checking for the condition where all <maxDendritesPerCell> dendrtes
#                 # are used on every cell and still NO Match was found.
#                 if np.amax(cells_numUnusedDendrites) == 0:
#                     print(f'Issue 001 encountered! For column: {j}')  
#                 else:
#                     # Select randomly one cell from the set of all cells with equal number of UnusedDendrites
#                     cellIdx_leastUsedCell = random.choice(np.where(cells_numUnusedDendrites == np.amax(cells_numUnusedDendrites))[0])
#                     # Grow a new dendrite on this cell with connections to 'prev_WinnerCells'
#                     self.net_arch[cellIdx_leastUsedCell, j].grow_cell_newDendrite(prev_winnerCells)               
#                     
#                     winner_cells.append((cellIdx_leastUsedCell, j))
#                     
#             #__________CASE: MATCHING DENDRITE FOUND___________________________    
#             
#             else:
#                 # Reinforce connections with active cells in 'prev_state' and punish all other 
#                 # synaptic connections.
#                 self.net_arch[bestMatch_CellDendrite[0], j].update_cell_dendritePermanences(dendrite_idx=bestMatch_CellDendrite[1],
#                                                                                                prev_state=prev_state)
#                 winner_cells.append((bestMatch_CellDendrite[0], j))
#                 
#                 
#                 #________Growing New Synapses To 'prev_winnerCells'____________
#                 
#                 # 1. When there is capacity on dendrite to grow new synapses.
#                 if self.net_arch[bestMatch_CellDendrite[0], j].get_cell_numSynapsesOnDendrite(bestMatch_CellDendrite[1]) < self.maxSynapsesPerDendrite:
#                     print(color.GREEN,'Growing New Synapses on cell {',bestMatch_CellDendrite[0], j,'}; dendrite: ',bestMatch_CellDendrite[1],
#                           color.END)
#                     self.net_arch[bestMatch_CellDendrite[0], j].grow_cell_newSynapsesOnDendrite(dendrite_idx=bestMatch_CellDendrite[1],
#                                                                                                    prev_winnerCells=prev_winnerCells)
#                 # 2. When the dendrite has NO capacity to grow new synapses.
#                 else:
#                     print(color.DARKCYAN,'Growing New Synapses on a Different Dendrite', color.END)
#                     
#                     # Check if there is a significant overlap between the reinforced
#                     # synapses from above and 'prev_winnerCells'. If the overlap is less 
#                     # than the cell's NMDA threshold, then grow a new dendrite, else do not!
#                     list_reinforcedSynapses = get_idx_nonZeroElements(self.net_arch[bestMatch_CellDendrite[0], j].dendrites[bestMatch_CellDendrite[1]]*prev_state)
#                     list_prev_winnerCells = get_idx_nonZeroElements(prev_winnerCells)
#                     
#                     num_overlapCellIdx = len(set(list_reinforcedSynapses).intersection(list_prev_winnerCells))
#                     
#                     if num_overlapCellIdx < self.net_arch[bestMatch_CellDendrite[0], j].nmdaThreshold:
#                         if np.amax(cells_numUnusedDendrites) == 0:
#                             print(f'Issue 002 encountered! For column: {j}')
#                         else:
#                             cellIdx_leastUsedCell = random.choice(np.where(cells_numUnusedDendrites == np.amax(cells_numUnusedDendrites))[0])
#                             self.net_arch[cellIdx_leastUsedCell, j].grow_cell_newDendrite(prev_winnerCells)               
#                             print(color.DARKCYAN,'Growing New Synapses on cell {', cellIdx_leastUsedCell, j,'}', color.END)
#                             
#                             winner_cells.remove((bestMatch_CellDendrite[0], j))
#                             winner_cells.append((cellIdx_leastUsedCell, j))
#         
#         return winner_cells          
#     
# =============================================================================

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
#     def compute_deadCells(self):
#         
#         
#         list_net_cellDutyCycle.append(curr_state+curr_pred)
#         
#         if len(list_net_cellDutyCycle) > 1000:
#             net_cellDutyCycle = np.zeros([self.M, self.N])
#             for step in range(len(list_net_cellDutyCycle)):
#                 net_cellDutyCycle += list_net_cellDutyCycle[step]      
# =============================================================================
