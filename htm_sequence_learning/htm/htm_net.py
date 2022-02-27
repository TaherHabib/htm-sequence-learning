"""
MODULE DESCRIPTION:
-------------------

Implementation of the HTM network interface using HTM_NET() class. Needless to say,
the network is composed of HTM cells – interface imported from htm_cell.py. 
    
"""

import numpy as np
import random

from htm_cell import HTM_CELL
from htm_sequence_learning.htm.utils import dot_prod, get_idx_nonZeroElements
from htm_sequence_learning.reber_grammar.graph_configs import chars
from .utils import get_num_columns


class HTM_NET:

    def __init__(self,
                 columns_per_char=None,
                 cells_per_column=None,
                 max_dendrites_per_cell=None,
                 max_synapses_per_dendrite=None,
                 nmda_threshold=None,
                 permanence_threshold=None,
                 learning_threshold=None,
                 permanence_init=None,
                 permanence_init_sd=None,
                 permanence_decrement=None,
                 permanence_increment=None,
                 permanence_decay=None,
                 max_dendrite_dormancy=None,
                 verbosity=1):

        self.M = cells_per_column
        self.k = columns_per_char
        self.N = get_num_columns(columns_per_char, chars)
        self.max_dendrites_per_cell = max_dendrites_per_cell
        self.max_synapses_per_dendrite = max_synapses_per_dendrite
        self.nmda_threshold = nmda_threshold
        self.learning_threshold = learning_threshold

        self.permanence_threshold = permanence_threshold
        self.permanence_init = permanence_init
        self.permanence_init_sd = permanence_init_sd
        self.permanence_decrement = permanence_decrement
        self.permanence_increment = permanence_increment
        self.permanence_decay = permanence_decay
        self.max_dendrite_dormancy = max_dendrite_dormancy

        self.verbosity = verbosity

        # Setting up network architecture
        self.net_architecture = self.set_net_architecture

    @classmethod
    def from_json(cls, model_params, verbosity):
        return cls(columns_per_char=model_params['columns_per_char'],
                   cells_per_column=model_params['cells_per_column'],
                   max_dendrites_per_cell=model_params['max_dendrites_per_cell'],
                   max_synapses_per_dendrite=model_params['max_synapses_per_dendrite'],
                   nmda_threshold=model_params['nmda_threshold'],
                   permanence_threshold=model_params['permanence_threshold'],
                   learning_threshold=model_params['learning_threshold'],
                   permanence_init=model_params['permanence_init'],
                   permanence_init_sd=model_params['permanence_init_sd'],
                   permanence_decrement=model_params['permanence_decrement'],
                   permanence_increment=model_params['permanence_increment'],
                   permanence_decay=model_params['permanence_decay'],
                   max_dendrite_dormancy=model_params['max_dendrite_dormancy'],
                   verbosity=verbosity)

    @property
    def set_net_architecture(self):
        net_architecture = np.empty([self.M, self.N], dtype=HTM_CELL)
        for i in range(self.M):
            for j in range(self.N):
                net_architecture[i, j] = HTM_CELL(cells_per_column=self.M,
                                                  columns_per_char=self.k,
                                                  max_dendrites_per_cell=self.max_dendrites_per_cell,
                                                  max_synapses_per_dendrite=self.max_synapses_per_dendrite,
                                                  nmda_threshold=self.nmda_threshold,
                                                  permanence_threshold=self.permanence_threshold,
                                                  permanence_init=self.permanence_init,
                                                  permanence_init_sd=self.permanence_init_sd,
                                                  permanence_decrement=self.permanence_decrement,
                                                  permanence_increment=self.permanence_increment,
                                                  permanence_decay=self.permanence_decay,
                                                  max_dendrite_dormancy=self.max_dendrite_dormancy)
        return net_architecture

    def get_net_oneStepPrediction(self, net_state=None):
        """
        Computes the current step's predictions.

        Parameters
        ----------
        net_state : binary array of shape (MxN), containing the activity of 
        cell population from current time step.
        
        Returns
        -------
        pred_state : binary array of shape (MxN), containing the current timestep's 
        predictions (input chars for the next timestep).
        
        dict_predDendrites : dictionary storing indices of the dendrites on a cell
        which led to its predictive state. Only cells which enter the predictive
        state are stored with their indices (i,j).

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

                cell_predictivity, predDendrites = self.net_arch[i, j].get_cell_predictivity(net_state)

                if cell_predictivity:
                    pred_state[i, j] = 1
                    dict_predDendrites[(i, j)] = predDendrites

        return pred_state, dict_predDendrites

    def get_net_state(self, prev_pred=None, curr_input=None):
        """
        Computes the current timestep's network activity and predictions, based
        on the previous timestep's state of the network and the current 
        timestep's input.

        Parameters
        ----------
        prev_pred : MxN binary matrix of network's prediction at the previous
        timestep.
        
        curr_input : binary vector of current input, shape (N,), with 'k' 1's.

        Returns
        -------
        
        curr_state : binary MxN matrix of network's activity at current timestep. 
        
        curr_pred : binary MxN matrix of current timestep's predictions (i.e. 
        input chars for the next timestep).
        
        curr_predDendrites : dictionary storing indices of the dendrites on a cell
        which led to its predictive state. Only cells which enter the predictive
        state are stored with their indices (i,j).
        """

        curr_state = []

        # ______________________________________________________________________
        # Computing net state such that all minicolumns with current inputs 
        # are fully activated.
        for m in range(self.M):
            curr_state.append(curr_input)
        curr_state = np.array(curr_state, dtype=np.int8)  # MxN binary matrix

        # 'curr_state*prev_pred' gives MxN binary matrix of only those cells 
        # that are predicted AND present in the current input. Adding 
        # 'net_state' to this gives binary MxN 'net_state' from line 144 above
        # but with the predicted cells with value '2'. The next step is to find 
        # those columnsin 'curr_state*prev_pred + curr_state' with '2' as an 
        # entry and subtract 1. The following 6 lines of code are computing 
        # eq. 1, pg. 6 in the proposal.

        curr_state = curr_state * prev_pred + curr_state

        winning_cols = np.where(curr_input)[0]

        for j in winning_cols:
            if 2 in curr_state[:, j]:
                curr_state[:, j] -= 1

                # NOTE: Although the learning rules are designed to make the following
        # impossible, but even if it so happens that TWO DIFFERENT cells are 
        # predicted in the same minicolumn at a particular time step, then the 
        # equation below will make those cells become silent or active 
        # depending on whether that particular minicolumn is in the set of 
        # current timestep's input or not. In other words, the equation is 
        # robust to such special cases.
        # ______________________________________________________________________

        curr_pred, curr_predDendrites = self.get_net_oneStepPrediction(curr_state)
        # 'curr_pred' is a binary matrix with predictions for current timetep.

        return curr_state, curr_pred, curr_predDendrites

    def update_net_synapticPermanences(self,
                                       curr_state=None, prev_state=None,
                                       prev_pred=None, prev_predDendrites=None,
                                       prev_winnerCells=None):
        '''
        Handles the learning process in the network.
        Cases include:
            - when a column bursts
            - when a column is correctly predicted in the previous timestep
            - when a column is predicted but does not subsequently get activated
              in the current timestep (may occur in case of multiple predictions 
                                       or incorrect predictions)
            - when a symbol/letter repeats in the input reber string

        Parameters
        ----------
        curr_state : binary MxN matrix of network's activity at current timestep.
        prev_state : binary MxN matrix of network's activity in the previous timestep. 
        prev_pred : binary MxN matrix of network's prediction at the previous
        timestep.
        prev_predDendrites : dictionary storing indices of the dendrites on a cell
        which led to its predictive state in the previous timestep. 
        prev_winnerCells : binary MxN matrix of network's reinforced (winner) cells 
        for the previous timestep.

        Returns
        -------
        winnerCells : binary MxN matrix of network's reinforced (winner) cells 
        for the current timestep.

        '''

        # ______________________________________________________________________
        # From currently active columns, collect all columns that are bursting,
        # i.e. unpredicted (minicols with all 1s) and correctly and 
        # incorrectly predicted (minicols with more than one 1).

        active_cols = list(np.unique(np.where(curr_state)[1]))
        # np.array of length <k>

        predicted_cols = list(np.unique(np.where(prev_pred)[1]))
        # np.array of max. possible length <self.N>

        prevWinner_cols = list(np.unique(np.where(prev_winnerCells)[1]))
        # list of length <k>

        bursting_cols = [col for col in active_cols if curr_state[:, col].sum() == self.M]

        correctlyPredicted_cols = [col for col in active_cols if col not in bursting_cols]

        otherPredicted_cols = [col for col in predicted_cols if col not in correctlyPredicted_cols]

        # _______________________CASE 0_________________________________________
        # Checking if there is a repetition of symbols in the inputstream
        # ---------------------------------------------------------------------
        if active_cols == prevWinner_cols:

            repetition_winnerCells = self.case_net_repetitionOfActiveCols(prev_winnerCells=prev_winnerCells)

            if repetition_winnerCells is not None:
                winnerCells = np.zeros([self.M, self.N], dtype=np.int8)
                for cell_idx in repetition_winnerCells:
                    winnerCells[cell_idx] = 1
            else:
                winnerCells = None

            return winnerCells

        else:

            winnerCells = np.zeros([self.M, self.N], dtype=np.int8)

            # _______________________CASE I_________________________________________

            # When an active column is NOT PREDICTED, it will burst. 
            # (This would happen in the initial stages of learning)
            # ---------------------------------------------------------------------

            if len(bursting_cols) != 0:
                burstColumns_winnerCells = self.case_net_burstColumns(burstCols=bursting_cols,
                                                                      prev_state=prev_state,
                                                                      prev_winnerCells=prev_winnerCells)
                if burstColumns_winnerCells is not None:
                    for cell_idx in burstColumns_winnerCells:
                        winnerCells[cell_idx] = 1
                else:
                    return None

            # _______________________CASE II________________________________________

            # When winning column IS CORRECTLY PREDICTED (can have more than 1 
            # predicted cells)
            # ---------------------------------------------------------------------

            if len(correctlyPredicted_cols) != 0:
                reinforcePredictedColumns_winnerCells = self.case_net_reinforcePredictedColumns(
                    corrPredictedCols=correctlyPredicted_cols,
                    prev_state=prev_state,
                    prev_pred=prev_pred,
                    prev_predDendrites=prev_predDendrites)
                for cell_idx in reinforcePredictedColumns_winnerCells:
                    winnerCells[cell_idx] = 1

            # _______________________CASE III_______________________________________

            # When a column IS Otherwise PREDICTED (can have more than 1 predicted 
            # cells)
            # ---------------------------------------------------------------------

            if len(otherPredicted_cols) != 0:
                self.case_net_decayPredictedColumns(otherPredictedCols=otherPredicted_cols,
                                                    prev_state=prev_state,
                                                    prev_pred=prev_pred,
                                                    prev_predDendrites=prev_predDendrites)

            return winnerCells

    def case_net_repetitionOfActiveCols(self, prev_winnerCells=None):
        """
        Handles the case of repetition of input symbols/letters in the inputstream.

        Parameters
        ----------
        prev_winnerCells : binary MxN matrix of network's reinforced (winner) cells 
        for the previous timestep.

        Returns
        -------
        winner_cells : list of indices (i,j) of the cells which had one of their 
        dendrites reinforced.

        """

        # Checking if the 'prev_winnerCells' have autapses to themselves
        # If yes, reinforce these autapses. Else, grow new dendrites on each of
        # them to form these autapses

        winner_cells = []
        list_prev_winnerCells = get_idx_nonZeroElements(prev_winnerCells)

        if len(list_prev_winnerCells) != self.k:
            print(color.RED, color.BOLD, ':::::::::::::::::::::::::::::::::::::::::::Issue 005 encountered' \
                                         ' :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)
            print(color.RED, color.BOLD, '::::::::::::::::::::::::::::::::::::::::::: number of prev_winnerCells: ',
                  len(list_prev_winnerCells), \
                  ' :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)
            return None

        else:
            cell_dendrite_selection = []
            # will be an array of the same length as the 'list_prev_winnerCells'

            # ------------------------------------------------------------------
            # Finding a dendrite matching with 'prev_winnerCells'
            for cell_idx in list_prev_winnerCells:

                matchFound = False
                for dendrite_idx in range(self.maxDendritesPerCell):

                    if self.net_arch[cell_idx].dendrites[dendrite_idx] is None:
                        continue

                    else:
                        dendrite_overlapSynapses = np.count_nonzero(
                            prev_winnerCells * self.net_arch[cell_idx].dendrites[dendrite_idx])
                        if dendrite_overlapSynapses == len(list_prev_winnerCells):
                            matchFound = True
                            cell_dendrite_selection.append((cell_idx, dendrite_idx))
                            break

                        else:
                            continue

                if not matchFound:
                    cell_dendrite_selection.append((cell_idx, None))
            # ------------------------------------------------------------------

            counter_winners = 0
            for cell_dendrite_idx in cell_dendrite_selection:

                if cell_dendrite_idx[1] is None:
                    if self.net_arch[cell_dendrite_idx[0]].get_cell_numUnusedDendrites() == 0:
                        print(color.RED, color.BOLD,
                              ':::::::::::::::::::::::::::::::::::::::::::REPETITION / NO CELL CAPACITY. Cell {',
                              cell_dendrite_idx[0], \
                              '} :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)
                        break

                    else:
                        # Grow a new dendrite on 'cell_idx' with connections to 
                        # 'prev_WinnerCells'
                        newDendrite_idx = self.net_arch[cell_dendrite_idx[0]].grow_cell_newDendrite(prev_winnerCells)

                        if self.verbose > 1:
                            print(color.DARKGREEN, color.BOLD, 'REPETITION / NO MATCH FOUND. Growing new dendrite: ',
                                  newDendrite_idx, ' on cell {', cell_dendrite_idx[0], \
                                  '} with ',
                                  self.net_arch[cell_dendrite_idx[0]].get_cell_numSynapsesOnDendrite(newDendrite_idx), \
                                  ' new synapses.', color.END, color.END)
                        winner_cells.append(cell_dendrite_idx[0])
                        counter_winners += 1

                else:
                    reinforcedSynapses, tot_numSynapses = self.net_arch[
                        cell_dendrite_idx[0]].update_cell_dendritePermanences(dendrite_idx=cell_dendrite_idx[1],
                                                                              prev_state=prev_winnerCells)
                    if self.verbose > 1:
                        print(color.DARKGREEN, color.BOLD, 'REPETITION / MATCH FOUND. Cell {', cell_dendrite_idx[0],
                              '}; dendrite: ', cell_dendrite_idx[1], \
                              '. OverlapBySynapses: ', dendrite_overlapSynapses, \
                              '. Reinforcing ', np.count_nonzero(reinforcedSynapses), ' synapses. Existing synapses= ',
                              tot_numSynapses, color.END, color.END)
                    winner_cells.append(cell_dendrite_idx[0])
                    counter_winners += 1

            if counter_winners == len(list_prev_winnerCells):
                return winner_cells

            else:
                return None

    def case_net_burstColumns(self, burstCols=None, prev_state=None, prev_winnerCells=None):
        '''
        Handles the case when a column in the network bursts.

        Parameters
        ----------
        burstCols : list of indices bursting columns.
        prev_state : binary MxN matrix of network's activity in the previous timestep. 
        prev_winnerCells : binary MxN matrix of network's reinforced (winner) cells 
        for the previous timestep.

        Returns
        -------
        None, if multiple matching dendrites with connections to previous timestep's
        active cells are found OR if there is no capacity on any of the cells in 
        the column to grow new dendrites.
        
        winner cells : list of indices (i,j) of the cells which had one of their 
        dendrites reinforced.

        '''
        winner_cells = []

        for j in burstCols:

            cells_numUnusedDendrites = []
            # this will be an array of <M> elements with integer entries
            # between 0 and <maxDendritesPerCell>.

            cells_overlapSynapses = []
            cells_overlapScore = []

            # ------------------------------------------------------------------
            for i in range(self.M):

                cells_numUnusedDendrites.append(self.net_arch[i, j].get_cell_numUnusedDendrites())

                for dendrite_idx in range(self.maxDendritesPerCell):
                    if self.net_arch[i, j].dendrites[dendrite_idx] is None:
                        cells_overlapSynapses.append(((i, dendrite_idx), -1))
                        cells_overlapScore.append(((i, dendrite_idx), -1))

                    else:
                        cells_overlapSynapses.append(((i, dendrite_idx), np.count_nonzero(
                            prev_state * self.net_arch[i, j].dendrites[dendrite_idx])))
                        cells_overlapScore.append(
                            ((i, dendrite_idx), dot_prod(prev_state, self.net_arch[i, j].dendrites[dendrite_idx])[0]))
                        # 'prev_state' is a binary and 'dendrites[idx]' is a 
                        # float (numpy) MxN matrix. Hence,'overlapScore' 
                        # will be a numpy array of length 1, with a single 
                        # float value, greater than or equal to 0.

            # __________________________________________________________________
            # Finding the best matching segment using 
            # 'cells_overlapScorePerDendrite'.

            _overlapSynapses = [cells_overlapSynapses[i][1] for i in range(len(cells_overlapSynapses))]
            maxOverlap_bySynapses = np.amax(_overlapSynapses)

            # __________CASE: NO MATCHING DENDRITE IS FOUND_____________________
            # Grow a new one on least used cell in the column.
            if maxOverlap_bySynapses <= self.learningThreshold:

                # Checking for the condition where all <maxDendritesPerCell> 
                # dendrtes are used on every cell and still NO Match was found.
                if np.amax(cells_numUnusedDendrites) == 0:
                    print(color.RED, color.BOLD, ':::::::::::::::::::::::::::::::::::::::::::Issue 001 in col: ', j, \
                          ' :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)
                    return None

                else:
                    # Select randomly one cell from the set of all cells with 
                    # equal number of UnusedDendrites
                    cellIdx_leastUsedCell = random.choice(
                        np.where(cells_numUnusedDendrites == np.amax(cells_numUnusedDendrites))[0])
                    # Grow a new dendrite on this cell with connections to 
                    # 'prev_WinnerCells'
                    newDendrite_idx = self.net_arch[cellIdx_leastUsedCell, j].grow_cell_newDendrite(prev_winnerCells)

                    if self.verbose > 1:
                        print(color.GREEN, 'NO MATCH FOUND. Growing new dendrite: ', newDendrite_idx, ' on cell {',
                              cellIdx_leastUsedCell, j, \
                              '} with ',
                              self.net_arch[cellIdx_leastUsedCell, j].get_cell_numSynapsesOnDendrite(newDendrite_idx), \
                              ' new synapses.', color.END)

                    winner_cells.append((cellIdx_leastUsedCell, j))

            # _____________CASE: MATCHING DENDRITE FOUND________________________
            else:
                cells_maxOverlap_bySynapses = [cells_overlapSynapses[i] for i in
                                               np.where(_overlapSynapses == maxOverlap_bySynapses)[0]]

                # ____________Multiple Cells with MaxOverlapSynapses_______________
                # ________________with 'prev_state' Found______________________
                if len(cells_maxOverlap_bySynapses) > 1:
                    print(color.RED, color.BOLD, ':::::::::::::::::::::::::::::::::::::::::::Issue 003 in col: ', j, \
                          ' :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)
                    print(cells_maxOverlap_bySynapses)
                    return None

                # =============================================================================
                #                     filtered_cells_overlapScore = []
                #                     for cell_dend in cells_maxOverlap_bySynapses:
                #                         for c_d in cells_overlapScore:
                #                             if c_d[0] == cell_dend[0]:
                #                                 filtered_cells_overlapScore.append(c_d)
                #
                #                     _overlapScores = [filtered_cells_overlapScore[i][1] for i in range(len(filtered_cells_overlapScore))]
                #                     cells_maxOverlap_byScore = [filtered_cells_overlapScore[i] for i in np.where(_overlapScores == np.amax(_overlapScores))[0]]
                #
                #                     if len(cells_maxOverlap_byScore) > 1:
                #                         return None
                #
                #                     else:
                #                         bestMatch_CellDendrite = cells_maxOverlap_byScore[0][0]
                #                         bestMatch_Score = cells_maxOverlap_byScore[0][1]
                #
                #                         reinforcedSynapses, tot_numSynapses = self.net_arch[bestMatch_CellDendrite[0], j].update_cell_dendritePermanences(dendrite_idx=bestMatch_CellDendrite[1],
                #                                                                                                                                           prev_state=prev_state)
                #                         if self.verbose > 1:
                #                             print(color.PURPLE, 'MULT. MATCH / BEST SCORE FOUND. Cell {', bestMatch_CellDendrite[0],',',j,'}; dendrite: ',bestMatch_CellDendrite[1], \
                #                                   '. OverlapScore: ', bestMatch_Score, \
                #                                   '. Reinforcing ', np.count_nonzero(reinforcedSynapses),' synapses. Existing synapses= ', tot_numSynapses, color.END)
                #
                #                         winner_cells.append((bestMatch_CellDendrite[0], j))
                #
                #                         for cell_dend in cells_maxOverlap_bySynapses:
                #                             if cell_dend[0] == bestMatch_CellDendrite:
                #                                 continue
                #                             else:
                #                                 resetSynapses_ = np.zeros(shape=[self.M,self.N])
                #                                 for i in get_idx_nonZeroElements(prev_state*self.net_arch[cell_dend[0][0],j].dendrites[cell_dend[0][1]]):
                #                                     resetSynapses_[i] = 1
                #                                 self.net_arch[cell_dend[0][0], j].reset_cell_dendrite(dendrite_idx=cell_dend[0][1],
                #                                                                                       resetSynapses=resetSynapses_)
                # =============================================================================

                # ____________Only Single Cell with MaxOverlapSynapses_____________
                # ________________with 'prev_state' Found______________________
                else:
                    bestMatch_CellDendrite = cells_maxOverlap_bySynapses[0][0]
                    bestMatch_Score = cells_maxOverlap_bySynapses[0][1]

                    # Reinforce connections with active cells in 'prev_state' and 
                    # punish all other synaptic connections.
                    # Due to the previous restriction of having at least 
                    # <learningThreshold> matching synapses for a dendritic segment 
                    # to be counted as 'bestMatch', AT LEAST <learningThreshold> 
                    # synapses will be reinforced at any given timestep.
                    reinforcedSynapses, tot_numSynapses = self.net_arch[
                        bestMatch_CellDendrite[0], j].update_cell_dendritePermanences(
                        dendrite_idx=bestMatch_CellDendrite[1],
                        prev_state=prev_state)
                    if self.verbose > 1:
                        print(color.GREEN, 'MATCH FOUND. Cell {', bestMatch_CellDendrite[0], ',', j, '}; dendrite: ',
                              bestMatch_CellDendrite[1], \
                              '. OverlapScore: ', bestMatch_Score, \
                              '. Reinforcing ', np.count_nonzero(reinforcedSynapses), ' synapses. Existing synapses= ',
                              tot_numSynapses, color.END)

                    winner_cells.append((bestMatch_CellDendrite[0], j))

                    # Finding capacity on the reinforced dendritic segment
                    bestMatch_dendriteCapacity = self.maxSynapsesPerDendrite - tot_numSynapses

                    # VERY RARE
                    if bestMatch_dendriteCapacity < 0:
                        print(color.RED, color.BOLD, ':::::::::::::::::::::::::::::::::::::::::::Issue 002 in cell {',
                              bestMatch_CellDendrite[0], ',', j, \
                              '}; dendrite: ', bestMatch_CellDendrite[1],
                              ' :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)

                    # ________Growing New Synapses To 'prev_winnerCells'________

                    list_reinforcedSynapses = get_idx_nonZeroElements(reinforcedSynapses)
                    # will have at least <learningThreshold> elements

                    list_prev_winnerCells = get_idx_nonZeroElements(prev_winnerCells)

                    # Finding all the unlearnt 'prev_winnerCells' that need to be 
                    # connected/learnt.
                    list_unlearnt_winnerCells = set(list_prev_winnerCells) - set(list_prev_winnerCells).intersection(
                        list_reinforcedSynapses)

                    # Generating matrix of 'unlearnt_winnerCells'
                    unlearnt_winnerCells = np.zeros(shape=[self.M, self.N])
                    for i in list_unlearnt_winnerCells:
                        unlearnt_winnerCells[i] = 1

                    if len(list_unlearnt_winnerCells) <= bestMatch_dendriteCapacity:
                        total_newSynapses, remain_prevWinnerCells = self.net_arch[
                            bestMatch_CellDendrite[0], j].grow_cell_newSynapsesOnDendrite(
                            dendrite_idx=bestMatch_CellDendrite[1],
                            prev_winnerCells=prev_winnerCells)
                        if self.verbose > 1:
                            print(color.YELLOW, 'Growing ', total_newSynapses,
                                  ' new Synapses. Remaining Winner Cells: ', len(remain_prevWinnerCells), color.END)


                    elif len(list_unlearnt_winnerCells) > self.nmdaThreshold:
                        # =============================================================================
                        #                             winnerCells_ = prev_winnerCells
                        #                             list_winnerCells_ = set(list_prev_winnerCells)
                        #                             if self.verbose > 1:
                        #                                 print(color.PURPLE, 'Unlearnt Winner Cells <= NMDA threshold', color.END)
                        #                         else:
                        # =============================================================================
                        winnerCells_ = unlearnt_winnerCells
                        list_winnerCells_ = list_unlearnt_winnerCells
                        if self.verbose > 1:
                            print(color.DARKCYAN, 'Unlearnt Winner Cells > NMDA threshold', color.END)

                        # --------------------------------------------------
                        # Finding if there already exists a dendrite in the minicolummn
                        # with match to 'unlearnt_winnerCells'.
                        cellDendrites_ = []
                        for cell_idx in range(self.M):
                            for dendrite_idx in range(self.maxDendritesPerCell):
                                if self.net_arch[cell_idx, j].dendrites[dendrite_idx] is None:
                                    continue
                                else:
                                    existing_synapticPerm = get_idx_nonZeroElements(
                                        self.net_arch[cell_idx, j].dendrites[dendrite_idx])
                                    if list_winnerCells_ - list_winnerCells_.intersection(
                                            existing_synapticPerm) == set():
                                        cellDendrites_.append((cell_idx, dendrite_idx))
                        # --------------------------------------------------

                        if len(cellDendrites_) == 0:
                            if np.amax(cells_numUnusedDendrites) == 0:
                                print(color.RED, color.BOLD,
                                      ':::::::::::::::::::::::::::::::::::::::::::Issue 004 in col: ', j, \
                                      ' :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)
                                return None

                            else:
                                cellIdx_leastUsedCell = random.choice(
                                    np.where(cells_numUnusedDendrites == np.amax(cells_numUnusedDendrites))[0])
                                newDendrite_idx = self.net_arch[cellIdx_leastUsedCell, j].grow_cell_newDendrite(
                                    winnerCells_)

                                if self.verbose > 1:
                                    print(color.YELLOW, 'NO EXISTING DENDRITE. Growing new dendrite: ', newDendrite_idx,
                                          ' on cell {', cellIdx_leastUsedCell, j, \
                                          '} with ',
                                          self.net_arch[cellIdx_leastUsedCell, j].get_cell_numSynapsesOnDendrite(
                                              newDendrite_idx), ' new synapses.', color.END)

                                # Undoing the Previous Reinforcement of Dendrites and de-selecting
                                # it from 'winner_cells'
                                self.net_arch[bestMatch_CellDendrite[0], j].undo_cell_dendriteReinforcement(
                                    dendrite_idx=bestMatch_CellDendrite[1],
                                    reinforcedSynapses=reinforcedSynapses)
                                winner_cells.remove((bestMatch_CellDendrite[0], j))
                                winner_cells.append((cellIdx_leastUsedCell, j))

                        else:
                            random.seed()
                            cellDendrite_idx = random.choice(cellDendrites_)
                            reinforcedSynapses_, tot_numSynapses_ = self.net_arch[
                                cellDendrite_idx[0], j].update_cell_dendritePermanences(
                                dendrite_idx=cellDendrite_idx[1],
                                prev_state=winnerCells_)
                            if self.verbose > 1:
                                print(color.YELLOW, 'EXISTING DENDRITE FOUND. Cell {', cellDendrite_idx[0], ',', j,
                                      '}; dendrite: ', cellDendrite_idx[1], \
                                      '. Reinforcing ', np.count_nonzero(reinforcedSynapses_),
                                      ' synapses. Existing synapses= ', tot_numSynapses_, color.END)

                            # Undoing the Previous Reinforcement of Dendrites and de-selecting
                            # it from 'winner_cells'
                            self.net_arch[bestMatch_CellDendrite[0], j].undo_cell_dendriteReinforcement(
                                dendrite_idx=bestMatch_CellDendrite[1],
                                reinforcedSynapses=reinforcedSynapses)
                            winner_cells.remove((bestMatch_CellDendrite[0], j))
                            winner_cells.append((cellDendrite_idx[0], j))

                    else:
                        continue

        return winner_cells

    def case_net_reinforcePredictedColumns(self, corrPredictedCols=None, prev_state=None,
                                           prev_pred=None, prev_predDendrites=None):
        '''
        Handles the case when a column is correctly predicted.

        Parameters
        ----------
        corrPredictedCols : list of indices correctly predicted columns.
        prev_state : binary MxN matrix of network's activity in the previous timestep.
        prev_pred : binary MxN matrix of network's prediction at the previous
        timestep.
        prev_predDendrites : dictionary storing indices of the dendrites on a cell
        which led to its predictive state in the previous timestep. 

        Returns
        -------
        winner_cells : list of indices (i,j) of the cells which had one of their 
        dendrites reinforced.

        '''
        winner_cells = []

        for j in corrPredictedCols:

            # extract i-indices of all CORRECTLY predicted cells in the column
            cells_i = np.where(prev_pred[:, j])[0]

            # __________________________________________________________________
            # Reinforce active dendrites for all of predicted cells in the 
            # minicolumn.
            for i in cells_i:

                winner_cells.append((i, j))
                # for indices of all dendrites that led to cell's prediction.
                for dendrite_idx in prev_predDendrites[(i, j)]:
                    reinforcedSynapses, tot_numSynapses = self.net_arch[i, j].update_cell_dendritePermanences(
                        dendrite_idx=dendrite_idx,
                        prev_state=prev_state)

                    # print(color.DARKCYAN, 'CORRECT PRED. Cell {', i,',',j,'}; dendrite: ', dendrite_idx, \
                    #     '. Reinforcing ', np.count_nonzero(reinforcedSynapses),' synapses. Existing synapses= ', tot_numSynapses, color.END)

        return winner_cells

    def case_net_decayPredictedColumns(self, otherPredictedCols=None, prev_state=None,
                                       prev_pred=None, prev_predDendrites=None):
        """
        Handles the case when a column is predicted but does not subsequently get 
        activated in the current timestep 
        (may occur in case of multiple predictions or incorrect predictions).

        Parameters
        ----------
        otherPredictedCols : list of indices of other predicted columns.
        prev_state : binary MxN matrix of network's activity in the previous timestep.
        prev_pred : binary MxN matrix of network's prediction at the previous
        timestep.
        prev_predDendrites : dictionary storing indices of the dendrites on a cell
        which led to its predictive state in the previous timestep. 

        Returns
        -------
        None.

        """

        for j in otherPredictedCols:

            # extract i-indices of all OTHER predicted cells in the column
            cells_i = np.where(prev_pred[:, j])[0]

            # __________________________________________________________________
            # Punish active dendrites for all of OTHER predicted cells in the 
            # minicolumn.
            for i in cells_i:

                # for indices of all dendrites that led to cell's prediction.
                for dendrite_idx in prev_predDendrites[(i, j)]:
                    _, _ = self.net_arch[i, j].update_cell_dendritePermanences(dendrite_idx=dendrite_idx,
                                                                               prev_state=prev_state,
                                                                               decay=True)

        return

    def update_net_dendriteDutyCycle(self):
        """
        Used to update the duty cycle of every dendrite in the network.
        - Decreases an existing dendrite's duty cycle by 1.
        - If the dendrite's duty cycle falls below 1, then it prunes the dendrite.

        Returns
        -------
        None.

        """

        for i in range(self.M):
            for j in range(self.N):
                if self.net_arch[i, j].get_cell_numDendrites() != 0:

                    for dendrite_idx in range(self.maxDendritesPerCell):
                        if self.net_arch[i, j].dendrites_dutyCycle[dendrite_idx] is not None:
                            self.net_arch[i, j].dendrites_dutyCycle[dendrite_idx] -= 1
                        else:
                            continue

                    for dendrite_idx in range(self.maxDendritesPerCell):
                        if self.net_arch[i, j].dendrites_dutyCycle[dendrite_idx] is not None:
                            if self.net_arch[i, j].dendrites_dutyCycle[dendrite_idx] <= 0:
                                self.net_arch[i, j].dendrites_dutyCycle[dendrite_idx] = None
                                self.net_arch[i, j].dendrites[dendrite_idx] = None
                        else:
                            continue

                else:
                    continue

        return

    def prune_net_permanences(self):
        """
        This function ensures that the permanence levels of all synapses in the
        network remains between 0 and 1 at all times.

        Returns
        -------
        None.

        """

        for i in range(self.M):
            for j in range(self.N):
                for dendrite in self.net_arch[i, j].get_cell_allDendrites():
                    if dendrite is None:
                        continue
                    else:
                        dendrite[dendrite < 0] = 0.0
                        dendrite[dendrite > 1] = 1.0
        return

    def get_NETWORK(self, char_onehot='all'):

        if char_onehot == 'all':
            return self.net_arch

        else:
            return self.net_arch[:, np.where(char_onehot)[0]]

    def get_net_dims(self):

        return (self.M, self.N)


class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    DARKGREEN = '\033[32m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
