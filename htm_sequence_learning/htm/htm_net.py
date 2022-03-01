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

        # Setting up network architecture
        self.net_architecture = self.set_net_architecture
        self.verbosity = verbosity

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

    def get_net_onestep_prediction(self, net_state=None):
        """
        Computes the current step's predictions.

        Parameters
        ----------
        net_state : binary array of shape (MxN), containing the activity of cell population from current time step.
        
        Returns
        -------
        pred_state : binary array of shape (MxN), containing the current time step's predictions (input chars for the 
        next time step).
        dict_pred_dendrites : dictionary storing indices of the dendrites on a cell which led to its predictive state. 
        Only cells which enter the predictive state are stored with their indices (i,j).
        """

        pred_state = np.zeros([self.M, self.N], dtype=np.int8)  # MxN binary numpy array to store the predictive states 
        # of all cells.

        dict_pred_dendrites = {}  # Dictionary to store the index of the dendrites that led to the predictive states of 
        # the cell (i,j). :: key: index of the cell, (i,j), value: 1D numpy array of max. possible length 
        # <maxDendritesPerCell> with integer entries between 0 and <maxDendritesPerCell>-1.

        for j in range(self.N):
            for i in range(self.M):

                cell_predictivity, pred_dendrites = self.net_architecture[i, j].get_cell_predictivity(net_state)

                if cell_predictivity:
                    pred_state[i, j] = 1
                    dict_pred_dendrites[(i, j)] = pred_dendrites

        return pred_state, dict_pred_dendrites

    def get_net_state(self, prev_pred=None, curr_input=None):
        """
        Computes the current time step's network activity and predictions, based on the previous time step's state of 
        the network and the current time step's input.

        Parameters
        ----------
        prev_pred : MxN binary matrix of network's prediction at the previous time step.
        curr_input : binary vector of current input, shape (N,), with 'k' 1's.

        Returns
        -------
        curr_state : binary MxN matrix of network's activity at current time step. 
        curr_pred : binary MxN matrix of current time step's predictions (i.e. input chars for the next time step).
        curr_pred_dendrites : dictionary storing indices of the dendrites on a cell which led to its predictive state. 
        Only cells which enter the predictive state are stored with their indices (i,j).
        """

        curr_state = []

        # Computing net state such that all minicolumns with current inputs are fully activated.
        for m in range(self.M):
            curr_state.append(curr_input)
        curr_state = np.array(curr_state, dtype=np.int8)  # MxN binary matrix

        # 'curr_state*prev_pred' gives MxN binary matrix of only those cells that are predicted AND present in the 
        # current input. Adding 'net_state' to this gives binary MxN 'net_state' from line 144 above but with the 
        # predicted cells with value '2'. The next step is to find those columns in 'curr_state*prev_pred + curr_state' 
        # with '2' as an entry and subtract 1. The following 6 lines of code are computing eq. 1, pg. 6 in the proposal.
        curr_state = curr_state * prev_pred + curr_state
        winning_cols = np.where(curr_input)[0]

        for j in winning_cols:
            if 2 in curr_state[:, j]:
                curr_state[:, j] -= 1

        # NOTE: Although the learning rules are designed to make the following impossible, but even if it so happens 
        # that TWO DIFFERENT cells are predicted in the same minicolumn at a particular time step, then the equation 
        # below will make those cells become silent or active depending on whether that particular minicolumn is in the 
        # set of current time step's input or not. In other words, the equation is robust to such special cases.

        curr_pred, curr_pred_dendrites = self.get_net_onestep_prediction(curr_state)
        # 'curr_pred' is a binary matrix with predictions for current time step.

        return curr_state, curr_pred, curr_pred_dendrites

    def update_net_synaptic_permanences(self,
                                        curr_state=None,
                                        prev_state=None,
                                        prev_pred=None,
                                        prev_pred_dendrites=None,
                                        prev_winner_cells=None):
        """
        Handles the learning process in the network.
        Cases include:
            - when a column bursts
            - when a column is correctly predicted in the previous time step
            - when a column is predicted but does not subsequently get activated
              in the current time step (may occur in case of multiple predictions 
                                       or incorrect predictions)
            - when a symbol/letter repeats in the input reber string

        Parameters
        ----------
        curr_state : binary MxN matrix of network's activity at current time step.
        prev_state : binary MxN matrix of network's activity in the previous time step. 
        prev_pred : binary MxN matrix of network's prediction at the previous
        time step.
        prev_pred_dendrites : dictionary storing indices of the dendrites on a cell
        which led to its predictive state in the previous time step. 
        prev_winner_cells : binary MxN matrix of network's reinforced (winner) cells 
        for the previous time step.

        Returns
        -------
        winner_cells : binary MxN matrix of network's reinforced (winner) cells 
        for the current time step.
        """

        # From currently active columns, collect all columns that are bursting,
        # i.e. unpredicted (minicols with all 1s) and correctly and 
        # incorrectly predicted (minicols with more than one 1).

        active_cols = list(np.unique(np.where(curr_state)[1]))  # np.array of length <k>

        predicted_cols = list(np.unique(np.where(prev_pred)[1]))  # np.array of max. possible length <self.N>

        prev_winner_cols = list(np.unique(np.where(prev_winner_cells)[1]))  # list of length <k>

        bursting_cols = [col for col in active_cols if curr_state[:, col].sum() == self.M]

        correctly_predicted_cols = [col for col in active_cols if col not in bursting_cols]

        other_predicted_cols = [col for col in predicted_cols if col not in correctly_predicted_cols]

        # _______________________CASE 0_________________________________________________________________________________
        # Checking if there is a repetition of symbols in the inputstream
        if active_cols == prev_winner_cols:

            repetition_winner_cells = self.case_net_repetitionOfActiveCols(prev_winner_cells=prev_winner_cells)

            if repetition_winner_cells is not None:
                winner_cells = np.zeros([self.M, self.N], dtype=np.int8)
                for cell_idx in repetition_winner_cells:
                    winner_cells[cell_idx] = 1
            else:
                winner_cells = None

            return winner_cells

        else:

            winner_cells = np.zeros([self.M, self.N], dtype=np.int8)

            # _______________________CASE I_____________________________________________________________________________
            # When an active column is NOT PREDICTED, it will burst. (This would happen in the initial stages of 
            # learning)
            if len(bursting_cols) != 0:
                burstColumns_winner_cells = self.case_net_burstColumns(burst_cols=bursting_cols,
                                                                       prev_state=prev_state,
                                                                       prev_winner_cells=prev_winner_cells)
                if burstColumns_winner_cells is not None:
                    for cell_idx in burstColumns_winner_cells:
                        winner_cells[cell_idx] = 1
                else:
                    return None

            # _______________________CASE II____________________________________________________________________________
            # When winning column IS CORRECTLY PREDICTED (can have more than 1 predicted cells)
            if len(correctly_predicted_cols) != 0:
                reinforcePredictedColumns_winner_cells = self.case_net_reinforcePredictedColumns(
                    corr_predicted_cols=correctly_predicted_cols,
                    prev_state=prev_state,
                    prev_pred=prev_pred,
                    prev_pred_dendrites=prev_pred_dendrites)
                for cell_idx in reinforcePredictedColumns_winner_cells:
                    winner_cells[cell_idx] = 1

            # _______________________CASE III___________________________________________________________________________
            # When a column IS Otherwise PREDICTED (can have more than 1 predicted cells)

            if len(other_predicted_cols) != 0:
                self.case_net_decayPredictedColumns(other_predicted_cols=other_predicted_cols,
                                                    prev_state=prev_state,
                                                    prev_pred=prev_pred,
                                                    prev_pred_dendrites=prev_pred_dendrites)

            return winner_cells

    def case_net_repetitionOfActiveCols(self,
                                        prev_winner_cells=None):
        """
        Handles the case of repetition of input symbols/letters in the inputstream.

        Parameters
        ----------
        prev_winner_cells : binary MxN matrix of network's reinforced (winner) cells 
        for the previous time step.

        Returns
        -------
        winner_cells : list of indices (i,j) of the cells which had one of their 
        dendrites reinforced.
        """

        # Checking if the 'prev_winner_cells' have autapses to themselves. If yes, reinforce these autapses. 
        # Else, grow new dendrites on each of them to form these autapses
        winner_cells = []
        list_prev_winner_cells = get_idx_nonZeroElements(prev_winner_cells)

        if len(list_prev_winner_cells) != self.k:
            print(color.RED, color.BOLD, ':::::::::::::::::::::::::::::::::::::::::::Issue 005 encountered'
                                         ':::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)
            print(color.RED, color.BOLD, '::::::::::::::::::::::::::::::::::::::::::: number of prev_winner_cells: ',
                  len(list_prev_winner_cells), ' :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)
            return None

        else:
            cell_dendrite_selection = []  # will be an array of the same length as the 'list_prev_winner_cells'

            # ----------------------------------------------------------------------------------------------------------
            # Finding a dendrite matching with 'prev_winner_cells'
            for cell_idx in list_prev_winner_cells:
                matchFound = False
                for dendrite_idx in range(self.max_dendrites_per_cell):
                    if self.net_architecture[cell_idx].dendrites[dendrite_idx] is None:
                        continue
                    else:
                        dendrite_overlapSynapses = np.count_nonzero(
                            prev_winner_cells * self.net_architecture[cell_idx].dendrites[dendrite_idx])
                        if dendrite_overlapSynapses == len(list_prev_winner_cells):
                            matchFound = True
                            cell_dendrite_selection.append((cell_idx, dendrite_idx))
                            break
                        else:
                            continue
                if not matchFound:
                    cell_dendrite_selection.append((cell_idx, None))
            # ----------------------------------------------------------------------------------------------------------

            counter_winners = 0
            for cell_dendrite_idx in cell_dendrite_selection:
                if cell_dendrite_idx[1] is None:
                    if self.net_architecture[cell_dendrite_idx[0]].get_cell_numUnusedDendrites() == 0:
                        print(color.RED, color.BOLD, ':::::::::::::::::::::::::::::::::::::::::::'
                                                     'REPETITION / NO CELL CAPACITY. Cell {', cell_dendrite_idx[0],
                              '} :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)
                        break
                    else:
                        # Grow a new dendrite on 'cell_idx' with connections to 'prev_winner_cells'
                        newDendrite_idx = self.net_architecture[cell_dendrite_idx[0]].grow_cell_newDendrite(
                            prev_winner_cells)
                        if self.verbosity > 1:
                            print(color.DARKGREEN, color.BOLD, 'REPETITION / NO MATCH FOUND. Growing new dendrite: ',
                                  newDendrite_idx, ' on cell {', cell_dendrite_idx[0], '} with ',
                                  self.net_architecture[cell_dendrite_idx[0]].get_cell_numSynapsesOnDendrite(
                                      newDendrite_idx), ' new synapses.', color.END, color.END)
                        winner_cells.append(cell_dendrite_idx[0])
                        counter_winners += 1
                else:
                    reinforcedSynapses, tot_numSynapses = self.net_architecture[
                        cell_dendrite_idx[0]].update_cell_dendritePermanences(dendrite_idx=cell_dendrite_idx[1],
                                                                              prev_state=prev_winner_cells)
                    if self.verbosity > 1:
                        print(color.DARKGREEN, color.BOLD, 'REPETITION / MATCH FOUND. Cell {', cell_dendrite_idx[0],
                              '}; dendrite: ', cell_dendrite_idx[1], '. OverlapBySynapses: ', dendrite_overlapSynapses,
                              '. Reinforcing ', np.count_nonzero(reinforcedSynapses), ' synapses. Existing synapses= ',
                              tot_numSynapses, color.END, color.END)
                    winner_cells.append(cell_dendrite_idx[0])
                    counter_winners += 1

            if counter_winners == len(list_prev_winner_cells):
                return winner_cells
            else:
                return None

    def case_net_burstColumns(self,
                              burst_cols=None,
                              prev_state=None,
                              prev_winner_cells=None):
        """
        Handles the case when a column in the network bursts.

        Parameters
        ----------
        burst_cols : list of indices bursting columns.
        prev_state : binary MxN matrix of network's activity in the previous time step.
        prev_winner_cells : binary MxN matrix of network's reinforced (winner) cells for the previous time step.

        Returns
        -------
        None, if multiple matching dendrites with connections to previous time step's active cells are found OR if there
        is no capacity on any of the cells in the column to grow new dendrites.
        winner cells : list of indices (i,j) of the cells which had one of their dendrites reinforced.
        """

        winner_cells = []

        for j in burst_cols:
            cells_numUnusedDendrites = []  # this will be an array of <M> elements with integer entries between
            # 0 and <maxDendritesPerCell>
            cells_overlapSynapses = []
            cells_overlapScore = []

            # ----------------------------------------------------------------------------------------------------------
            for i in range(self.M):
                cells_numUnusedDendrites.append(self.net_architecture[i, j].get_cell_numUnusedDendrites())
                for dendrite_idx in range(self.max_dendrites_per_cell):
                    if self.net_architecture[i, j].dendrites[dendrite_idx] is None:
                        cells_overlapSynapses.append(((i, dendrite_idx), -1))
                        cells_overlapScore.append(((i, dendrite_idx), -1))
                    else:
                        cells_overlapSynapses.append(((i, dendrite_idx), np.count_nonzero(
                            prev_state * self.net_architecture[i, j].dendrites[dendrite_idx])))
                        cells_overlapScore.append(
                            ((i, dendrite_idx),
                             dot_prod(prev_state, self.net_architecture[i, j].dendrites[dendrite_idx])[0]))
                        # 'prev_state' is a binary and 'dendrites[idx]' is a float (numpy) MxN matrix. Hence,
                        # 'overlapScore' will be a numpy array of length 1, with a single float value, greater than or
                        # equal to 0.

            # __________________________________________________________________________________________________________
            # Finding the best matching segment using 'cells_overlapScorePerDendrite'.
            _overlapSynapses = [cells_overlapSynapses[i][1] for i in range(len(cells_overlapSynapses))]
            maxOverlap_bySynapses = np.amax(_overlapSynapses)

            # =============================CASE: NO MATCHING DENDRITE FOUND=============================================
            # Grow a new one on least used cell in the column.
            if maxOverlap_bySynapses <= self.learning_threshold:
                # Checking for the condition where all <maxDendritesPerCell> dendrites are used on every cell and
                # still NO Match was found.
                if np.amax(cells_numUnusedDendrites) == 0:
                    print(color.RED, color.BOLD, ':::::::::::::::::::::::::::::::::::::::::::Issue 001 in col: ', j,
                          ' :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)
                    return None
                else:
                    # Select randomly one cell from the set of all cells with equal number of UnusedDendrites
                    cellIdx_leastUsedCell = random.choice(
                        np.where(cells_numUnusedDendrites == np.amax(cells_numUnusedDendrites))[0])
                    # Grow a new dendrite on this cell with connections to 'prev_winner_cells'
                    newDendrite_idx = self.net_architecture[cellIdx_leastUsedCell, j].grow_cell_newDendrite(
                        prev_winner_cells)
                    if self.verbosity > 1:
                        print(color.GREEN, 'NO MATCH FOUND. Growing new dendrite: ', newDendrite_idx, ' on cell {',
                              cellIdx_leastUsedCell, j, '} with ',
                              self.net_architecture[cellIdx_leastUsedCell, j].get_cell_numSynapsesOnDendrite(
                                  newDendrite_idx), ' new synapses.', color.END)
                    winner_cells.append((cellIdx_leastUsedCell, j))

            # =============================CASE: MATCHING DENDRITE FOUND================================================
            else:
                cells_maxOverlap_bySynapses = [cells_overlapSynapses[i] for i in
                                               np.where(_overlapSynapses == maxOverlap_bySynapses)[0]]
                # ____________Multiple Cells with MaxOverlapSynapses with 'prev_state' Found____________________________
                if len(cells_maxOverlap_bySynapses) > 1:
                    print(color.RED, color.BOLD, ':::::::::::::::::::::::::::::::::::::::::::Issue 003 in col: ', j,
                          ' :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)
                    print(cells_maxOverlap_bySynapses)
                    return None

                # ____________Only Single Cell with MaxOverlapSynapses with 'prev_state' Found__________________________
                else:
                    bestMatch_CellDendrite = cells_maxOverlap_bySynapses[0][0]
                    bestMatch_Score = cells_maxOverlap_bySynapses[0][1]
                    # Reinforce connections with active cells in 'prev_state' and punish all other synaptic connections.
                    # Due to the previous restriction of having at least <learning_threshold> matching synapses for a
                    # dendritic segment to be counted as 'bestMatch', AT LEAST <learning_threshold> synapses will be
                    # reinforced at any given time step.
                    reinforcedSynapses, tot_numSynapses = self.net_architecture[
                        bestMatch_CellDendrite[0], j].update_cell_dendritePermanences(
                        dendrite_idx=bestMatch_CellDendrite[1],
                        prev_state=prev_state)
                    if self.verbosity > 1:
                        print(color.GREEN, 'MATCH FOUND. Cell {', bestMatch_CellDendrite[0], ',', j, '}; dendrite: ',
                              bestMatch_CellDendrite[1], '. OverlapScore: ', bestMatch_Score,
                              '. Reinforcing ', np.count_nonzero(reinforcedSynapses), ' synapses. Existing synapses= ',
                              tot_numSynapses, color.END)
                    winner_cells.append((bestMatch_CellDendrite[0], j))

                    # Finding capacity on the reinforced dendritic segment
                    bestMatch_dendriteCapacity = self.max_synapses_per_dendrite - tot_numSynapses

                    # VERY RARE
                    if bestMatch_dendriteCapacity < 0:
                        print(color.RED, color.BOLD, ':::::::::::::::::::::::::::::::::::::::::::Issue 002 in cell {',
                              bestMatch_CellDendrite[0], ',', j,
                              '}; dendrite: ', bestMatch_CellDendrite[1],
                              ' :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)

                    # ________Growing New Synapses To 'prev_winner_cells'________
                    list_reinforcedSynapses = get_idx_nonZeroElements(reinforcedSynapses)  # will have at least
                    # <learning_threshold> elements
                    list_prev_winner_cells = get_idx_nonZeroElements(prev_winner_cells)

                    # Finding all the unlearnt 'prev_winner_cells' that need to be connected/learnt.
                    list_unlearnt_winner_cells = set(list_prev_winner_cells) - \
                                                 set(list_prev_winner_cells).intersection(list_reinforcedSynapses)

                    # Generating matrix of 'unlearnt_winner_cells'
                    unlearnt_winner_cells = np.zeros(shape=[self.M, self.N])
                    for i in list_unlearnt_winner_cells:
                        unlearnt_winner_cells[i] = 1

                    if len(list_unlearnt_winner_cells) <= bestMatch_dendriteCapacity:
                        total_newSynapses, remain_prev_winner_cells = self.net_architecture[
                            bestMatch_CellDendrite[0], j].grow_cell_newSynapsesOnDendrite(
                            dendrite_idx=bestMatch_CellDendrite[1],
                            prev_winner_cells=prev_winner_cells)
                        if self.verbosity > 1:
                            print(color.YELLOW, 'Growing ', total_newSynapses,
                                  ' new Synapses. Remaining Winner Cells: ', len(remain_prev_winner_cells), color.END)

                    elif len(list_unlearnt_winner_cells) > self.nmda_threshold:
                        # ---------------------------------
                        # winner_cells_ = prev_winner_cells
                        # list_winner_cells_ = set(list_prev_winner_cells)
                        # if self.verbosity > 1:
                        #    print(color.PURPLE, 'Unlearnt Winner Cells <= NMDA threshold', color.END)
                        # else:
                        # ---------------------------------
                        winner_cells_ = unlearnt_winner_cells
                        list_winner_cells_ = list_unlearnt_winner_cells
                        if self.verbosity > 1:
                            print(color.DARKCYAN, 'Unlearnt Winner Cells > NMDA threshold', color.END)

                        # --------------------------------------------------------------
                        # Finding if there already exists a dendrite in the minicolummn with match to
                        # 'unlearnt_winner_cells'.
                        cell_dendrites_ = []
                        for cell_idx in range(self.M):
                            for dendrite_idx in range(self.max_dendrites_per_cell):
                                if self.net_architecture[cell_idx, j].dendrites[dendrite_idx] is None:
                                    continue
                                else:
                                    existing_synapticPerm = get_idx_nonZeroElements(
                                        self.net_architecture[cell_idx, j].dendrites[dendrite_idx])
                                    if list_winner_cells_ - list_winner_cells_.intersection(
                                            existing_synapticPerm) == set():
                                        cell_dendrites_.append((cell_idx, dendrite_idx))
                        # --------------------------------------------------
                        if len(cell_dendrites_) == 0:
                            if np.amax(cells_numUnusedDendrites) == 0:
                                print(color.RED, color.BOLD,
                                      ':::::::::::::::::::::::::::::::::::::::::::Issue 004 in col: ', j,
                                      ' :::::::::::::::::::::::::::::::::::::::::::', color.END, color.END)
                                return None
                            else:
                                cellIdx_leastUsedCell = random.choice(
                                    np.where(cells_numUnusedDendrites == np.amax(cells_numUnusedDendrites))[0])
                                newDendrite_idx = self.net_architecture[cellIdx_leastUsedCell, j].grow_cell_newDendrite(
                                    winner_cells_)
                                if self.verbosity > 1:
                                    print(color.YELLOW, 'NO EXISTING DENDRITE. Growing new dendrite: ', newDendrite_idx,
                                          ' on cell {', cellIdx_leastUsedCell, j, '} with ',
                                          self.net_architecture[cellIdx_leastUsedCell, j].get_cell_numSynapsesOnDendrite(
                                              newDendrite_idx), ' new synapses.', color.END)

                                # Undoing the Previous Reinforcement of Dendrites and de-selecting it from 'winner_cells'
                                self.net_architecture[bestMatch_CellDendrite[0], j].undo_cell_dendriteReinforcement(
                                    dendrite_idx=bestMatch_CellDendrite[1],
                                    reinforcedSynapses=reinforcedSynapses)
                                winner_cells.remove((bestMatch_CellDendrite[0], j))
                                winner_cells.append((cellIdx_leastUsedCell, j))
                        else:
                            random.seed()
                            cellDendrite_idx = random.choice(cell_dendrites_)
                            reinforcedSynapses_, tot_numSynapses_ = self.net_architecture[
                                cellDendrite_idx[0], j].update_cell_dendritePermanences(
                                dendrite_idx=cellDendrite_idx[1],
                                prev_state=winner_cells_)
                            if self.verbosity > 1:
                                print(color.YELLOW, 'EXISTING DENDRITE FOUND. Cell {', cellDendrite_idx[0], ',', j,
                                      '}; dendrite: ', cellDendrite_idx[1], '. Reinforcing ',
                                      np.count_nonzero(reinforcedSynapses_), ' synapses. Existing synapses= ',
                                      tot_numSynapses_, color.END)
                            # Undoing the Previous Reinforcement of Dendrites and de-selecting it from 'winner_cells'
                            self.net_architecture[bestMatch_CellDendrite[0], j].undo_cell_dendriteReinforcement(
                                dendrite_idx=bestMatch_CellDendrite[1],
                                reinforcedSynapses=reinforcedSynapses)
                            winner_cells.remove((bestMatch_CellDendrite[0], j))
                            winner_cells.append((cellDendrite_idx[0], j))

                    else:
                        continue

        return winner_cells

    def case_net_reinforcePredictedColumns(self,
                                           corr_predicted_cols=None,
                                           prev_state=None,
                                           prev_pred=None,
                                           prev_pred_dendrites=None):
        """
        Handles the case when a column is correctly predicted.

        Parameters
        ----------
        corr_predicted_cols : list of indices correctly predicted columns.
        prev_state : binary MxN matrix of network's activity in the previous time step.
        prev_pred : binary MxN matrix of network's prediction at the previous time step.
        prev_pred_dendrites : dictionary storing indices of the dendrites on a cell
        which led to its predictive state in the previous time step.

        Returns
        -------
        winner_cells : list of indices (i,j) of the cells which had one of their dendrites reinforced.
        """

        winner_cells = []

        for j in corr_predicted_cols:
            cells_i = np.where(prev_pred[:, j])[0]  # extract i-indices of all CORRECTLY predicted cells in the column

            # Reinforce active dendrites for all of predicted cells in the minicolumn.
            for i in cells_i:
                winner_cells.append((i, j))
                # for indices of all dendrites that led to cell's prediction.
                for dendrite_idx in prev_pred_dendrites[(i, j)]:
                    reinforcedSynapses, tot_numSynapses = self.net_architecture[i, j].update_cell_dendritePermanences(
                        dendrite_idx=dendrite_idx,
                        prev_state=prev_state)
                    # -------------------------------
                    # print(color.DARKCYAN, 'CORRECT PRED. Cell {', i,',',j,'}; dendrite: ', dendrite_idx, \
                    #     '. Reinforcing ', np.count_nonzero(reinforcedSynapses),' synapses. Existing synapses= ',
                    #     tot_numSynapses, color.END)
                    # -------------------------------
        return winner_cells

    def case_net_decayPredictedColumns(self,
                                       other_predicted_cols=None,
                                       prev_state=None,
                                       prev_pred=None,
                                       prev_pred_dendrites=None):
        """
        Handles the case when a column is predicted but does not subsequently get activated in the current time step
        (may occur in case of multiple predictions or incorrect predictions).

        Parameters
        ----------
        other_predicted_cols : list of indices of other predicted columns.
        prev_state : binary MxN matrix of network's activity in the previous time step.
        prev_pred : binary MxN matrix of network's prediction at the previous
        time step.
        prev_pred_dendrites : dictionary storing indices of the dendrites on a cell
        which led to its predictive state in the previous time step.

        Returns
        -------
        None.
        """

        for j in other_predicted_cols:

            cells_i = np.where(prev_pred[:, j])[0]  # extract i-indices of all OTHER predicted cells in the column

            # Punish active dendrites for all of OTHER predicted cells in the minicolumn.
            for i in cells_i:

                # for indices of all dendrites that led to cell's prediction.
                for dendrite_idx in prev_pred_dendrites[(i, j)]:
                    _, _ = self.net_architecture[i, j].update_cell_dendritePermanences(dendrite_idx=dendrite_idx,
                                                                                       prev_state=prev_state,
                                                                                       decay=True)

    def update_net_dendrite_dutycycle(self):
        """
        Used to update the duty cycle of every dendrite in the network.
        - Decreases an existing dendrite's duty cycle by 1.
        - If the dendrite's duty cycle falls below 1, then it prunes the dendrite.

        Returns
        -------
        None
        """

        for i in range(self.M):
            for j in range(self.N):
                if self.net_architecture[i, j].get_cell_numDendrites() != 0:
                    for dendrite_idx in range(self.max_dendrites_per_cell):
                        if self.net_architecture[i, j].dendrites_dutyCycle[dendrite_idx] is not None:
                            self.net_architecture[i, j].dendrites_dutyCycle[dendrite_idx] -= 1
                        else:
                            continue

                    for dendrite_idx in range(self.max_dendrites_per_cell):
                        if self.net_architecture[i, j].dendrites_dutyCycle[dendrite_idx] is not None:
                            if self.net_architecture[i, j].dendrites_dutyCycle[dendrite_idx] <= 0:
                                self.net_architecture[i, j].dendrites_dutyCycle[dendrite_idx] = None
                                self.net_architecture[i, j].dendrites[dendrite_idx] = None
                        else:
                            continue
                else:
                    continue

    def normalize_net_permanence(self):
        """
        This function ensures that the permanence levels of all synapses in the network remains between 0 and 1
        at all times.

        Returns
        -------
        None
        """

        for i in range(self.M):
            for j in range(self.N):
                for dendrite in self.net_architecture[i, j].get_cell_allDendrites():
                    if dendrite is None:
                        continue
                    else:
                        dendrite[dendrite < 0] = 0.0
                        dendrite[dendrite > 1] = 1.0

    def get_NETWORK(self, char_onehot='all'):
        if char_onehot == 'all':
            return self.net_architecture
        else:
            return self.net_architecture[:, np.where(char_onehot)[0]]

    def get_net_dims(self):
        return self.M, self.N


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
