"""
MODULE DESCRIPTION:
-------------------

Implementation of the HTM cell interface using HTM_CELL() class.
    
"""

import numpy as np
import random

from .utils import dot_prod, get_idx_nonZeroElements, get_num_columns
from htm_sequence_learning.reber_grammar.graph_configs import chars


class HTM_CELL:

    def __init__(self,
                 columns_per_char=None,
                 cells_per_column=None,
                 max_dendrites_per_cell=None,
                 max_synapses_per_dendrite=None,
                 nmda_threshold=None,
                 permanence_threshold=None,
                 permanence_init=None,
                 permanence_init_sd=None,
                 permanence_decrement=None,
                 permanence_increment=None,
                 permanence_decay=None,
                 max_dendrite_dormancy=None):

        self.M = cells_per_column
        self.k = columns_per_char
        self.N = get_num_columns(columns_per_char, chars)
        self.max_dendrites_per_cell = max_dendrites_per_cell
        self.max_synapses_per_dendrite = max_synapses_per_dendrite  # max. number of CONNECTED synapses on a
        # single dendritic segment.
        self.nmda_threshold = nmda_threshold
        self.permanence_threshold = permanence_threshold
        self.permanence_init = permanence_init
        self.permanence_init_sd = permanence_init_sd
        self.permanence_decrement = permanence_decrement
        self.permanence_increment = permanence_increment
        self.permanence_decay = permanence_decay
        self.max_dendrite_dormancy = max_dendrite_dormancy
        # self.highDecayFactor = 30

        # list containing the (numpy array) MxN matrices of potential synapses (permanence values) for each dendrite
        # of the HTM cell; shape: (<maxDendritesPerCell>,M,N). There are NO dendrites initially.
        self.dendrites = self.init_cell_dendrites
        self.dendrites_duty_cycle = self.init_dendrites_duty_cycle

    @property
    def init_cell_dendrites(self):
        return [None for i in range(self.max_dendrites_per_cell)]

    @property
    def init_dendrites_duty_cycle(self):
        return [None for i in range(self.max_dendrites_per_cell)]

    def get_cell_numDendrites(self):
        """
        Returns the number of initialized dendrites on the cell.
        """

        numDendrites = 0
        for dendrite in self.dendrites:
            if dendrite is None:
                continue
            else:
                numDendrites += 1

        return numDendrites

    def get_cell_numUnusedDendrites(self):
        """
        Returns the capacity to grow/initialize new dendrites on the cell
        """

        num_UnusedDendrites = 0
        for dendrite in self.dendrites:
            if dendrite is None:
                num_UnusedDendrites += 1
            else:
                continue

        return num_UnusedDendrites

    def get_cell_numSynapsesOnDendrite(self, dendrite_idx=None):
        """
        Returns the number of synapses on the dendrite 'dendrite_idx' 
        of the cell. 
        
        Parameters
        ----------
        dendrite_idx : int, 0-<maxDendritesPerCell>, the index of the dendrite
        on the cell.
        """

        return np.count_nonzero(self.dendrites[dendrite_idx])

    def grow_cell_newDendrite(self, prev_winner_cells=None):
        """
        Grows a new dendrite on the cell with synaptic connections to the winner cells from the previous time step.
        The new connections are all initialized with a permanence value of 'permInit'.

        Parameters
        ----------
        prev_winner_cells : MxN binary matrix of previous time step's winner cells.

        Returns
        -------
        newDendrite_idx : int, 0-<maxDendritesPerCell>, index of the newly grown/initialized dendrite.
        """

        newDendrite = np.array(np.random.normal(loc=self.permanence_init, scale=self.permanence_init_sd,
                                                size=[self.M, self.N]), dtype=np.float64)

        # A Dendrite will only have maximum <maxSynapsesPerDendrite> synapses. Hence, if there are more than <k>
        # winner cells in 'prev_winnerCells', then <k> are randomly chosen for synaptic connection.
        if np.count_nonzero(prev_winner_cells) > self.k:
            prev_winnerCells_ = np.zeros(shape=[self.M, self.N])
            random.seed()
            list_prev_WinnerCells_idx = random.choices(get_idx_nonZeroElements(prev_winner_cells), k=self.k)
            for i in list_prev_WinnerCells_idx:
                prev_winnerCells_[i] = 1

        else:
            prev_winnerCells_ = prev_winner_cells

        newDendrite = newDendrite * prev_winnerCells_
        # 'newDendrite' will ONLY have connections (at permInit level) to 'presynaptic_WinnerCells'. The rest will be
        # all 0.0. Any dendrite will have atmost <maxSynapsesPerDendrite> synapses.

        newDendrite_idx = None

        # 'newDendrite' will be assigned to the first "non-NONE" index in 
        # self.dendrites array
        for i in range(self.max_dendrites_per_cell):
            if self.dendrites[i] is None:
                newDendrite_idx = i
                self.dendrites[newDendrite_idx] = newDendrite
                self.dendrites_duty_cycle[newDendrite_idx] = self.max_dendrite_dormancy
                break
            else:
                continue

        return newDendrite_idx

    def grow_cell_newSynapsesOnDendrite(self,
                                        dendrite_idx=None,
                                        prev_winner_cells=None):
        """
        Given a specific dendrite, this function grows new synapses connecting to previous time step's winner cells.
        The total number of synapses on a dendrite are always limited to 'maxSynapsesPerDendrite'. As a result of an
        upper limit to the number of synapses on a dendrite, it is not always possible to grow synapses with all the
        cells in 'prev_winnerCells'. The function selects at random which cells from among 'prev_winnerCells' should be
        chosen to initialize a synapse to on the current dendrite 'dendrite_idx'.

        Parameters
        ----------
        dendrite_idx : int, 0-<maxDendritesPerCell>, index of the dendrite.
        prev_winner_cells : MxN binary matrix of previous time step's winner cells.

        Returns
        -------
        numNewSynapses : int, number of new synapses grown/initialized on the dendrite.
        list_prev_WinnerCells_idx : list of indices (m,n) of the previous time step's winner cells which were not
        chosen for synaptic connection to the current dendrite 'dendrite_idx'.
        """

        list_prev_WinnerCells_idx = get_idx_nonZeroElements(prev_winner_cells)

        newSynapsesCapacity = self.max_synapses_per_dendrite - self.get_cell_numSynapsesOnDendrite(dendrite_idx)
        numNewSynapses = 0

        while len(list_prev_WinnerCells_idx) > 0 and newSynapsesCapacity > 0:
            preSynapticWinnerCell = random.choice(list_prev_WinnerCells_idx)

            # If there is no synapse existing between 'preSynapticWinnerCell' and the 'dendrite_idx',
            # create one synapse!
            if self.dendrites[dendrite_idx][preSynapticWinnerCell] == 0.0:
                self.dendrites[dendrite_idx][preSynapticWinnerCell] = np.random.normal(loc=self.permanence_init,
                                                                                       scale=self.permanence_init_sd)
                self.dendrites_duty_cycle[dendrite_idx] = self.max_dendrite_dormancy
                newSynapsesCapacity -= 1
                numNewSynapses += 1

            list_prev_WinnerCells_idx.remove(preSynapticWinnerCell)

        return numNewSynapses, list_prev_WinnerCells_idx

    def update_cell_dendritePermanences(self,
                                        dendrite_idx=None,
                                        prev_state=None,
                                        decay=False):
        """
        Updates the permanence values of synaptic connection on the given dendrite of the cell. If 'decay' is True,
        it decays the permanence values; else it reinforces them.

        Parameters
        ----------
        dendrite_idx : int, 0-<maxDendritesPerCell>, index of the dendrite.
        prev_state : MxN binary matrix of previous time step's active cells.
        decay : boolean, choice of decaying/reinforcing the synaptic connections.
    
        Returns
        -------
        prev_state_ : MxN binary matrix of cells in the network to which synaptic connections were either
        reinforced/decayed.
        """
        prev_state_ = (self.dendrites[dendrite_idx] * prev_state) > 0
        prev_state_ = prev_state_.astype(int)

        if decay:
            self.dendrites[dendrite_idx] = self.dendrites[dendrite_idx] - self.permanence_decay * prev_state_
        # -------------------------------
        # elif high_decay:
        #     self.dendrites[dendrite_idx] = self.dendrites[dendrite_idx] - self.highDecayFactor*self.perm_decay*prev_state_
        # -------------------------------
        else:
            self.dendrites[dendrite_idx] = self.dendrites[dendrite_idx] + self.permanence_increment * prev_state_ \
                                           - self.permanence_decrement

        # Pruning Negative Permanence values (setting to 0.0)
        self.dendrites[dendrite_idx][self.dendrites[dendrite_idx] < 0] = 0.0

        # Pruning Positive Permanence values (setting to 1.0)
        self.dendrites[dendrite_idx][self.dendrites[dendrite_idx] > 1] = 1.0

        return prev_state_, self.get_cell_numSynapsesOnDendrite(dendrite_idx)

    def undo_cell_dendriteReinforcement(self,
                                        dendrite_idx=None,
                                        reinforced_synapses=None):
        """
        Function to undo the reinforcement of synaptic connections on a given dendrite of the cell.
        
        Parameters
        ----------
        dendrite_idx : int, 0-<maxDendritesPerCell>, index of the dendrite.
        reinforced_synapses : MxN binary matrix of cells in the network synapses
        to which were reinforced/decayed on the given dendrite.

        Returns
        -------
        None
        """

        nonzero_synapses = np.zeros(shape=[self.M, self.N])
        list_reinforcedSynapses_idx = get_idx_nonZeroElements(reinforced_synapses)
        list_nonzeroSynapses_idx = get_idx_nonZeroElements(self.dendrites[dendrite_idx] > 0)
        list_nonzeroSynapses_idx = set(list_nonzeroSynapses_idx) - \
                                   set(list_nonzeroSynapses_idx).intersection(list_reinforcedSynapses_idx)

        for i in list_nonzeroSynapses_idx:
            nonzero_synapses[i] = 1

        self.dendrites[dendrite_idx] = self.dendrites[dendrite_idx] - self.permanence_increment * reinforced_synapses + \
                                       self.permanence_decrement * nonzero_synapses

    def reset_cell_dendrite(self,
                            dendrite_idx=None,
                            reset_synapses=None):
        """
        Used to reset synaptic permanence values of certain synaptic connections, given by 'resetSynapses', on the dendrite.

        Parameters
        ----------
        dendrite_idx : int, 0-<maxDendritesPerCell>, index of the dendrite.
        reset_synapses : MxN binary matrix of cells in the network synapses to which are to be reset to the 'initPerm'
        permanence value, on the given dendrite.

        Returns
        -------
        None
        """

        mask_otherSynapses = (1 - reset_synapses) * self.dendrites[dendrite_idx]

        self.dendrites[dendrite_idx] = mask_otherSynapses + np.array(np.random.normal(loc=self.permanence_init,
                                                                                      scale=self.permanence_init_sd,
                                                                                      size=[self.M, self.N]),
                                                                     dtype=np.float64) * reset_synapses
        self.dendrites_duty_cycle[dendrite_idx] = self.max_dendrite_dormancy

    def get_cell_connectedSynapses(self):
        """
        For getting the connected synapses on all the dendrites of the cell.
        """

        cell_connectedSynapses = []
        for dendrite in self.dendrites:
            if dendrite is None:
                cell_connectedSynapses.append(None)
            else:
                cell_connectedSynapses.append(dendrite > self.permanence_threshold)

        return cell_connectedSynapses  # numpy array of length <maxDendritesPerCell> of either 'None' elements or
        # MxN (numpy) boolean matrices.

    def get_cell_predictivity(self, net_state=None):
        """
        Checks if the cell is in a predictive state, given the current 
        time step's network activity.

        Parameters
        ----------
        net_state : MxN binary matrix of current time step's active cells.

        Returns
        -------
        cell_predictivity : boolean, TRUE if the cell is in a predictive state.
        pred_dendrites : array containing indices of dendrites on the cell which
        led to its predictive state.
        """

        # 'dendritesSpikes' will be a list containing <maxDendritesPerCell> elements, either 'None' if the dendrites
        # have NO synapses; OR, a boolean value.
        dendritesSpikes = []
        cell_predictivity = False
        pred_dendrites = None  # stores the indices of all dendrites that led to the prediction of the cell.

        cell_connectedSynapses = self.get_cell_connectedSynapses()

        for dendrite in cell_connectedSynapses:
            if dendrite is None:
                dendritesSpikes.append(None)
            else:
                # 'is_nmdaSpike' is a numpy boolean array of length 1, with either a True or False.
                is_nmdaSpike = dot_prod(net_state, dendrite) > self.nmda_threshold
                dendritesSpikes.append(is_nmdaSpike[0])

        if any(dendritesSpikes):
            cell_predictivity = True
            pred_dendrites = np.where(dendritesSpikes)[0]  # 1D numpy array of max. possible length
            # <maxDendritesPerCell>, with integer entries
            # between 0 and <maxDendritesPerCell>-1.
            for dendrite_idx in pred_dendrites:
                self.dendrites_duty_cycle[dendrite_idx] = self.max_dendrite_dormancy
        return cell_predictivity, pred_dendrites

    def get_cell_allDendrites(self):
        """
        For getting the permanence values of all synapses on all the dendrites of the cell.
        """

        return np.array(self.dendrites, dtype=object)
