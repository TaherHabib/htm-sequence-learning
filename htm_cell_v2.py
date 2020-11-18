import numpy as np
import pandas as pd


class HTM_CELL():
    
    """
    
    """
    
    def __init__(self, cellsPerColumn=None, numColumns=None, 
                 maxDendritesPerCell=None, maxSynapsesPerDendrite=None, 
                 nmdaThreshold=None, permThreshold=None, permInit=None, permInit_sd=None):
        
        self.M = cellsPerColumn
        self.N = numColumns 

        self.maxDendritesPerCell = maxDendritesPerCell
        self.maxSynapsesPerDendrite = maxSynapsesPerDendrite # max. number of CONNECTED synapses on a 
                                                             # single dendritic segment.
        self.nmdaThreshold = nmdaThreshold
        self.permThreshold = permThreshold
        self.permInit = permInit
        self.permInit_sd = permInit_sd
        
        # list containing the matrices of potential synapses (permanence values) for each dendrite
        # of the HTM cell; shape: (<maxDendritesPerCell>,M,N). There are NO dendrites initially.
        self.dendrites = [None for i in range(maxDendritesPerCell)]
        
        self.dutycycle = []
        
        return
    
    
    def get_cell_numDendrites(self):
        
        numDendrites = 0
        for i in range(self.maxDendritesPerCell):
            if self.dendrites[i] is None:
                continue
            else:
                numDendrites+=1
        
        return numDendrites
    
    
    def get_cell_numSynapsesOnDendrite(self, dendrite_idx):
        
        numSynapsesOnDendrite = None
        
        return numSynapsesOnDendrite
    
    
    def grow_cell_newDendrite(self, presynaptic_WinnerCells):
        
        newDendrite = np.array(np.random.normal(loc=self.permInit, scale=self.permInit_sd, size=[self.M, self.N]), 
                               dtype=np.float64)
        
        # 'newDendrite' will ONLY have connections (at permInit level) to 'presynaptic_WinnerCells'.
        # The rest will be all 0.0.
        newDendrite = newDendrite*presynaptic_WinnerCells
        
        # 'newDendrite' will be assigned to the first "non-NONE" index in self.dendrites array
        for i in range(self.maxDendritesPerCell):
            if self.dendrites[i] is None:
                dendrite_idx = i
                break
            else:
                continue
            
        self.dendrites[dendrite_idx] = newDendrite

        return
    
    
    def get_cell_connectedSynapses(self):
        """
        For getting the connected synapses on all the dendrites of the cell.
        
        """
        
        if len(self.dendrites) == 0:
            return None
        else:
            return np.array(np.array(self.dendrites) > self.perm_th) # boolean list of <n_dendrites> MxN 
                                                                     # binary matrices.
        
            
    def get_cell_dendrites(self):
        """
        For getting the permanence values of all synapses on all the dendrites 
        of the cell.
        
        """
        
        return np.array(self.dendrites)
    
    
    def update_cell_dutycycle(self, prev_state=None, prev_pred=None):
        
        
        return