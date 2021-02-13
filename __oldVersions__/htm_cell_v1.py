import numpy as np


class HTM_CELL():
    
    """
    
    """
    
    def __init__(self, cellsPerColumn=None, numColumns=None, maxDendritesPerCell=None, maxSynapsesPerDendrite=None, 
                 nmdaThreshold=None, permThreshold=None, permInit=None, permInit_sd=None,
                 avgLen_reberString=None):
        
        """
        
        """
        
        self.M = cellsPerColumn 
        self.N = numColumns
        
        self.n_dendrites = maxDendritesPerCell
        self.n_synapses = maxSynapsesPerDendrite
        
        self.perm_th = permThreshold
        self.perm_init = permInit
        self.perm_init_sd = permInit_sd
        
        # list containing the matrices of potential synapses (permanence values) for each dendrite
        # of the HTM cell; numpy array of 32 MxN matrices, shape: (32,M,N)
        self.dendrites = np.array([np.random.normal(loc=self.perm_init, scale=self.perm_init_sd, size=[self.M, self.N])
                          for i in range(self.n_dendrites)], dtype=np.float64)
        
        self.avgLen_reberString = avgLen_reberString
        self.dutycycle = []
        
        
        return
    
    
    def get_cell_connSynapses(self):
        """
        For getting the connected synapses on all the dendrites of the cell.
        
        Returns
        -------
        A Boolean array of size (n_dendrites, M, N)

        """
        
        return np.array(self.dendrites > self.perm_th) # boolean list of 32 MxN binary matrices, 
                                                        # shape: (32,M,N)
            
        
    def get_cell_synaPermanences(self):
        """
        For getting the permanence values of all synapses on all the dendrites 
        of the cell.
        
        Returns
        -------
        A Boolean array of size (n_dendrites, M, N)

        """
        
        return self.dendrites