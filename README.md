# Implicit Acquisition of Simple Reber Grammar in a Neocortex-inspired Neural Network
(My master's thesis' code repository)

## Project Summary
The human brain seems to acquire sequential information – mainly, recognition and prediction of temporally-correlated patterns – almost seamlessly. Moreover, the acquisition is even implicit in some cases like language. In my thesis, I attempt to study this feature of the biological brain by addressing the question of implicit (unsupervised) learning of nontrivial and higher-order sequential information over time.

I use Numenta’s [Hierarchical Temporal Memory (HTM) (2016)](https://numenta.com/neuroscience-research/research-publications/papers/why-neurons-have-thousands-of-synapses-theory-of-sequence-memory-in-neocortex/) – which deploys several architectural and functional features of the neocortex in addition to utilizing Hebbian plasticity based learning techniques. The task used in the experiments is to implicitly learn higher-order temporal dependencies in sentences (strings) generated from an artificial grammar, that is known to model the implicit acquisition mechanisms of language processing in the human brain.

## Repository Structure
`experiment_modules/` directory contains python modules defining interfaces for an HTM cell, an HTM network, a Reber Grammar Generator and an Experimentor; along with all their corresponding functions used to perform different aspects of the experiment.


#### Description of Issues 001-005
~~~
Issue Description:
------------
Issue 001: 
    When a column bursts, but no (matching) dendrite with connections to the previous timestep's activity 
    are found AND when all HTM cells in a given minicolumn run out of their capacity to grow any new
    dendrite (given by 'maxDendritesPerCell').
    	
Issue 002:
    When a dendrite has more synapses than its capacity given by 'maxSynapsesPerDendrite'.
    
Issue 003:
    When multiple matching dendrites are found in a bursting column.
    
Issue 004:
    To be read in the same context as Issue 001. See htm_net.py.
    
Issue 005:
    This issue reports a fundamental flaw in the learning of SDRs. If the total number of cells with 
    permanence reinforcement on any one of their dendrites at any given timestep during execcution
    falls below the set NMDA threshold of the network, issue 005 is reported at the output terminal.
    It breaks the execution of the program for the current reber string and starts execution from the
    next reber string in the input stream.
    In the current implementation of HTM, this issue is generally found to be in 5% of the total
    number of reber strings in the inputstream.
~~~

## Thesis
The full thesis can be found [here](https://docs.google.com/document/d/10CVceFrXVdygoLiY0-jKl_dnbHttWX6Iyar5BQjbg8I/edit?usp=sharing).

## Presentation Slides
Presentation slides can be found [here](https://github.com/TaherHabib/sequence-learning-model/blob/master/Modelling%20Implicit%20Acquisition%20of%20Sequential%20Information%20Using%20a%20Neocortical%20Neural%20Network%20Hierarchical%20Temporal%20Memory.pdf).
