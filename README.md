# Implicit Acquisition of Simple Reber Grammar in a Neocortex-inspired Neural Network
(NOTE: Master's Thesis Code Repository)

### Abstract
The human brain seems to acquire sequential information – mainly, recognition and prediction of temporally-correlated patterns – almost seamlessly. Moreover, the acquisition is even implicit in some cases like language. In my thesis, I attempted to study this feature of the biological brain by addressing the question of implicit (unsupervised) learning of nontrivial and higher-order sequential information over time, which is also an important direction of inquiry that lacks a satisfactory solution in the problem domain of Sequence Learning.

I use Numenta’s [Hierarchical Temporal Memory (HTM) (2016)](https://numenta.com/neuroscience-research/research-publications/papers/why-neurons-have-thousands-of-synapses-theory-of-sequence-memory-in-neocortex/) – which is deploys several architectural and functional features of the neocortex in addition to utilizing Hebbian plasticity based learning techniques. The task used in the experiments is to implicitly learn higher-order temporal dependencies in sentences (strings) generated from an artificial grammar – known to model the implicit acquisition mechanisms of language processing in the human brain [5, 30, 32]. The used solution model exploits two computational principles found in the brain – first, sparse distributed encoding of information (temporal, in case of sequence learning); second, use of dendrites as nonlinear, locally trainable, independent subunits that detect these sparse synchronous activation patterns in the neuronal population [1, 13]. The HTM network is shown to learn long-range dependencies implicitly in an online fashion. However, it is found to have a variable-order memory that attempts to “memorize” sample sequences, instead of acquiring the structure and syntax of the Artificial Grammar.




Numenta's [Hierarchical Temporal Memory model (2016)] is used on the problem domain of Artificial Grammar Learning (AGL). Benchmarks used from AGL are: learning Simple Reber Grammar (SRG) transition rules; and, learning Embedded Reber Grammar (ERG) transition rules.

A detailed explanation of the project with the underlying motivation can be found in the presentation slides PDF file above.
