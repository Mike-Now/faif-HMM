FAIF - Fast(Funny) Artificial Intelligence Framework
----------------------------------------------------

The C++ header only library for bioinformatics and artificial intelligence.

This library defines basic abstractions (like nomnial value, domain, point, space, state, etc.)
and implements classifier algorithms (Naive Bayesian, Decision Tree ID3 inspired, K Nearest Neighbors);
cross validator;
space search methods ( Evolutionary Algorithm, Expectaiton-Maximization, Hill Climbing);
timeseries primitives (TimeSeriesDigit, TimeSeriesReal, linear resampling, autoregressive predictor);
DNA primitives (Nucleotide, Chain, EnergyNucleo, SecStruct, FoldedChain, FoldedPair, Codon, CodonAminoTable,
Nussinov algorithm); Random generators; Gaussian eliminator.
Serialization to text or XML based on boost::serialization.

The FAIF abstractions and algorithms are generic (as STL and boost).

The main goal is education of AI, where simple and portable C++ library is needed.
The library will be used also for broad spectrum of applications
where some artificial intelligence algorithm are required.

Implementation:
--------------
C++ ISO 2003,

Dependencies:
-----------
boost 1.35 or later



