class Module:
    path = './src/'
    head = [ ]
    test = [ ]


#klasy bazowe
pr = Module()
pr.path = ''
src = pr.path
pr.head = [ src+'Version.hpp', src+'ExceptionsFaif.hpp', src+'Value.hpp', src+'Point.hpp']

#klasy pomocnicze (utils)
utils = Module()
utils.path = 'utils/'
src = utils.path
utils.head = [ src+'Random.hpp', src+'RandomCustomDistr.hpp', src+'GaussEliminator.h', src + 'Power.hpp']

#klasy DNA
dna = Module()
dna.path = 'dna/'
src = dna.path
dna.head = [ src+'ExceptionsDna.h', src+'Nucleotide.h', src+'Chain.h', src+'EnergyNucleo.h',
             src+'SecStruct.h',
             src + 'FoldedMatrix.h', src+'FoldedChain.h', src + 'FoldedPair.h',
             src+'Codon.h', src+'CodonAminoTable.h' ]

#klasy DNA
hapl = Module()
hapl.path = 'hapl/'
src = hapl.path
hapl.head = [ src+'Loci.hpp' ]


#klasy timeseries
ts = Module()
ts.path = 'timeseries/'
src = ts.path
ts.head = [ src+'TimeSeries.hpp', src + 'TimeseriesExceptions.hpp', src + 'Transformations.hpp', src + 'Predictions.hpp', src + 'Discretizer.hpp' ]

#klasy learning
learn = Module()
learn.path = 'learning/'
src = learn.path
learn.head = [ src+'Belief.hpp', src + 'Fusion.hpp', src+'Classifier.hpp', src + 'Validator.hpp',
               src+'NaiveBayesian.hpp', src+'DecisionTree.hpp', src+'KNearestNeighbor.hpp',src + 'MLReg.hpp',src+'MachineLearningExceptions.hpp' ]

#przeszukiwanie
search = Module()
search.path = 'search/'
src = search.path
search.head = [ src+'Node.hpp', src+'Space.hpp',
                src+'TreeNodeImpl.hpp', src+'DepthFirst.h', src+'BreadthFirst.h', src+'UnifiedCost.h', src + 'AStar.h',
                src + 'HillClimbing.hpp', src+'EvolutionaryAlgorithm.hpp', src + 'VectorIndividual.hpp',
                src+'ExpectationMaximization.hpp' ]

modules = [ pr, utils, dna, hapl, ts, learn, search ]
