faif-MLR  [![build-status](https://travis-ci.org/Mike-Now/faif-MLR.svg)](https://travis-ci.org/Mike-Now/faif-MLR)
========

A fork of the [faif](http://faif.sourceforge.net/) (Fast Artificial Intelligence Framework). 

This fork includes Multinomial Logistic Regression classifier, which uses Batch Gradient Descent for training.

# Build

    scons build
    
# Usage


Various usage examples can be found in the examples/ directory. Also check the original page ([faif](http://faif.sourceforge.net/)).

# Multinomial Logistic Regression Classifier

- learning algorithm : Batch Gradient Descent
- plain implementation : no Bold Driver, no Weight Decay and no Regularization
- accepts only Nominal values (as almost every algorithm in this framework)
