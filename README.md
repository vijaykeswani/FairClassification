This repository provides the implementation for an algorithm that aims to reduce bias in classification.

The associated paper is the following :

Classification with Fairness Constraints: A Meta-Algorithm with Provable Guarantees : https://arxiv.org/abs/1806.06055

To run the algorithm, clone the repository and follow the given steps.

In this repository, we have implemented 3 algorithms. The difference between the algorithms is the fairness function they want to optimize. 

The fairness functions considered here are :
* Statistical parity
* False Discovery
* Statistical parity + False Discovery

To get a better idea of the algorithm and the constraint functions, we refer the reader to the paper.

The algorithms can be run on the pre-processed datasets provided (Adult and Compass). To run any of the above algorithms, simply run the corresponding file with the data folder as the argument : 

        $ python3 FalseDiscovery.py ../Data/data

This will run the algorithm for input fairness parameter 0.1, 0.2, ..., 1, and for each case output the accuracy over the test data and the observed fairness values.

## Fairness Comparison

We also provide another way to run the algorithm and compare it with other fair-classification algorithms. To do this, use the framework provided by https://github.com/algofairness/fairness-comparison/blob/master/README.md. The details of the implementation are provided in the FairnessComparison folder.

## Adding a new algorithm
To add a new algorithm based on a different constraint, simply extend the General.py file, and implement the functions for optimization based on the constraint function chosen. 
