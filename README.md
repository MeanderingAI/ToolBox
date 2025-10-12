# Tool Box

The cmake generates shared objects which can be used with other projects for each of these categories of machine learning.
We also include gunit tests for the entire suite.

## Filters

Here we provide a Kalman Filter library which contains CPP implementation of the standard array of implementations;

1. Standard Kalman Filter
2. Extended Kalman Filter
3. Unscented Kalman Filter
4. Sequential Monte Carlo


## Generalized Linear Models

Current implementation includes Linear Regression with;

1. Stochastic Gradient Decent
2. Closed Form Solution

## Decision Trees

Here we provide the following Decision Tree Estimators;

1. Entropy 
2. Gini Index

With the following variations;

1. Rule Set
2. Random Forest
3. Boosted Trees

## Statistical Distributions

Here we provide the following Standard Statistical Distributions;

1. Bernoulli Distribution
2. Binomial Distribution
3. Categorical Distribution
4. Exponential Distribution
5. Gamma Distribution
6. Inverse Gaussian Distribution
7. Laplace Distribution
8. Multinomial Distribution
9. Normal Distribution
10. Poisson Distribution

These distributions have implementations of pdf, log_pdf, cdf, log_cdf, and sampling.

## Hidden Markov Model

Traditional implementation of forward and backward algorithm.

## Multi-Arm Bandit

Here we provide the following multi-arm bandit implementations;

1. Epsilon Greedy Agent
2. Decaying Epsilon Greedy Agent
3. Upper Confidence Bound Agent 
4. Thompson Sampling Agent

## Support Vector Machine

Here we provide implementation of support vector machines with the following kernels;

1. Linear
2. Polynomial
3. Gaussian 
4. Sigmoid

# Dependencies

## GSL

This library uses libgsl, on ubuntu it can be installed with

```
sudo apt install libgsl-dev
```

## Eigen

This library is automatically installed by cmake during compilation

## GoogleTest

This library is automatically installed by cmake during compilation

# Compiling

use the `./clean&build` script.

Generated libraries can be found in;

`build/src/lib*.so`
