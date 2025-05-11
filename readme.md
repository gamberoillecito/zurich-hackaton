# README

## Overview

This project implements a quantum-inspired portfolio optimization algorithm using either QUBO (Quadratic Unconstrained Binary Optimization) formulation or QAOA approximation based on a cost function hamiltonian. 
The goal was to be able to fully exploit quantum advantage in the choice of the assets, having the possibility to choose the most advantageous method.

The script pre-processes financial data, calculates a covariance matrix, and selects an optimal portfolio of assets using one of the two quantum approaches. 

The solution is saved as a JSON file containing the selected assets and their historical data.

## Features

- **Data Preprocessing**: Tools for filtering and preprocessing financial data based on date and asset constraints. This step was implemented as a way to reduce the size of the dataset aiming at preserving a statistically relevant subset.
- **Covariance Matrix Calculation**: Scripts to compute covariance matrices for asset returns.
- **Portfolio Optimization**:
  - QUBO formulation with simulated annealing.
  - QAOA-based optimization using cost function Hamiltonians.
- **Monte Carlo Simulations**: Generate random paths for asset prices.
- **Markowitz Optimization**: Classical portfolio optimization using the Markowitz model.
- **Machine Learning**: Includes LSTM-based models for asset price prediction.
- **Result Export**: Saves selected assets and their historical data to JSON files.

## Relevant functions and choices

### `DataPruner`

This class performs a coarse reduction of the dataset while trying to preserve a diversified subset of data.
The main reason why we decided to implement this feature is to specify an upper bound of the dataset size to be tuned according to the hardware capabilities of the machine. In this way it would be possible to operate on dataset that exceed the available resources without compromising to much on data quality.

The way the pruner works is by looking for different aspects by which to characterise the data (e.g.`region`, `sector`, `industry`) and trying to pick a varied subset that represents these categories.

### `AssetSelector`

This class is in charge of the selection of the optimal portfolio, it does so by first excluding the dates in the future to avoid the accidental use of "unrealistic" data.

After this step we build the covariance matrix and proceed with the definition of the models to be solved. As stated before we used to alternative approaches to explore different aspects of quantum computing:

- QUBO: the QUBO formulation is know to be suited for the execution on quantum annealers as well as gate-based quantum computers and offers a simple formalism to model this problem.
- Ising model: although similar to the previous formalism, this model allows ((FONTE)) for an easier implementation in gate-based computers. Despite our efforts though, this implementation resulted in a worse solution due to difficulties encoutered. We want to highlight that the bottleneck is in the algorithm used to create the circuit and not in the circuit itself, giving more hopes for a better implementation.



## Example Workflow

1. Load the input data using `read_input_file`.
2. Compute the covariance matrix using `covariance_matrix`.
3. Define a QUBO model and solve it using `anneal_qubo`.
4. Extract the selected assets using `get_selected_assets`.

## Notes

- The script assumes that all assets have the same number of historical data points.
- The portfolio size and other parameters can be adjusted in the script.
