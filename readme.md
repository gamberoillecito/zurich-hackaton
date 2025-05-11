# README

## Overview

This project implements a quantum-inspired portfolio optimization algorithm using either QUBO (Quadratic Unconstrained Binary Optimization) formulation or QAOA based on a cost function hamiltonian. The script processes financial data, calculates a covariance matrix, and selects an optimal portfolio of assets using one of the two quantum approaches. The solution is saved as a JSON file containing the selected assets and their historical data.

## Features

- **Data Preprocessing**: Filters and preprocesses input data based on date and asset constraints.
- **Covariance Matrix Calculation**: Computes the covariance matrix for the given assets.
- **Portfolio Optimization**: Uses QUBO and simulated annealing to select an optimal portfolio of assets.
- **Result Export**: Saves the selected assets and their historical data to a JSON file.

## Requirements

The following Python libraries are required:

- `qubovert`
- `qiskit-aer`
- `qiskit`
- `numpy`
- `pandas`

Install the dependencies using pip:

```bash
pip install qubovert qiskit-aer qiskit numpy pandas
```

## Usage

1. **Input Data**: Provide a JSON file containing asset data with historical prices. The file should have the following structure:
   ```json
   {
       "evaluation_date": "yyyy-mm-dd",
       "assets": {
           "asset_name_1": {
               "history": {
                   "yyyy-mm-dd": price,
                   ...
               }
           },
           ...
       }
   }
   ```

2. **Run the Script**: Execute the script to process the data and optimize the portfolio. Example:
   ```bash
   python giacomo.py
   ```

3. **Output**: The selected assets and their historical data will be saved to `selected_assets.json`.

## Functions

### `filter_dates_before(input_dict, specified_date)`
Filters a dictionary to include only entries with dates before a specified date.

### `read_input_file(input_file_name, date_limit=None, max_assets=None)`
Reads and preprocesses the input JSON file. Filters data by date and limits the number of assets.

### `covariance_matrix(data)`
Calculates the covariance matrix for the given assets.

### `get_selected_assets(data, sol)`
Parses the optimization solution and saves the selected assets to a JSON file.

## Example Workflow

1. Load the input data using `read_input_file`.
2. Compute the covariance matrix using `covariance_matrix`.
3. Define a QUBO model and solve it using `anneal_qubo`.
4. Extract the selected assets using `get_selected_assets`.

## Notes

- The script assumes that all assets have the same number of historical data points.
- The portfolio size and other parameters can be adjusted in the script.

## License

This project is licensed under the MIT License.