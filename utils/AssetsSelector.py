# %pip install qubovert
# %pip install qiskit-aer
# %pip install qiskit
# %pip install pylatexenc
# %pip install qiskit-ibm-runtime
import json
import numpy as np
from datetime import datetime
from qubovert import QUBO, boolean_var
from qubovert.sim import anneal_qubo
from qubovert.utils import qubo_to_matrix
import random
import math
import pandas as pd

from qiskit.circuit.library import qaoa_ansatz
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit, transpile

from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator, SamplerV2 as Sampler
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, kron, eye

def pauli_z_sparse():
    """Returns the sparse Pauli Z matrix."""
    return csr_matrix([[1, 0], [0, -1]])

Z = pauli_z_sparse()
I = eye(2, format='csr')  # Sparse identity matrix of size 2
def z_j_sparse(n, j):
    """Returns the sparse matrix Z_j for n qubits applied to qubit j."""

    # Start with the identity matrix for n qubits
    result = I

    for i in range(1, n + 1):
        if i == j:
            result = kron(result, Z, format='csr')  # Apply Z to qubit j
        else:
            result = kron(result, I, format='csr')  # Apply I to other qubits

    return result

class AssetSelector():
    def __init__(self, data,
                        date_limit:str=None,
                        max_assets:int=None):
        '''Given a file, it returns the data related to it,
        ignoring data before `date_limit`
        '''
        self.data = data 
        self.date_limit = date_limit
        self.solution = None
        self.qubo_model = None
        self.cost_hamiltonian = None
        self.ansatz = None
        ######################
        # DATA PREPROCESSING #
        ######################


        # Select only `max_assets` at random (for debuggin purposes)
        if max_assets:
            asset_names = list(self.data['assets'].keys())
            selected_asset_names = random.sample(asset_names, max_assets)
            
            for a in list(self.data['assets'].keys()):
                if a not in selected_asset_names:
                    self.data['assets'].pop(a)


        # Filter out all assets with a number of history items different than the trend (i.e. 751)
        # We recognize that this approach works well only in cases where the number of outliers is reduced
        # as in this case (9 out of ~900 assets).

        filtered_data = {key:self.data[key] for key in ['evaluation_date']}
        filtered_data['assets'] = {}
        for a in self.data['assets']:
            if len(self.data['assets'][a]['history']) == 751:
                filtered_data['assets'][a] = self.data['assets'][a]

        self.data = filtered_data
        
        # Filter by date
        for a in self.data['assets']:
            # print(a)
            self.data['assets'][a]['history'] = self.filter_dates_before(self.data['assets'][a]['history'], self.date_limit)

        
    def filter_dates_before(self, input_dict, specified_date):
        """
        Filters the input dictionary to include only entries with dates before the specified date.

        Parameters:
        input_dict (dict): A dictionary with dates as keys in 'yyyy-mm-dd' format.
        specified_date (str): A date in 'yyyy-mm-dd' format to filter the dictionary.

        Returns:
        dict: A new dictionary containing only the entries with dates before the specified date.
        """
        if specified_date is None:
            return input_dict
        # Convert the specified date string to a datetime object
        specified_date_obj = datetime.strptime(specified_date, '%Y-%m-%d')
        
        # Create a new dictionary to hold the filtered results
        filtered_dict = {}
        
        # Iterate through the input dictionary
        for date_str, data in input_dict.items():
            # Convert the date string to a datetime object
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Check if the date is before the specified date
            if date_obj < specified_date_obj:
                filtered_dict[date_str] = data
                
        return filtered_dict


    @property
    def covariance_matrix(self):
        '''Returns the covariance matrix of the assets in `data`
        using history up to date `date_limit`'''

        assets = list(self.data['assets'].keys())
        # number of history elements (assumed to be equal for all assets)
        num_history_dates = len(self.data['assets'][assets[0]]['history'])
        # print(num_history_dates)
        # number of assets we are taking into account
        N_considered_assets = len(self.data['assets'])

        # Each row of assets_matrix represents a variable, and each column a single observation of all those variables.
        assets_matrix = np.empty((N_considered_assets, num_history_dates))

        for i, asset in enumerate(assets):
            history = self.data['assets'][asset]['history']
            assets_matrix[i, :] = list(history.values())

        return np.corrcoef(assets_matrix)
    
    def compute_cost_hamiltonian(self):
        num_assets = self.covariance_matrix.shape[0]
        n = num_assets
        HC = np.zeros_like(z_j_sparse(n, 1))
        bigI = np.eye(z_j_sparse(n, 1).shape[0])

        for j in range(num_assets):
            HC = 1/2 * (bigI - z_j_sparse(n, j) + HC)

        delta = 1000

        triui = np.triu_indices_from(self.covariance_matrix, k=1)
        for t in range(len(triui[0])):
            j = triui[0][t]
            k = triui[1][t]
            HC = delta * 1/4 * (bigI - z_j_sparse(n, j) - z_j_sparse(n, k) + np.kron(z_j_sparse(n,j), z_j_sparse(n,k)))
            # HC = np.kron(z_j_sparse(n,j), z_j_sparse(n,k))

        self.cost_hamiltonian = SparsePauliOp.from_operator(HC)

    def compute_ansatz(self):
        assert self.cost_hamiltonian, "You must compute the cost hamiltonian first"
        ansatz = qaoa_ansatz(cost_operator=self.cost_hamiltonian, reps=2)
        ansatz.measure_all()
        self.ansatz = ansatz
    
    def solve(self):
        assert self.cost_hamiltonian, "You must compute the cost hamiltonian first"
        assert self.ansatz, "You must compute the cost ansatz first"
        ansatz = self.ansatz
        initial_gamma = np.pi
        initial_beta = np.pi/2
        init_params = [initial_gamma, initial_beta, initial_gamma, initial_beta]
        
        def cost_func_estimator(params, ansatz, hamiltonian, estimator):

            # transform the observable defined on virtual qubits to
            # an observable defined on all physical qubits
            isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

            pub = (ansatz, isa_hamiltonian, params)
            job = estimator.run([pub])

            results = job.result()[0]
            cost = results.data.evs

            objective_func_vals.append(cost)


            return cost
        
        backend = AerSimulator(method='statevector')
        ansatz = transpile(ansatz, backend)
        objective_func_vals = [] # Global variable
        with Session(backend=backend) as session:
            # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `session=`
            estimator = Estimator(mode=session)
            estimator.options.default_shots = 1000

            # Set simple error suppression/mitigation options
            estimator.options.dynamical_decoupling.enable = True
            estimator.options.dynamical_decoupling.sequence_type = "XY4"
            estimator.options.twirling.enable_gates = True
            estimator.options.twirling.num_randomizations = "auto"

            result = minimize(
                cost_func_estimator,
                init_params,
                args=(ansatz, self.cost_hamiltonian, estimator),
                method="COBYLA",
                tol=1e-2,
            )
            print(result)
        
        plt.figure(figsize=(12, 6))
        plt.plot(objective_func_vals)
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.show()
        optimized_circuit = ansatz.assign_parameters(result.x)
        optimized_circuit.draw('mpl', fold=False, idle_wires=False)
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = 10000

        # Set simple error suppression/mitigation options
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
        sampler.options.twirling.enable_gates = True
        sampler.options.twirling.num_randomizations = "auto"

        pub= (optimized_circuit, )
        job = sampler.run([pub], shots=int(1e4))
        counts_int = job.result()[0].data.meas.get_int_counts()
        counts_bin = job.result()[0].data.meas.get_counts()
        shots = sum(counts_int.values())
        final_distribution_int = {key: val/shots for key, val in counts_int.items()}
        final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}
        # print(final_distribution_int)
        def to_bitstring(integer, num_bits):
            result = np.binary_repr(integer, width=num_bits)
            return [int(digit) for digit in result]

        keys = list(final_distribution_int.keys())
        values = list(final_distribution_int.values())
        most_likely = keys[np.argmax(np.abs(values))]
        most_likely_bitstring = to_bitstring(most_likely, self.covariance_matrix.shape[0] + 1)
        # most_likely_bitstring.reverse()

        all_assets = list(self.data['assets'].keys())

        selected_assets = {ass: self.data['assets'][ass] for i, ass in enumerate(all_assets) if most_likely_bitstring[i] == 1}
        
        return selected_assets