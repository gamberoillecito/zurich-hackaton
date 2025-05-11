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

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp
from qiskit import QuantumCircuit, transpile

from qiskit_ibm_runtime import Session, EstimatorV2 as Estimator, SamplerV2 as Sampler
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

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

        
        return  self.data
        pass

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

    @property
    def qubo_model(self):
        model = QUBO()
        P = 0
        # bool_vars = np.zeros_like(cvm).tolist()

        triui = np.triu_indices_from(self.covariance_matrix, k=1)
        for k in range(len(triui[0])):
            model = QUBO()
            self.covariance_matrix = 0
            x = [boolean_var(f"x_{i}") for i in range(self.covariance_matrix.shape[0])]
            for i, row in enumerate(self.covariance_matrix):
                for j, weight in enumerate(row):
                    if weight != 0:
                        # apart for the weight this is the formula on the pdf
                        model += weight*(2*x[i]*x[j] - x[i] - x[j])
            model = -model
        return model

    def solve_qubo(self):
        self.solution = anneal_qubo(self.qubo_model, num_anneals=10).best

    def get_selected_assets(self):
        '''Given the solution of the optimization,
        it returns a dataset with name and history of the 
        selected assetes'''

        assert self.solution != None, "No solution present, have you run a solver?"


        asset_names = list(self.data['assets'].keys())
        def parse_bool_var(x):
            name, i, j = x.split('_')
            if name == 'x':
                return int(i), int(j)
            return None, None 
        
        selected_assets_ids = ()
        for x in self.solution.state:
            if self.solution.state[x] == 1:
                i, j = parse_bool_var(x)
                if i is not None and j is not None:
                    selected_assets_ids = (*selected_assets_ids, i, j)
        
        selected_assets = {asset_names[i]: self.data['assets'][asset_names[i]] for i in selected_assets_ids}

        return selected_assets

    def solve_ising(self):
        qubo = self.model.to_qubo()
        qubo_mat_dim = qubo.num_binary_variables
        # np_qubo = np.zeros((qubo_mat_dim,qubo_mat_dim))

        pauli_list = []
        # Z = ZGate().to_matrix()
        # HC = np.zeros_like(np_qubo)
        for i in range(qubo_mat_dim):
            for j in range(qubo_mat_dim):
                if (qubo[i,j] == 0):
                    continue
                
                s = ['I' for i in range(qubo_mat_dim)]
                s[i] = 'Z'
                s[j] = 'Z'
                pauli_list.append((''.join(s), qubo[i,j]))

        cost_hamiltonian = SparsePauliOp.from_list(pauli_list)
        ansatz = QAOAAnsatz(cost_hamiltonian, reps=2)


        # qubo_to_ising(model).decompose(reps=3).draw('mpl', scale=0.2, fold=200)
        ansatz.measure_all()
        # ansatz.draw('mpl')
        # ansatz.decompose(reps=3).draw('mpl', scale=0.2, fold=200)
        ansatz.parameters

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
                args=(ansatz, cost_hamiltonian, estimator),
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



        # If using qiskit-ibm-runtime<0.24.0, change `mode=` to `backend=`
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


        # auxiliary functions to sample most likely bitstring
        def to_bitstring(integer, num_bits):
            result = np.binary_repr(integer, width=num_bits)
            return [int(digit) for digit in result]

        keys = list(final_distribution_int.keys())
        values = list(final_distribution_int.values())
        most_likely = keys[np.argmax(np.abs(values))]
        most_likely_bitstring = to_bitstring(most_likely, cvm.shape[0])
        most_likely_bitstring.reverse()

        print("Result bitstring:", most_likely_bitstring)


        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams.update({"font.size": 10})
        final_bits = final_distribution_bin
        values = np.abs(list(final_bits.values()))
        top_4_values = sorted(values, reverse=True)[:4]
        positions = []
        for value in top_4_values:
            positions.append(np.where(values == value)[0])
        fig = plt.figure(figsize=(11, 6))
        ax = fig.add_subplot(1, 1, 1)
        plt.xticks(rotation=45)
        plt.title("Result Distribution")
        plt.xlabel("Bitstrings (reversed)")
        plt.ylabel("Probability")
        ax.bar(list(final_bits.keys()), list(final_bits.values()), color="tab:grey")
        for p in positions:
            ax.get_children()[int(p)].set_color("tab:purple")
        plt.show()





