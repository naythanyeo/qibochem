"""
Measurement protocols evaluate observables on a prepared circuit.

The protocol object is intended as a lightweight wrapper around expectation
value evaluation, especially for workflows that need to evaluate multiple
observables at once.

Intended usage:

    values = protocol.evaluate(circuit, observables)

Inputs:
    circuit:
        A circuit that prepares the quantum state to evaluate.

    observables:
        Either a dictionary of observables or a single observable. Observables
        are currently Qibo Hamiltonian objects.

Output:
    values:
        Expectation values in the same structure as the input. If
        ``observables`` is a dictionary, this is a dictionary with the same
        keys. If ``observables`` is a single Hamiltonian, this is the
        expectation value itself.

Future considerations:
    Observables may become a separate class so protocols can act as compilers
    for different observable types, such as spin observables. Adaptive-shot
    protocols are still TBC.
"""

import re
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import numpy as np
from qibo import Circuit
from qibochem.measurement.optimization import measurement_basis_rotations, group_commuting_terms
from qibochem.measurement.result import expectation_from_samples, v_expectation
from qibochem.measurement.shot_allocation import allocate_shots

@dataclass
class StateVectorProtocol:
    """Protocol for evaluating observables on a statevector simulator.
    Each observables is a qibo.hamiltonian object
    """

    def evaluate(self, circuit, observables):
        """Before evaluation, circuit and observables must be specified"""
        if circuit is None:
            raise ValueError("Specifify a circuit to evaluate")
        if observables is None:
            raise ValueError("Specifify observables to evaluate")

        if not isinstance(observables, Mapping):
            return observables.expectation(circuit)

        return {
            key: observable.expectation(circuit)
            for key, observable in observables.items()
        }

# ------------------------------------------------------------
# BaseMeasurementProtocol
# ------------------------------------------------------------
@dataclass
class BaseMeasurementProtocol:
    grouping: str = "qwc"
    """
    Base measurement class for measurements with rotation
    After initialised, circuit with final_state can be re run to save overhead cost
    """
    def evaluate(self, circuit, observables) -> dict:
        """
        INPUT: circuits and observables 
        OUTPUT: expectation values in nested dictionaries 
        Requires subclass to define the sampling method to read in probabilities 
        Rough process
        1) Check if circuit.final_state exist, if not run it
        2) loop through every observable term 
        3) for every observable, group commuting terms 
        4) For every group, 
        - rotate final state into measurement basis
        - compute exact probabilities 
        - pass into subclass sampling method called _sample_groups 
        - output of sample_groups is a dictionary {10k_s1: , 10k_s2: ...}
        5) For all the keys in sample_groups, evaluate all the terms in observable 
        6) now my "output" is a dictionary {10k_s1 ...}. at every group, add on to this dictionary key value 
        7) each observable will have a dictionary of evaluated expectations
        8) reorder the nested dictionaries 
        """

        if circuit._final_state is None: # Re-run circuit only if it has not been run yet
            circuit()
        final_state = circuit._final_state
        self.n_qubits = circuit.nqubits
        # Generate z_vectors for measurement later
        self._generate_z_vectors()
        # Handles single input observables 
        if not isinstance(observables, Mapping):
            observables = {"O1": observables}
        # Loops through all the observables 
        full_observables_data = defaultdict(dict)
        for observable_key, observable in observables.items():
            # Output of this observable will be a dictionary of samples 
            sample_observable_expectations = defaultdict(float)
            # Group commuting terms
            grouped_terms = measurement_basis_rotations(observable, grouping=self.grouping)
            for group_expression, measurement_gates in grouped_terms:
                rotated_final_state = self._rotate_final_state(measurement_gates, final_state)
                exact_probabilities = np.abs(rotated_final_state) ** 2
                # From the exact_probabilities, do the sampling here
                # Sampled_probabilities is a dictionary with keys for different samples
                sampled_probabilities = self._sample_probabilities(exact_probabilities)
                for sampling_key, probability_vector in sampled_probabilities.items():
                    expectation_value = self._probabilities2expectation(probability_vector, group_expression)
                    sample_observable_expectations[sampling_key] += expectation_value
            full_observables_data[observable_key] = sample_observable_expectations
        return self._format_output(full_observables_data)

    def _rotate_final_state(self, measurement_gates, final_state):
        """
        Input: measurement_gates and final_state
        Output: rotated_final_states
        """
        rotation_circuit = Circuit(self.n_qubits)
        rotation_circuit.add(measurement_gates)
        result = rotation_circuit(initial_state=final_state)
        return result.state()

    def _sample_probabilities(self, exact_probabilities):
        """
        Defined at a subclass level for how to sample the probabilities
        """
        raise NotImplementedError

    def _generate_z_vectors(self):
        """
        The tensor product sequence will produce probabilities sorted lexicographically 
        Evaluating Z0 for eg, will yield eigenvalue 1 for the first 2^(n-1) terms, then -1 for the rest 
        Every successive term Zq will "swap" eigenvalues twice as often because binary 
        So the swapping length between eigenvalues is 2^(n-q-1)
        Pre calculate and cache z_vectors so that you can reconstruct measurement easier
        """
        n = self.n_qubits
        z_vectors = {}
        for q in range(n):
            swap_length = 2 ** (n - q - 1) 
            block = np.array([1] * swap_length + [-1] * swap_length)
            repeats = 2 ** q
            z_vectors[q] = np.tile(block, repeats)
        self._z_vectors = z_vectors
    
    def _probabilities2expectation(self, probability_vector, group_expression):
        # Read in the group_expression and probability_vector, output overall summed expectation_value
        expectation_value = 0.0
        for term, coeff in group_expression.as_coefficients_dict().items():
            if term == 1:
                expectation_value += coeff
                continue
            # Find a list of qubit positions for each term
            qubits = [int(q) for q in re.findall(r"[XYZ](\d+)", str(term))] 
            z_vector = np.prod([self._z_vectors[q] for q in qubits], axis = 0)
            expectation_value += coeff * np.dot(z_vector, probability_vector)
        return expectation_value
    
    @staticmethod
    def _format_output(full_observables_data):
        """
        Helper function to format the output here because it is currently a nested dictionary of
        sample keys inside a dictionary of observable keys. For protocols that do not call samples
        or have only 1 observable, then we do not return nested dictionaries. This allows protocol
        to be more robust and output exactly what is necessary. If nested dictionary is output, this
        function also swaps it, such that sample keys are on the outer nest which makes more sense imo
        """
        if len(full_observables_data) == 1:
            _, sample_values = list(full_observables_data.items())[0]
            if len(sample_values) == 1:
                _, value = list(sample_values.items())[0]
                return value
            return dict(sample_values)
        output = defaultdict(dict)
        for observable_key, sample_values in full_observables_data.items():
            for sample_key, value in sample_values.items():
                output[sample_key][observable_key] = value
        if len(output) == 1:
            _, observable_values = list(output.items())[0]
            return dict(observable_values)
        return {
            sample_key: dict(observable_values)
            for sample_key, observable_values in output.items()
        }


# ------------------------------------------------------------
# ExactMeasurementProtocol
# ------------------------------------------------------------
# Measurement-based exact protocol.
# Uses exact probabilities directly; no multinomial sampling.

class ExactMeasurementProtocol(BaseMeasurementProtocol):
    def _evaluate_group_from_probabilities(self, exact_probabilities, group_expression) -> dict:
        # Input:
        #   exact_probabilities: exact probability vector for one group
        #   group_expression: commuting group terms
        # Output:
        #   {"exact": [group_expectation]}
        # Description:
        #   Converts exact probabilities directly into one exact group expectation.
        pass


# ------------------------------------------------------------
# ShotMeasurementProtocol
# ------------------------------------------------------------
# Fixed-shot protocol.
# Generates one or more independent sampled estimates for a fixed shot count.

class ShotMeasurementProtocol(BaseMeasurementProtocol):
    def __init__(self, shots, n_repeats=1, seed=None):
        # Input:
        #   shots: number of shots per estimate
        #   n_repeats: number of independent estimates
        #   seed: optional RNG seed
        # Output: protocol instance
        # Description:
        #   Stores fixed-shot sampling configuration.
        pass

    def _evaluate_group_from_probabilities(self, exact_probabilities, group_expression) -> dict:
        # Input:
        #   exact_probabilities: exact probability vector for one group
        #   group_expression: commuting group terms
        # Output:
        #   {shots: [group_expectation_1, group_expectation_2, ...]}
        # Description:
        #   Samples probabilities n_repeats times and reconstructs group expectations.
        pass


# ------------------------------------------------------------
# BootstrapMeasurementProtocol
# ------------------------------------------------------------
# Bootstrap-style protocol.
# Generates repeated sampled estimates for multiple shot/sample sizes.

class BootstrapMeasurementProtocol(BaseMeasurementProtocol):
    def __init__(self, sample_sizes, n_resamples, seed=None):
        # Input:
        #   sample_sizes: list[int], e.g. [1000, 10000, 50000]
        #   n_resamples: number of estimates per sample size
        #   seed: optional RNG seed
        # Output: protocol instance
        # Description:
        #   Stores bootstrap sampling configuration.
        pass

    def _evaluate_group_from_probabilities(self, exact_probabilities, group_expression) -> dict:
        # Input:
        #   exact_probabilities: exact probability vector for one group
        #   group_expression: commuting group terms
        # Output:
        #   {sample_size: [group_expectation_1, group_expectation_2, ...]}
        # Description:
        #   For each sample size, draws n_resamples multinomial samples and reconstructs group expectations.
        pass
