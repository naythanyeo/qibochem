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
from functools import reduce
from math import ceil, gcd
import numpy as np
import qibo
from qibo import Circuit
from scipy.stats import multinomial
from qibochem.measurement.optimization import measurement_basis_rotations, group_commuting_terms
from qibochem.measurement.result import constant_term, expectation_from_samples, v_expectation
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
    grouping: str = "qwc_fast"
    """
    Base measurement class for measurements with rotation
    This first does grouping and rotation, then sample with qibo backend from the
    exact probabilities vector -> sample probabilities vector -> expectation value
    This base protocol is for 1 shot, built for subclassing for other shot protocols
    For bootstrap / multi shot / adaptive / shot allocations, they can be subclassed 
    from this base protocol and added on on top of this
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
        final_state = circuit._final_state.state()
        self.n_qubits = circuit.nqubits
        # Generate z_vectors for measurement later
        self._generate_z_vectors()
        # Handles single input observables 
        if not isinstance(observables, Mapping):
            observables = {"O1": observables}
        # Loops through all the observables once first to store the commuting groups
        # This dictionary will be used to assign and allocate shots 
        self.observable_shot_groupings = defaultdict(dict)
        for observable_key, observable in observables.items():
            grouped_terms = measurement_basis_rotations(observable, grouping=self.grouping)
            for group_expression, measurement_gates in grouped_terms:
                self.observable_shot_groupings[observable_key][group_expression] = measurement_gates
        # Subsequent implementations of adaptive or allocated shots use this attribute 
        # Function defined in the subclass to assign shot allocations to each observable 
        self._get_shot_allocation()
        # After this loop through the data again
        full_observables_data = defaultdict(dict)
        for observable_key, observable_data in self.observable_shot_groupings.items():
            sample_observable_expectations = defaultdict(float)
            for group_expression, measurement_gates in observable_data.items():
                rotated_final_state = self._rotate_final_state(measurement_gates, final_state)
                exact_probabilities = np.abs(rotated_final_state) ** 2
                # From the exact_probabilities, do the sampling here
                # Sampled_probabilities is a dictionary with keys for different samples
                # Within sampled_probabilities, this will use observable_key and group_expressino as
                # keys in shot_allocation to determine the shots used to sample 
                sampled_probabilities = self._sample_probabilities(exact_probabilities,
                                                                   observable_key,
                                                                   group_expression)
                for sampling_key, probability_vector in sampled_probabilities.items():
                    expectation_value = self._probabilities2expectation(probability_vector, group_expression)
                    sample_observable_expectations[sampling_key] += expectation_value
            for sampling_key in sample_observable_expectations:
                sample_observable_expectations[sampling_key] += constant_term(observables[observable_key])
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

    def _get_shot_allocation(self):
        """
        By default, no shots will be allocated (for exact measurement can use this)
        Else the shot_allocation can be defined in each subclass 
        It will use self.observable_shot_groupings to determine shots allocated
        to each observable / group based on unique protocol requirements 
        Returns an attribute self.shot_allocation --> used by sample_probabilities
        Shot allocation will be a dictionary: shot_allocation[observable_key][group_expression]
        """
        self.shot_allocation = None

    def _sample_probabilities(self, exact_probabilities, observable_key, group_expression):
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

class ExactMeasurementProtocol(BaseMeasurementProtocol):
    """
    SHOULD in theory give the same result as StateVector protocol. But Exact measurements does it by
    rotating the final state and then measuring all in Z basis, while statevector directly measure it
    in the X or Y basis without rotation. 
    This should in theory be slightly slower than statevector because of that
    """
    def _sample_probabilities(self, exact_probabilities, observable_key, group_expression):
        return {"SV": exact_probabilities}


# ------------------------------------------------------------
# ShotMeasurementProtocol
# ------------------------------------------------------------

class ShotMeasurementProtocol(BaseMeasurementProtocol):
    """
    Base Shot Measurement Protocol that just samples shot probabilities 
    for one shot size. Most basic shot protocol.
    """
    def __init__(self, n_shots=None, t_shots=None, n_repeats=1, grouping="qwc_fast"):
        super().__init__(grouping=grouping)
        if n_shots is None and t_shots is None:
            raise ValueError("Specify either n_shots or t_shots.")
        if isinstance(n_shots, int) and n_shots <= 0:
            raise ValueError("n_shots must be a positive integer.")
        if isinstance(n_shots, dict):
            if not all(isinstance(value, int) and value > 0 for value in n_shots.values()):
                raise ValueError("n_shots dictionary values must be positive integers.")
        elif n_shots is not None and not isinstance(n_shots, int):
            raise TypeError("n_shots must be an integer, dictionary, or None.")
        if t_shots is not None and (not isinstance(t_shots, int) or t_shots <= 0):
            raise ValueError("t_shots must be a positive integer.")
        if not isinstance(n_repeats, int) or n_repeats <= 0:
            raise ValueError("n_repeats must be a positive integer.")
        self.n_shots = n_shots
        self.t_shots = t_shots
        self.n_repeats = n_repeats
        self.observable_shots = None
        self.backend = qibo.get_backend()

    def _get_shot_allocation(self):
        """
        n_shots means shots per observable. t_shots means total shots across observables.
        Within each observable, shots are distributed uniformly across its commuting groups.
        """
        observable_keys = list(self.observable_shot_groupings)
        # If n_shots is manually specified as dictionary, use it directly
        if isinstance(self.n_shots, dict):
            if set(self.n_shots) != set(observable_keys):
                raise ValueError("n_shots dictionary keys must match observable keys.")
            self.observable_shots = dict(self.n_shots)
            # Check that the total shots match (if both are specified)
            if self.t_shots is not None and sum(self.observable_shots.values()) != self.t_shots:
                raise ValueError("n_shots values must sum to t_shots.")
            self.t_shots = sum(self.observable_shots.values())
        # If n_shots is specified as integer, the allocate it uniformly across observables
        elif isinstance(self.n_shots, int):
            self.observable_shots = {observable_key: self.n_shots for observable_key in observable_keys}
            expected_total = self.n_shots * len(observable_keys)
            # If t_shots is also specified, check that they match
            if self.t_shots is not None and self.t_shots != expected_total:
                raise ValueError("t_shots must match n_shots times number of observables.")
            self.t_shots = expected_total
        # It t_shots instead is specified, then allocate as uniformly as possible
        else:
            base_shots = self.t_shots // len(observable_keys)
            remainder = self.t_shots % len(observable_keys)
            if base_shots == 0:
                raise ValueError("t_shots must be at least the number of observables.")
            self.observable_shots = {
                observable_key: base_shots + int(index < remainder)
                for index, observable_key in enumerate(observable_keys)
            }
        # Now allocate shots across commuting groups within each observable
        # TBC can implement other shot allocation here with functions
        self.shot_allocation = defaultdict(dict)
        for observable_key, groups in self.observable_shot_groupings.items():
            group_expressions = list(groups)
            observable_budget = self.observable_shots[observable_key]
            base_shots = observable_budget // len(group_expressions)
            remainder = observable_budget % len(group_expressions)
            if base_shots == 0:
                raise ValueError("Each observable needs at least one shot per commuting group.")
            for index, group_expression in enumerate(group_expressions):
                self.shot_allocation[observable_key][group_expression] = base_shots + int(index < remainder)

    def _sample_probabilities(self, exact_probabilities, observable_key, group_expression) -> dict:
        """
        Samples the exact_probabilities with qibo backend
        n_repeats is the number of samples wanted 
        """
        n_shots = self.shot_allocation[observable_key][group_expression]
        sampled_probabilities = {}

        for repeat in range(self.n_repeats):
            frequencies = self.backend.sample_frequencies(exact_probabilities, n_shots)

            probability_vector = np.zeros_like(exact_probabilities)
            for basis_index, count in frequencies.items():
                probability_vector[basis_index] = count / n_shots

            sampled_probabilities[f"shots_{repeat}"] = probability_vector

        return sampled_probabilities


# ------------------------------------------------------------
# MultiShotProtocol
# ------------------------------------------------------------

class MultiShotProtocol(BaseMeasurementProtocol):
    """
    Multi-shot protocol that samples probailities for multiple shot sizes at once 
    This will save the grouping and rotation overhead cost. Might be better compared
    to bootstrap protocol depending on sample size. 
    
    """
    def __init__(
        self,
        sample_sizes=(1000, 10000, 20000, 50000, 100000),
        n_repeats=10,
        grouping="qwc_fast",
    ):
        super().__init__(grouping=grouping)
        if not sample_sizes:
            raise ValueError("sample_sizes cannot be empty.")
        if not all(isinstance(sample_size, int) and sample_size > 0 for sample_size in sample_sizes):
            raise ValueError("sample_sizes must contain positive integers.")
        if not isinstance(n_repeats, int) or n_repeats <= 0:
            raise ValueError("n_repeats must be a positive integer.")

        self.sample_sizes = tuple(sample_sizes)
        self.n_repeats = n_repeats
        self.backend = qibo.get_backend()
        self.shot_allocation = None

    def _get_shot_allocation(self):
        """
        sample_sizes are shots per observable.
        Within each observable, every sample size is split uniformly across groups.
        """
        self.shot_allocation = defaultdict(dict)
        for observable_key, groups in self.observable_shot_groupings.items():
            group_expressions = list(groups)
            for sample_size in self.sample_sizes:
                base_shots = sample_size // len(group_expressions)
                remainder = sample_size % len(group_expressions)
                if base_shots == 0:
                    raise ValueError("Each sample size needs at least one shot per commuting group.")
                for index, group_expression in enumerate(group_expressions):
                    self.shot_allocation[observable_key].setdefault(group_expression, {})
                    self.shot_allocation[observable_key][group_expression][sample_size] = (
                        base_shots + int(index < remainder)
                    )
    def _sample_probabilities(self, exact_probabilities, observable_key, group_expression) -> dict:
        sampled_probabilities = {}
        for sample_size in self.sample_sizes:
            group_shots = self.shot_allocation[observable_key][group_expression][sample_size]
            for repeat in range(self.n_repeats):
                frequencies = self.backend.sample_frequencies(exact_probabilities, group_shots)
                probability_vector = np.zeros_like(exact_probabilities)
                for basis_index, count in frequencies.items():
                    probability_vector[basis_index] = count / group_shots
                sampled_probabilities[f"shots_{sample_size}_{repeat}"] = probability_vector
        return sampled_probabilities


# ------------------------------------------------------------
# BootstrapMeasurementProtocol
# ------------------------------------------------------------

class BootstrapMeasurementProtocol(BaseMeasurementProtocol):
    """
    Boostrap protocol for sampling probabilities. This is useful if many many samples are
    required, and minimum sample size is large (so total number of bootstrap samples is small)
    It first samples the highest common factor shot value multiple times, then bootstrap
    from that set of samples. Eg to get 10k shots, it will pick 10 samples of 1k shots and 
    average them. However, this protocol might be slower than multi-shot protocol because
    there is significant overhead when calling backend.sample_frequencies. And sampling 100k
    shots for eg won't necesarrily be 100 times slower than sampling 1k shots because this is
    reconstructed by probabilities rather than actual samples. 
    So if the sample pattern is something like: 10k, 50k, 100k, 200k, 300k, 500k, 750k, 1M 
    Then maximum only got 100 samples (1M / 10k) so it could be beneficial 
    But if the shot pattern is closer to something like: 100, 1k, 10k, 100k, 1M
    Then we end up with 10k samples which is probably slower than just using multi-shot
    """
    def __init__(
        self,
        sample_sizes=(1000, 10000, 20000, 50000, 100000),
        n_resamples=10,
        pool_shots=None,
        grouping="qwc_fast",
    ):
        super().__init__(grouping=grouping)
        if not sample_sizes:
            raise ValueError("sample_sizes cannot be empty.")
        if not all(isinstance(sample_size, int) and sample_size > 0 for sample_size in sample_sizes):
            raise ValueError("sample_sizes must contain positive integers.")
        if not isinstance(n_resamples, int) or n_resamples <= 0:
            raise ValueError("n_resamples must be a positive integer.")
        if pool_shots is not None and (not isinstance(pool_shots, int) or pool_shots <= 0):
            raise ValueError("pool_shots must be a positive integer.")

        self.sample_sizes = tuple(sample_sizes)
        self.n_resamples = n_resamples
        self.sample_shots = reduce(gcd, self.sample_sizes)
        if self.sample_shots <= 0:
            raise ValueError("sample_shots must be positive.")
        if not all(sample_size % self.sample_shots == 0 for sample_size in self.sample_sizes):
            raise ValueError("All sample sizes must be divisible by sample_shots.")

        initial_pool_shots = max(self.sample_sizes) * n_resamples if pool_shots is None else pool_shots
        self.total_samples = ceil(initial_pool_shots / self.sample_shots)
        self.pool_shots = self.total_samples * self.sample_shots
        self.shot_allocation = None
        self.backend = qibo.get_backend()

    def _get_shot_allocation(self):
        """
        sample_shots is the number of shots in one pool vector per observable.
        Within each observable, sample_shots is distributed uniformly across groups.
        """
        self.shot_allocation = defaultdict(dict)

        for observable_key, groups in self.observable_shot_groupings.items():
            group_expressions = list(groups)
            base_shots = self.sample_shots // len(group_expressions)
            remainder = self.sample_shots % len(group_expressions)
            if base_shots == 0:
                raise ValueError("sample_shots must provide at least one shot per commuting group.")
            for index, group_expression in enumerate(group_expressions):
                self.shot_allocation[observable_key][group_expression] = base_shots + int(index < remainder)

    def _sample_probabilities(self, exact_probabilities, observable_key, group_expression) -> dict:
        group_block_shots = self.shot_allocation[observable_key][group_expression]
        pool_vectors = np.zeros((self.total_samples, len(exact_probabilities)))

        for pool_index in range(self.total_samples):
            frequencies = self.backend.sample_frequencies(exact_probabilities, group_block_shots)
            for basis_index, count in frequencies.items():
                pool_vectors[pool_index, basis_index] = count / group_block_shots

        sampled_probabilities = {}
        pool_probabilities = np.full(self.total_samples, 1 / self.total_samples)

        for sample_size in self.sample_sizes:
            n_blocks = sample_size // self.sample_shots
            for resample in range(self.n_resamples):
                weights = multinomial.rvs(n_blocks, pool_probabilities)
                sampled_probabilities[f"shots_{sample_size}_{resample}"] = weights @ pool_vectors / n_blocks

        return sampled_probabilities
