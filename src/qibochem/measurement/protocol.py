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
import sympy as sp
from qibo import Circuit
from scipy.stats import multinomial
from qibochem.measurement.optimization import measurement_basis_rotations
from qibochem.measurement.result import constant_term

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

        # For single observables just return
        if not isinstance(observables, Mapping):
            return observables.expectation(circuit)
        # If you want to evaluate each observable separately 
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
    shot_distribution: str = "uniform"
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
        2) collect every observable term and group all terms globally
        3) reconstruct each observable from the globally sampled groups
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

        if circuit is None:
            raise ValueError("Specify a circuit to evaluate.")
        if observables is None:
            raise ValueError("Specify observables to evaluate.")

        if circuit._final_state is None: # Re-run circuit only if it has not been run yet
            circuit()
        final_state = circuit._final_state.state()
        self.n_qubits = circuit.nqubits
        # Generate z_vectors for measurement later
        self._generate_z_vectors()
        # Handles single input observables 
        if not isinstance(observables, Mapping):
            observables = {"O1": observables}
        # Global terms is a list of unique term values (strings)
        global_terms, observable_terms = self._collect_global_terms(observables)
        # Groups all the commuting terms, in this case self.grouping by default is qwc_fast
        self.grouped_terms = measurement_basis_rotations(global_terms, grouping=self.grouping)
        # grouped_terms is: (group_expression, measurement gates)... 
        # Each group_expression is all the possible observables that exist
        self.observable_group_expressions = self._build_observable_group_expressions(
            observable_terms,
            self.grouped_terms,
        )
        self._get_shot_allocation()
        sampled_group_probabilities = self._sample_all_groups(final_state)
        full_observables_data = self._reconstruct_observables(observables, sampled_group_probabilities)
        return self._format_output(full_observables_data)

    def _collect_global_terms(self, observables):
        """
        Collect all nonconstant terms for global grouping, and retain the
        observable-specific coefficients for later reconstruction.
        """
        observable_terms = defaultdict(dict)
        unique_terms = {}

        for observable_key, observable in observables.items():
            # observables: (X0*Y1 .. , 0.5)
            for term, coeff in observable.form.as_coefficients_dict().items():
                if term == 1:
                    continue
                term_key = str(term)
                unique_terms.setdefault(term_key, (term, 1.0)) # Only add the new unique terms
                observable_terms[observable_key][term_key] = (term, coeff)

        return list(unique_terms.values()), observable_terms

    @staticmethod
    def _build_observable_group_expressions(observable_terms, grouped_terms):
        """
        For every global commuting group, rebuild only the terms that belong to
        each observable, with that observable's original coefficients.
        """
        observable_group_expressions = defaultdict(dict)

        for group_expression, _ in grouped_terms:
            group_terms = {
                str(term): term
                for term in group_expression.as_coefficients_dict()
                if term != 1
            }
            for observable_key, terms in observable_terms.items():
                observable_specific_terms = [
                    terms[term_key][1] * term
                    for term_key, term in group_terms.items()
                    if term_key in terms
                ]
                if observable_specific_terms:
                    observable_group_expressions[observable_key][group_expression] = (
                        sp.Add(*observable_specific_terms)
                    )

        return observable_group_expressions

    def _sample_all_groups(self, final_state):
        """
        Rotate and sample every global commuting group once.
        """
        sampled_group_probabilities = {}
        for group_expression, measurement_gates in self.grouped_terms:
            rotated_final_state = self._rotate_final_state(measurement_gates, final_state)
            exact_probabilities = np.abs(rotated_final_state) ** 2
            sampled_group_probabilities[group_expression] = self._sample_probabilities(
                exact_probabilities,
                group_expression,
            )
        return sampled_group_probabilities

    def _reconstruct_observables(self, observables, sampled_group_probabilities):
        """
        Reconstruct every observable from the shared sampled group probabilities.
        """
        sample_keys = {
            sample_key
            for sampled_probabilities in sampled_group_probabilities.values()
            for sample_key in sampled_probabilities
        }
        if not sample_keys:
            sample_keys = {"constant"}

        full_observables_data = defaultdict(dict)
        for observable_key, observable in observables.items():
            sample_observable_expectations = defaultdict(float)
            for global_group_expression, observable_expression in self.observable_group_expressions[observable_key].items():
                sampled_probabilities = sampled_group_probabilities[global_group_expression]
                for sampling_key, probability_vector in sampled_probabilities.items():
                    expectation_value = self._probabilities2expectation(
                        probability_vector,
                        observable_expression,
                    )
                    sample_observable_expectations[sampling_key] += expectation_value
            for sampling_key in sample_keys:
                sample_observable_expectations[sampling_key] += constant_term(observable)
            full_observables_data[observable_key] = sample_observable_expectations

        return full_observables_data

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
        It will use self.grouped_terms to determine shots allocated to each group
        based on unique protocol requirements
        Returns an attribute self.shot_allocation --> used by sample_probabilities
        """
        self.shot_allocation = None

    def _sample_probabilities(self, exact_probabilities, group_expression):
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
    def _sample_probabilities(self, exact_probabilities, group_expression):
        return {"SV": exact_probabilities}


# ------------------------------------------------------------
# ShotMeasurementProtocol
# ------------------------------------------------------------

class ShotMeasurementProtocol(BaseMeasurementProtocol):
    """
    Base Shot Measurement Protocol that just samples shot probabilities 
    for one shot size. Most basic shot protocol.
    """
    def __init__(
        self,
        n_shots=None,
        t_shots=None,
        n_repeats=1,
        grouping="qwc_fast",
        shot_distribution="uniform",
    ):
        super().__init__(grouping=grouping, shot_distribution=shot_distribution)
        if n_shots is None and t_shots is None:
            raise ValueError("Specify either n_shots or t_shots.")
        if isinstance(n_shots, int) and n_shots <= 0:
            raise ValueError("n_shots must be a positive integer.")
        if isinstance(n_shots, dict):
            raise TypeError("n_shots dictionaries are not supported. Use n_shots or t_shots across commuting groups.")
        elif n_shots is not None and not isinstance(n_shots, int):
            raise TypeError("n_shots must be an integer or None.")
        if t_shots is not None and (not isinstance(t_shots, int) or t_shots <= 0):
            raise ValueError("t_shots must be a positive integer.")
        if not isinstance(n_repeats, int) or n_repeats <= 0:
            raise ValueError("n_repeats must be a positive integer.")
        if shot_distribution != "uniform":
            raise NotImplementedError("Only uniform shot distribution is supported for now.")
        self.n_shots = n_shots
        self.t_shots = t_shots
        self.n_repeats = n_repeats
        self.backend = qibo.get_backend()

    def _get_shot_allocation(self):
        """
        n_shots means shots per commuting group.
        t_shots means total shots distributed uniformly across commuting groups.
        """
        group_expressions = [group_expression for group_expression, _ in self.grouped_terms]
        self.shot_allocation = {}
        if not group_expressions:
            return

        if isinstance(self.n_shots, int):
            expected_total = self.n_shots * len(group_expressions)
            if self.t_shots is not None and self.t_shots != expected_total:
                raise ValueError("t_shots must match n_shots times number of commuting groups.")
            for group_expression in group_expressions:
                self.shot_allocation[group_expression] = self.n_shots
            return

        base_shots = self.t_shots // len(group_expressions)
        remainder = self.t_shots % len(group_expressions)
        if base_shots == 0:
            raise ValueError("t_shots must be at least the number of commuting groups.")
        for index, group_expression in enumerate(group_expressions):
            self.shot_allocation[group_expression] = base_shots + int(index < remainder)

    def _sample_probabilities(self, exact_probabilities, group_expression) -> dict:
        """
        Samples the exact_probabilities with qibo backend
        n_repeats is the number of samples wanted 
        """
        n_shots = self.shot_allocation[group_expression]
        sampled_probabilities = {}

        for repeat in range(self.n_repeats):
            frequencies = self.backend.sample_frequencies(exact_probabilities, n_shots)

            probability_vector = np.zeros_like(exact_probabilities)
            for basis_index, count in frequencies.items():
                probability_vector[int(basis_index)] = count / n_shots

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
        shot_distribution="uniform",
    ):
        super().__init__(grouping=grouping, shot_distribution=shot_distribution)
        if not sample_sizes:
            raise ValueError("sample_sizes cannot be empty.")
        if not all(isinstance(sample_size, int) and sample_size > 0 for sample_size in sample_sizes):
            raise ValueError("sample_sizes must contain positive integers.")
        if not isinstance(n_repeats, int) or n_repeats <= 0:
            raise ValueError("n_repeats must be a positive integer.")
        if shot_distribution != "uniform":
            raise NotImplementedError("Only uniform shot distribution is supported for now.")

        self.sample_sizes = tuple(sample_sizes)
        self.n_repeats = n_repeats
        self.backend = qibo.get_backend()
        self.shot_allocation = None

    def _get_shot_allocation(self):
        """
        sample_sizes are shots per commuting group.
        """
        self.shot_allocation = defaultdict(dict)
        for group_expression, _ in self.grouped_terms:
            for sample_size in self.sample_sizes:
                self.shot_allocation[group_expression][sample_size] = sample_size

    def _sample_probabilities(self, exact_probabilities, group_expression) -> dict:
        sampled_probabilities = {}
        for sample_size in self.sample_sizes:
            group_shots = self.shot_allocation[group_expression][sample_size]
            for repeat in range(self.n_repeats):
                frequencies = self.backend.sample_frequencies(exact_probabilities, group_shots)
                probability_vector = np.zeros_like(exact_probabilities)
                for basis_index, count in frequencies.items():
                    probability_vector[int(basis_index)] = count / group_shots
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
        shot_distribution="uniform",
    ):
        super().__init__(grouping=grouping, shot_distribution=shot_distribution)
        if not sample_sizes:
            raise ValueError("sample_sizes cannot be empty.")
        if not all(isinstance(sample_size, int) and sample_size > 0 for sample_size in sample_sizes):
            raise ValueError("sample_sizes must contain positive integers.")
        if not isinstance(n_resamples, int) or n_resamples <= 0:
            raise ValueError("n_resamples must be a positive integer.")
        if pool_shots is not None and (not isinstance(pool_shots, int) or pool_shots <= 0):
            raise ValueError("pool_shots must be a positive integer.")
        if shot_distribution != "uniform":
            raise NotImplementedError("Only uniform shot distribution is supported for now.")

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
        sample_shots is the number of shots in one pool vector per commuting group.
        """
        self.shot_allocation = {}
        for group_expression, _ in self.grouped_terms:
            self.shot_allocation[group_expression] = self.sample_shots

    def _sample_probabilities(self, exact_probabilities, group_expression) -> dict:
        group_block_shots = self.shot_allocation[group_expression]
        pool_vectors = np.zeros((self.total_samples, len(exact_probabilities)))

        for pool_index in range(self.total_samples):
            frequencies = self.backend.sample_frequencies(exact_probabilities, group_block_shots)
            for basis_index, count in frequencies.items():
                pool_vectors[pool_index, int(basis_index)] = count / group_block_shots

        sampled_probabilities = {}
        pool_probabilities = np.full(self.total_samples, 1 / self.total_samples)

        for sample_size in self.sample_sizes:
            n_blocks = sample_size // self.sample_shots
            for resample in range(self.n_resamples):
                weights = multinomial.rvs(n_blocks, pool_probabilities)
                sampled_probabilities[f"shots_{sample_size}_{resample}"] = weights @ pool_vectors / n_blocks

        return sampled_probabilities
