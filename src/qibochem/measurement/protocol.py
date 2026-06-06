"""
Measurement protocols for bitmask observables.

All protocol classes in this module evaluate ``BitmaskObservable`` objects.
The sampled protocols group all observable terms globally, rotate the final
state once per commuting group, sample probability vectors, then reconstruct
each observable from the sampled group data.
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from math import ceil, gcd

import numpy as np
import qibo
from scipy.stats import multinomial

from qibochem.measurement.expectation import bitmask_expectation
from qibochem.measurement.grouping import group_bitmask_terms
from qibochem.measurement.rotation import rotate_basis
from qibochem.measurement.utils import (
    collect_global_terms,
    format_output,
    get_final_state,
    normalise_observables,
    transpose_group_probabilities,
)


@dataclass
class StateVectorProtocol:
    """Direct statevector expectation protocol for bitmask observables."""

    def evaluate(self, circuit, observables):
        if observables is None:
            raise ValueError("Specify observables to evaluate.")

        final_state = get_final_state(circuit)
        observables = normalise_observables(observables)

        term_expectations = {}
        full_observables_data = defaultdict(dict)

        for observable_key, observable in observables.items():
            value = observable.constant
            for term, coeff in observable.terms.items():
                if term not in term_expectations:
                    term_expectations[term] = bitmask_expectation(final_state, term)

                value += coeff * term_expectations[term]

            full_observables_data["SV"][observable_key] = value

        return format_output(full_observables_data)


@dataclass
class BaseMeasurementProtocol:
    """
    Base class for sampled bitmask measurement protocols.

    The current bitmask implementation uses qwc_fast grouping.
    """

    shot_distribution: str = "uniform"
    largest_first: bool = True

    def __post_init__(self):
        if self.shot_distribution != "uniform":
            raise NotImplementedError("Only uniform shot distribution is supported.")

    def evaluate(self, circuit, observables):
        if observables is None:
            raise ValueError("Specify observables to evaluate.")

        final_state = get_final_state(circuit)
        observables = normalise_observables(observables)

        global_terms = collect_global_terms(observables)
        self.groups, self.term2group_map = group_bitmask_terms(
            global_terms,
            largest_first=self.largest_first,
        )
        # Parity vectors is for caching each evaluated term in Z basis
        self._parity_vectors = {}
        # Shot allocations will be a subclass defined function, allocate shot for coommuting groups
        self._get_shot_allocation() 
        group_probabilities = self._sample_all_groups(final_state)
        sample_probabilities = transpose_group_probabilities(group_probabilities)
        full_observables_data = self._reconstruct_observables(observables, sample_probabilities)

        return format_output(full_observables_data)

    def _sample_all_groups(self, final_state):
        """Rotate and sample each global commuting group once."""
        group_probabilities = {}

        for group_index, group_mask in enumerate(self.groups):
            rotated_state = rotate_basis(final_state, group_mask)
            exact_probabilities = np.abs(rotated_state) ** 2
            group_probabilities[group_index] = self._sample_probabilities(
                exact_probabilities,
                group_index,
            )

        return group_probabilities

    def _reconstruct_observables(self, observables, sample_probabilities):
        """
        Reconstruct observable expectations from sampled group probabilities.
        """
        sample_keys = tuple(sample_probabilities) # Select the keys for sample probabilities
        full_observables_data = defaultdict(dict)

        for sample_key in sample_keys:
            group_vectors = sample_probabilities[sample_key]
            term_expectations = {}

            for observable_key, observable in observables.items():
                value = observable.constant

                for term, coeff in observable.terms.items():
                    if term not in term_expectations:
                        group_index = self.term2group_map[term]
                        probability_vector = group_vectors[group_index]
                        term_expectations[term] = self._probabilities2term_expectation(
                                                            probability_vector, term)
                    value += coeff * term_expectations[term]

                full_observables_data[sample_key][observable_key] = value

        return full_observables_data

    def _probabilities2term_expectation(self, probability_vector, term):
        """Convert one sampled probability vector into one Pauli expectation."""
        x_mask, y_mask, z_mask = term
        measured_mask = x_mask | y_mask | z_mask
        return np.dot(self._z_parity_vector(measured_mask), probability_vector)

    def _z_parity_vector(self, measured_mask):
        """Return cached Z-parity signs for one measured bitmask."""
        if measured_mask not in self._parity_vectors:
            basis_states = np.arange(self._probability_vector_size)
            occupied_counts = np.bitwise_count(basis_states & measured_mask).astype(int)
            self._parity_vectors[measured_mask] = (-1) ** occupied_counts

        return self._parity_vectors[measured_mask]

    def _get_shot_allocation(self):
        """
        Default allocation is none
        Specify shot allocation in specific subclass method
        """
        self.shot_allocation = None

    def _sample_probabilities(self, exact_probabilities, group_index):
        """
        Defined by subclasses.
        """
        raise NotImplementedError



class ExactMeasurementProtocol(BaseMeasurementProtocol):
    """Exact rotated-basis protocol using unsampled probabilities."""

    def _sample_probabilities(self, exact_probabilities, group_index):
        self._probability_vector_size = exact_probabilities.size
        return {"SV": exact_probabilities}


class ShotMeasurementProtocol(BaseMeasurementProtocol):
    """Shot protocol for one shot count and optional repeats."""

    def __init__(
        self,
        n_shots=None,
        t_shots=None,
        n_repeats=1,
        shot_distribution="uniform",
        largest_first=True,
    ):
        super().__init__(
            shot_distribution=shot_distribution,
            largest_first=largest_first,
        )

        if n_shots is None and t_shots is None:
            raise ValueError("Specify either n_shots or t_shots.")
        if isinstance(n_shots, dict):
            raise TypeError("n_shots dictionaries are not supported.")
        if n_shots is not None and (not isinstance(n_shots, int) or n_shots <= 0):
            raise ValueError("n_shots must be a positive integer or None.")
        if t_shots is not None and (not isinstance(t_shots, int) or t_shots <= 0):
            raise ValueError("t_shots must be a positive integer or None.")
        if not isinstance(n_repeats, int) or n_repeats <= 0:
            raise ValueError("n_repeats must be a positive integer.")

        self.n_shots = n_shots
        self.t_shots = t_shots
        self.n_repeats = n_repeats
        self.backend = qibo.get_backend()

    def _get_shot_allocation(self):
        """Allocate shots across global commuting groups."""
        self.shot_allocation = {}
        if not self.groups:
            return

        if isinstance(self.n_shots, int):
            expected_total = self.n_shots * len(self.groups)
            if self.t_shots is not None and self.t_shots != expected_total:
                raise ValueError("t_shots must match n_shots times number of commuting groups.")

            for group_index in range(len(self.groups)):
                self.shot_allocation[group_index] = self.n_shots
            return

        base_shots = self.t_shots // len(self.groups)
        remainder = self.t_shots % len(self.groups)
        if base_shots == 0:
            raise ValueError("t_shots must be at least the number of commuting groups.")

        for group_index in range(len(self.groups)):
            self.shot_allocation[group_index] = base_shots + int(group_index < remainder)

    def _sample_probabilities(self, exact_probabilities, group_index):
        self._probability_vector_size = exact_probabilities.size
        n_shots = self.shot_allocation[group_index]
        sampled_probabilities = {}

        for repeat in range(self.n_repeats):
            frequencies = self.backend.sample_frequencies(exact_probabilities, n_shots)
            probability_vector = np.zeros_like(exact_probabilities)
            for basis_index, count in frequencies.items():
                probability_vector[int(basis_index)] = count / n_shots

            sampled_probabilities[f"shots_{repeat}"] = probability_vector

        return sampled_probabilities


class MultiShotProtocol(BaseMeasurementProtocol):
    """Protocol that samples several shot counts in one grouped evaluation."""

    def __init__(
        self,
        sample_sizes=(1000, 10000, 20000, 50000, 100000),
        n_repeats=10,
        shot_distribution="uniform",
        largest_first=True,
    ):
        super().__init__(
            shot_distribution=shot_distribution,
            largest_first=largest_first,
        )

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
        """Each sample size is shots per global commuting group."""
        self.shot_allocation = defaultdict(dict)
        for group_index in range(len(self.groups)):
            for sample_size in self.sample_sizes:
                self.shot_allocation[group_index][sample_size] = sample_size

    def _sample_probabilities(self, exact_probabilities, group_index):
        self._probability_vector_size = exact_probabilities.size
        sampled_probabilities = {}

        for sample_size in self.sample_sizes:
            group_shots = self.shot_allocation[group_index][sample_size]
            for repeat in range(self.n_repeats):
                frequencies = self.backend.sample_frequencies(exact_probabilities, group_shots)
                probability_vector = np.zeros_like(exact_probabilities)
                for basis_index, count in frequencies.items():
                    probability_vector[int(basis_index)] = count / group_shots

                sampled_probabilities[f"shots_{sample_size}_{repeat}"] = probability_vector

        return sampled_probabilities


class BootstrapMeasurementProtocol(BaseMeasurementProtocol):
    """Block-bootstrap protocol from a pool of fixed-shot probability vectors."""

    def __init__(
        self,
        sample_sizes=(1000, 10000, 20000, 50000, 100000),
        n_resamples=10,
        pool_shots=None,
        shot_distribution="uniform",
        largest_first=True,
    ):
        super().__init__(
            shot_distribution=shot_distribution,
            largest_first=largest_first,
        )

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
        """sample_shots is the shot count in one pool vector per group."""
        self.shot_allocation = {}
        for group_index in range(len(self.groups)):
            self.shot_allocation[group_index] = self.sample_shots

    def _sample_probabilities(self, exact_probabilities, group_index):
        self._probability_vector_size = exact_probabilities.size
        group_block_shots = self.shot_allocation[group_index]
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
                sampled_probabilities[f"shots_{sample_size}_{resample}"] = (
                    weights @ pool_vectors / n_blocks
                )

        return sampled_probabilities
