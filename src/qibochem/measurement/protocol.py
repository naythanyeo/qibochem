"""
Measurement protocols for bitmask observables.

All protocol classes in this module evaluate ``BitmaskObservable`` objects.
The sampled protocols group all observable terms globally, rotate the final
state once per commuting group, sample probability vectors, then reconstruct
each observable from the sampled group data.

The input of the "circuit" can also refer to final states
the get_final_state function checks if its a final_state or circuit object so 
both of it are acceptable as inputs. 
"""

from collections import defaultdict
from dataclasses import dataclass
from functools import reduce
from math import ceil, gcd

import numpy as np
from scipy.stats import multinomial
from scipy.sparse import csr_matrix

from qibochem.measurement.expectation import bitmask_expectation
from qibochem.measurement.grouping import group_bitmask_terms
from qibochem.measurement.rotation import rotate_basis
from qibochem.measurement.utils import (
    collect_global_unique_terms,
    format_output,
    get_final_state,
    normalise_observables
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
    This base protocol mainly handles the grouping of commuting terms and sampling 
    The sampling method must be defined at a subclass level 
    
    The only key function sample_probabilities defined at the subclass level reads in the exact
    probabilities vector, and outputs a dictionary of {sample_key: sampled_probability}
    """

    largest_first: bool = True

    def evaluate(self, circuit, observables):
        if observables is None:
            raise ValueError("Specify observables to evaluate.")

        final_state = get_final_state(circuit)
        observables = normalise_observables(observables)
        # Get unique terms from all global observables 
        global_unique_terms = collect_global_unique_terms(observables)
        self.groups = group_bitmask_terms(global_unique_terms, largest_first=self.largest_first)
        # self.groups is {group_mask: [term1, term2...]}
        # Parity vectors is for caching each evaluated term in Z basis
        self._parity_vectors = {}
        self._probability_vector_size = final_state.size


        """
        Build a sparse coefficient matrix here to speed up the calculations later 
        C[observables, terms]
        Later on build a dense term matrix t[terms, samples] 
        Can do 1 matrix multiplication for C@t with numpy for more efficient calculations

        Sparse matrix will have row index is observable_index, column index as term_index
        term_index is a global mapping for every term

        constant_terms has shape len(observables), 1
        """
        rows, cols, data = [], [], []
        constant_terms = []
        term_index = {term: index for index, term in enumerate(global_unique_terms)}

        observable_keys = list(observables.keys())
        for observable_index, observable_key in enumerate(observable_keys):
            for term, coeff in observables[observable_key].terms.items():
                if term == (0, 0, 0):
                    continue
                rows.append(observable_index)
                cols.append(term_index[term])
                data.append(coeff)
            # Add constant terms to list
            constant_terms.append(observables[observable_key].constant)
        constant_terms = np.asarray(constant_terms)[:, None] # Make it shape (n_obs, 1)
        coeff_sparse_matrix = csr_matrix((data, (rows, cols)),
                                         shape = (len(observables), len(global_unique_terms)))
            
        # Build dense matrix term expectations x samples
        term_expectations = np.zeros((len(global_unique_terms), self.n_samples))
        for group_mask, terms in self.groups.items():
            rotated_state = rotate_basis(final_state, group_mask)
            exact_probabilities = np.abs(rotated_state) ** 2
            exact_probabilities = exact_probabilities / np.sum(exact_probabilities)

            # Sample_probabilities is a subclass defined function that takes samples of exact probabilities 
            # sample_probabilities is of the form {sample_key: sampled_probability}
            sampled_probabilities = self._sample_probabilities(exact_probabilities)
            sample_keys = list(sampled_probabilities.keys())
            # Flatten the sample probability vectors into a matrix
            sample_matrix = np.array([sampled_probabilities[key] for key in sample_keys])

            """
            These are group terms, we need to map them to the overall term_expectation index
            with the same term_index dictionary we used in constructing the sparse matrix
            So in this case each group we do one matrix multiplication for all the group terms
            We keep the term_rows so we know which rows to append it to in term_expectations

            Define n_basis as the number of determinants full expanded (eg 0000, 0001, 0010 ...)
            n_group_terms is the number of terms within a particular group
            Sample matrix has shape (n_samples, n_basis)
            z_parity_matrix has shape (n_group_terms, n_basis)
            """
            z_parity_matrix, term_rows = [], []
            for term in terms:
                x_mask, y_mask, z_mask = term
                measured_mask = (x_mask | y_mask | z_mask)
                z_parity_matrix.append(self._z_parity_vector(measured_mask))
                term_rows.append(term_index[term])

            z_parity_matrix = np.array(z_parity_matrix)
            term_rows = np.array(term_rows)

            # Append the group terms to the matrix to the correct rows
            term_expectations[term_rows, :] = z_parity_matrix @ sample_matrix.T

        # Reconstruct final output
        sample_observable_expectation = defaultdict(dict)
        
        sampled_expectations_matrix = constant_terms + coeff_sparse_matrix @ term_expectations
        for sample_index, sample_key in enumerate(sample_keys):
            for observable_index, observable_key in enumerate(observable_keys):
                sample_observable_expectation[sample_key][observable_key] = (
                    sampled_expectations_matrix[(observable_index, sample_index)]
                )
        return sample_observable_expectation

    def _probabilities2term_expectation(self, probability_vector, term):
        """Convert one sampled probability vector into one Pauli expectation."""
        x_mask, y_mask, z_mask = term
        measured_mask = x_mask | y_mask | z_mask
        return np.dot(self._z_parity_vector(measured_mask), probability_vector)

    def _z_parity_vector(self, measured_mask):
        """
        Return cached Z-parity signs for one measured bitmask.

        For a measured Pauli term, only the qubits included in the measured mask matter.
        Each computational basis state, e.g. 0000, 0001, 0010, ..., is AND-ed with the
        measured mask to select the occupied measured qubits.

        The parity of the number of selected occupied qubits determines the sign:
            even count -> +1
            odd count  -> -1

        The resulting parity vector stores this +1/-1 sign for every computational basis
        state. After X/Y basis rotations have been applied, the original Pauli type no
        longer matters here; only whether the term acts on a given qubit matters.
        """
        if measured_mask not in self._parity_vectors:
            basis_states = np.arange(self._probability_vector_size)
            occupied_counts = np.bitwise_count(basis_states & measured_mask).astype(int)
            self._parity_vectors[measured_mask] = (-1) ** occupied_counts

        return self._parity_vectors[measured_mask]

    def _sample_probabilities(self, exact_probabilities):
        """
        DEFINED BY SUBCLASS
        This function will take in the exact probabilities and output the sampled ones 
        Output is in the form of {sample_key: [sample vector], ...}
        Each subclass will be responsible for deciding the shots that each sample gets 
        """
        raise NotImplementedError



@dataclass
class ExactMeasurementProtocol(BaseMeasurementProtocol):
    """Exact rotated-basis protocol using unsampled probabilities."""

    def __post_init__(self):
        self.n_samples = 1

    def _sample_probabilities(self, exact_probabilities):
        return {"SV": exact_probabilities}


class ShotMeasurementProtocol(BaseMeasurementProtocol):
    """
    Shot protocol for one shot count and optional repeats.
    Currently shot allocation NOT supported, shots specified refer to shots per commuting group
    """
    def __init__(
        self,
        n_shots=10_000,
        n_repeats=1,
        largest_first=True
    ):
        super().__init__(
            largest_first=largest_first,
        )

        if not isinstance(n_shots, int) or n_shots <= 0:
            raise ValueError("n_shots must be a positive integer.")
       
        if not isinstance(n_repeats, int) or n_repeats <= 0:
            raise ValueError("n_repeats must be a positive integer.")

        self.n_shots = n_shots
        self.n_repeats = n_repeats
        self.n_samples = n_repeats

    def _sample_probabilities(self, exact_probabilities):
        sampled_probabilities = {}

        for repeat in range(self.n_repeats):
            counts = np.random.multinomial(self.n_shots, exact_probabilities)
            probability_vector = counts / self.n_shots

            sampled_probabilities[f"shots_{repeat}"] = probability_vector

        return sampled_probabilities


class MultiShotProtocol(BaseMeasurementProtocol):
    """Protocol that samples several shot counts in one grouped evaluation."""

    def __init__(
        self,
        sample_sizes=(1000, 10000, 20000, 50000, 100000),
        n_repeats=10,
        largest_first=True,
    ):
        super().__init__(
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
        self.n_samples = n_repeats * len(sample_sizes)

    def _sample_probabilities(self, exact_probabilities):
        sampled_probabilities = {}

        for sample_size in self.sample_sizes:
            for repeat in range(self.n_repeats):
                counts = np.random.multinomial(sample_size, exact_probabilities)
                probability_vector = counts / sample_size

                sampled_probabilities[f"shots_{sample_size}_{repeat}"] = probability_vector

        return sampled_probabilities


class BootstrapMeasurementProtocol(BaseMeasurementProtocol):
    """
    Block-bootstrap protocol from a pool of fixed-shot probability vectors.
    This method can be slower than multishot especially if the cost of each evaluation is small
    Eg for smaller probability vectors (small active space) because the overhead cost is larger
    However, for a small fixed pool of shots and huge vectors, bootstrapping can be faster
    Generally though multishot is recommended (usually faster)
    """

    def __init__(
        self,
        sample_sizes=(1000, 10000, 20000, 50000, 100000),
        n_resamples=10,
        total_samples=None,
        largest_first=True
    ):
        super().__init__(
            largest_first=largest_first,
        )

        if not sample_sizes:
            raise ValueError("sample_sizes cannot be empty.")
        if not all(isinstance(sample_size, int) and sample_size > 0 for sample_size in sample_sizes):
            raise ValueError("sample_sizes must contain positive integers.")
        if not isinstance(n_resamples, int) or n_resamples <= 0:
            raise ValueError("n_resamples must be a positive integer.")
        if total_samples is not None and (not isinstance(total_samples, int) or total_samples <= 0):
            raise ValueError("total_samples must be a positive integer.")

        self.sample_sizes = tuple(sample_sizes)
        self.n_resamples = n_resamples
        self.sample_shots = reduce(gcd, self.sample_sizes)
        self.n_samples = n_resamples * len(sample_sizes)
        if self.sample_shots <= 0:
            raise ValueError("Sample Sizes cannot be reduced properly")
        if not all(sample_size % self.sample_shots == 0 for sample_size in self.sample_sizes):
            raise ValueError("All sample sizes must be divisible by sample_shots.")

        # Default to be same number as total max shots such that it can be drawn independently
        if total_samples is None:
            self.total_samples = ceil(
                self.n_resamples * max(self.sample_sizes) / self.sample_shots
            )
        else:
            self.total_samples = total_samples


    def _sample_probabilities(self, exact_probabilities):
        pool_vectors = np.zeros((self.total_samples, len(exact_probabilities)))

        for pool_index in range(self.total_samples):
            counts = np.random.multinomial(self.sample_shots, exact_probabilities)
            pool_vectors[pool_index] = counts / self.sample_shots

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


"""
TBC FOR FUTURE IMPLEMENTATION

1) RAW SHOT MEASUREMENTS
It can be useful to implement a raw measurement protocol for hardware simulation 
The current shot sampling protocols only use numpy multinomial to sample, rather than
using qibo's backend. This sampling method is mathematically equivalent to using the
backend. The backend first samples all of the individual frequencies across all the
samples, then collates them. This is useful if you are interested in the exact sample
counts, maybe for simulating hardware noise or filtering certain kinds of shots. 
However, it is also quite inefficient and scales badly for larger shot counts because
eg 1 million different samples must be created which is RAM and time consuming. 
Using multinomial is the default mode above, way faster because it just directly 
samples the probability rather than generate individual shot counts. For all intents
and purposes, doing so is much more efficient and equivalent to using the backend. 

For RawShotCount measurement protocol, probably use in tandem with hardware noise
simulators. Will need to import and define backend from qibo because its currently 
not imported yet. 

Sample code with backend
frequencies = self.backend.sample_frequencies(exact_probabilities, n_shots)
probability_vector = np.zeros_like(exact_probabilities)
for basis_index, count in frequencies.items():
    probability_vector[int(basis_index)] = count / n_shots

2) ADAPTIVE SHOT STRATEGIES
Currently adaptive per-group shots are not supported. If this is needed later, it should
be implemented as a separate protocol class instead of leaving stale hooks in the default
sampled protocols.
"""
