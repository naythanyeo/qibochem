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


from collections.abc import Mapping
from dataclasses import dataclass

from qibochem.measurement.optimization import measurement_basis_rotations
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
# Shared base class for exact measurement, shots, and bootstrap.
# Uses circuit.final_state as persistent state cache.
# Does NOT persist exact probability vectors across protocols by default.

class BaseMeasurementProtocol:
    def evaluate(self, circuit, observables) -> dict:
        # Input:
        #   circuit: Qibo circuit
        #   observables: single observable, list, or dict[name, observable]
        # Output:
        #   dict[sample_label, list[dict[name, expectation_value]]]
        # Description:
        #   Main wrapper. Gets final_state, loops over observables, evaluates each, then transposes output format.
        pass

    def _get_final_state(self, circuit) -> np.ndarray:
        # Input: Qibo circuit
        # Output: final statevector
        # Description: Reuses circuit.final_state if available; otherwise executes circuit once.
        pass

    def _normalise_observables(self, observables) -> dict:
        # Input: observable / list[observable] / dict[name, observable]
        # Output: dict[name, observable]
        # Description: Converts observable input into consistent labelled dictionary.
        pass

    def _evaluate_one_observable(self, final_state, circuit, observable) -> dict:
        # Input:
        #   final_state: cached statevector
        #   circuit: Qibo circuit
        #   observable: one Hamiltonian/operator
        # Output:
        #   dict[sample_label, list[expectation_value]]
        # Description:
        #   Groups observable, rotates final_state per group, computes exact probabilities, and accumulates sampled/exact estimates.
        pass

    def _group_observable(self, observable) -> list:
        # Input: one observable
        # Output: list[(group_expression, measurement_gates)]
        # Description:
        #   Uses QiboChem grouping logic to split observable into commuting measurement groups.
        pass

    def _rotate_state(self, final_state, circuit, measurement_gates) -> np.ndarray:
        # Input:
        #   final_state: cached statevector
        #   circuit: Qibo circuit metadata, mainly nqubits/backend
        #   measurement_gates: basis-rotation / measurement info for one group
        # Output: rotated statevector
        # Description:
        #   Applies only the measurement-basis rotations to a copy of final_state.
        pass

    def _state_to_probabilities(self, rotated_state) -> np.ndarray:
        # Input: rotated statevector
        # Output: exact Born probability vector
        # Description:
        #   Converts amplitudes to probabilities via |amplitude|^2.
        pass

    def _sample_probabilities(self, exact_probabilities, shots) -> np.ndarray:
        # Input:
        #   exact_probabilities: exact probability vector
        #   shots: number of samples
        # Output: sampled probability vector
        # Description:
        #   Draws multinomial samples and returns normalised sampled probabilities.
        pass

    def _probabilities_to_group_expectation(self, probabilities, group_expression) -> complex:
        # Input:
        #   probabilities: exact or sampled probability vector
        #   group_expression: commuting group terms
        # Output: group expectation contribution
        # Description:
        #   Reconstructs expectation contribution of all terms in one commuting group.
        pass

    def _evaluate_group_from_probabilities(self, exact_probabilities, group_expression) -> dict:
        # Input:
        #   exact_probabilities: exact probability vector for one group
        #   group_expression: commuting group terms
        # Output:
        #   dict[sample_label, list[group_expectation]]
        # Description:
        #   Abstract protocol-specific step. Subclasses define how exact probabilities become expectation estimates.
        raise NotImplementedError

    def _initialise_observable_values(self) -> dict:
        # Input: none / protocol config
        # Output: dict[sample_label, list[initial_value]]
        # Description:
        #   Creates zero/constant-filled accumulators matching the protocol output shape.
        pass

    def _add_group_values(self, observable_values, group_values) -> dict:
        # Input:
        #   observable_values: current accumulated values
        #   group_values: contribution from one commuting group
        # Output: updated observable_values
        # Description:
        #   Adds group expectation contributions into observable-level estimates.
        pass

    def _transpose_results(self, by_observable) -> dict:
        # Input:
        #   dict[name, dict[sample_label, list[value]]]
        # Output:
        #   dict[sample_label, list[dict[name, value]]]
        # Description:
        #   Reorders output so each sample contains all observable values.
        pass


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

