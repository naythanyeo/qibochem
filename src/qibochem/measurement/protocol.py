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
        A dictionary of expectation values. If ``observables`` is a dictionary,
        this has the same keys. If ``observables`` is a single Hamiltonian, the
        return value is ``{"observable": expectation_value}``.

Future considerations:
    Observables may become a separate class so protocols can act as compilers
    for different observable types, such as spin observables. Adaptive-shot
    protocols are still TBC.
"""


from collections.abc import Mapping
from dataclasses import dataclass, field

from qibochem.measurement.optimization import measurement_basis_rotations
from qibochem.measurement.result import expectation_from_samples
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
            observables = {"observable": observables}

        return {
            key: observable.expectation(circuit)
            for key, observable in observables.items()
        }

@dataclass
class ShotProtocol:
    """Protocol for evaluating observables on a shot-based simulator.
    Each observable is a qibo.hamiltonian object
    """

    total_shots: int | None = None
    shots_per_group: int | None = None
    grouping: str = "qwc"
    group_allocation: str = "uniform"
    observable_allocation: str = "uniform"
    batching: str = "local"
    trial_shots: int | None = None
    observable_shot_allocation: dict = field(default_factory=dict)
    group_shot_allocation: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate shot protocol configuration."""
        if self.batching not in ("local", "global"):
            raise ValueError("batching must be 'local' or 'global'.")
        if self.group_allocation not in ("uniform", "coefficients", "variance"):
            raise ValueError(
                "group_allocation must be 'uniform', 'coefficients', or 'variance'."
            )
        if self.observable_allocation not in ("uniform", "weights"):
            raise ValueError("observable_allocation must be 'uniform' or 'weights'.")
        if self.total_shots is not None and self.shots_per_group is not None:
            raise ValueError(
                "Specify only one of total_shots or shots_per_group."
            )

        self._validate_positive_int("total_shots", self.total_shots)
        self._validate_positive_int("shots_per_group", self.shots_per_group)
        self._validate_positive_int("trial_shots", self.trial_shots)

    @staticmethod
    def _validate_positive_int(name, value):
        """Check that optional integer fields are positive when provided."""
        if value is not None and (not isinstance(value, int) or value <= 0):
            raise ValueError(f"{name} must be a positive integer.")

    def _as_observable_dict(self, observables):
        """Return observables as a dictionary keyed by observable label."""
        if isinstance(observables, Mapping):
            return dict(observables)

        return {"observable": observables}

    @staticmethod
    def _distribute_integer_budget(total, weights):
        """Distribute an integer budget proportionally to normalized weights."""
        raw_allocations = {
            key: total * weight
            for key, weight in weights.items()
        }
        allocations = {
            key: int(allocation)
            for key, allocation in raw_allocations.items()
        }
        remaining = total - sum(allocations.values())
        ordered_keys = sorted(
            weights,
            key=lambda key: raw_allocations[key] - allocations[key],
            reverse=True,
        )

        for key in ordered_keys[:remaining]:
            allocations[key] += 1

        return allocations

    def _allocate_observable_shots(self, observables, weights=None):
        """Allocate total shots between observables."""
        if self.shots_per_group is not None:
            return None
        if self.total_shots is None:
            raise ValueError("Specify total_shots or shots_per_group.")
        if not observables:
            self.observable_shot_allocation = {}
            return {}

        if self.observable_allocation == "uniform":
            uniform_weight = 1 / len(observables)
            normalized_weights = {
                key: uniform_weight
                for key in observables
            }
        elif self.observable_allocation == "weights":
            if weights is None or not isinstance(weights, Mapping):
                raise ValueError(
                    "observable_weights must be provided as a mapping when "
                    "observable_allocation='weights'."
                )
            missing_keys = set(observables) - set(weights)
            if missing_keys:
                raise ValueError(
                    "observable_weights is missing weights for "
                    f"{sorted(missing_keys, key=str)}."
                )
            weight_values = {
                key: weights[key]
                for key in observables
            }
            if any(weight < 0 for weight in weight_values.values()):
                raise ValueError("observable weights must be non-negative.")
            weight_sum = sum(weight_values.values())
            if weight_sum <= 0:
                raise ValueError("At least one observable weight must be positive.")
            normalized_weights = {
                key: weight / weight_sum
                for key, weight in weight_values.items()
            }

        self.observable_shot_allocation = self._distribute_integer_budget(
            self.total_shots,
            normalized_weights,
        )
        return self.observable_shot_allocation

    def evaluate(self, circuit, observables, observable_weights=None):
        """Before evaluation, circuit and observables must be specified"""
        if circuit is None:
            raise ValueError("Specifify a circuit to evaluate")
        if observables is None:
            raise ValueError("Specifify observables to evaluate")

        observables = self._as_observable_dict(observables)
        self.observable_shot_allocation = {}
        self.group_shot_allocation = {}

        if self.batching == "global":
            raise NotImplementedError("Global batching is not implemented yet.")

        return self._evaluate_local(
            circuit,
            observables,
            observable_weights=observable_weights,
        )

    def _evaluate_local(self, circuit, observables, observable_weights=None):
        """Evaluate observables using per-observable local commuting groups."""
        if self.total_shots is not None:
            observable_shots = self._allocate_observable_shots(
                observables,
                weights=observable_weights,
            )
        elif self.shots_per_group is not None:
            observable_shots = None
        else:
            raise ValueError("Specify total_shots or shots_per_group.")

        values = {}
        for key, observable in observables.items():
            grouped_terms = measurement_basis_rotations(
                observable,
                grouping=self.grouping,
            )

            if self.shots_per_group is not None:
                self.group_shot_allocation[key] = [
                    self.shots_per_group
                    for _ in grouped_terms
                ]
                values[key] = expectation_from_samples(
                    circuit,
                    observable,
                    n_shots=self.shots_per_group,
                    grouping=self.grouping,
                    n_shots_per_pauli_term=True,
                )
                continue

            observable_budget = observable_shots[key]
            if self.group_allocation == "variance":
                raise NotImplementedError(
                    "Variance-based group allocation not implemented yet."
                )

            method = {
                "uniform": "u",
                "coefficients": "c",
            }[self.group_allocation]
            allocation_list = (
                allocate_shots(grouped_terms, observable_budget, method=method)
                if grouped_terms
                else []
            )
            self.group_shot_allocation[key] = allocation_list
            values[key] = expectation_from_samples(
                circuit,
                observable,
                n_shots=observable_budget,
                grouping=self.grouping,
                n_shots_per_pauli_term=False,
                shot_allocation=allocation_list,
            )

        return values


# Future global batching:
#
# observables
# -> extract global Pauli terms
# -> deduplicate
# -> build global commuting groups
# -> execute shared measurement circuits
# -> reconstruct observable expectations


# Future AdaptiveMatrixShotProtocol idea:
#
# 1. Initial rough measurement
# 2. Assemble H and S matrices
# 3. Solve generalized eigenproblem
# 4. Determine observable importance weights
# 5. Allocate additional shots between observables
# 6. Re-evaluate important observables with more shots
#
# This future protocol should build on top of ShotProtocol.
