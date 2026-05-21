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

@dataclass
class ShotProtocol:
    """Protocol for evaluating observables on a shot-based simulator.
    Each observable is a qibo.hamiltonian object
    Observables and Circuit can be specified after initialising the Protocol Object
    """

    total_shots: int | None = 10000 # Default total shot budget for all observables to be 10k shots
    observable_shots: dict | None = None # Can manually specify shot count for each observable
    # If left unspecified, shots will be allocated uniformly across all observables
    shot_allocation_mode: str = "Uniform" # For grouping individual commuting terms within each observable
    grouping: str | None = "qwc" # Affects local Grouping
    trial_shots: int | None = 100 # Only for variance shot allocation mode

    def __post_init__(self):
        """Validate shot protocol configuration."""
        if self.shot_allocation_mode not in ("Uniform", "Coeff", "Var"):
            raise ValueError("shot_allocation_mode must be 'Uniform', 'Coeff', or 'Var'.")
        if self.grouping not in ("qwc", None):
            raise ValueError("Grouping currently only supports 'qwc' or None.")
        if not isinstance(self.trial_shots, int) or self.trial_shots <= 0:
            raise ValueError("trial_shots must be a positive integer")
        if not isinstance(self.total_shots, int) or self.total_shots <= 0:
            raise ValueError("total_shots must be a positive integer")
        # The observables if manually specified must be constarinted 
        if self.observable_shots is not None:
            for key, value in self.observable_shots.items():
                if not isinstance(value, int) or value <= 0:
                    raise ValueError("observable_shots values must be positive integers.")
            observable_shot_total = sum(self.observable_shots.values())
            # If total shots is unspecified, it will follow the manually input observable_shots
            if self.total_shots is None: 
                self.total_shots = observable_shot_total
            elif self.total_shots != observable_shot_total:
                raise ValueError("observable_shots values must sum to total_shots.")

    def allocate_observable_shots(self, observables):
        """Allocate shot budgets between observables."""
        if not observables:
            raise ValueError("observables cannot be empty.")

        # Can only validate this after observables is passed in for manually added shot allocation
        if self.observable_shots is not None:
            if set(self.observable_shots) != set(observables):
                raise ValueError("observable_shots keys must match observable keys.")
            return self.observable_shots

        # If not specified manually, the allocate all the shots to each observable uniformly
        base_shots = self.total_shots // len(observables)
        remainder = self.total_shots % len(observables)
        if base_shots == 0:
            raise ValueError("total_shots must assign at least one shot to each observable.")
        # Uniform distribution but preserve integer shot count
        self.observable_shots = {label: base_shots for label in observables}
        for label in list(observables)[:remainder]:
            self.observable_shots[label] += 1

        return self.observable_shots

    def evaluate(self, circuit, observables):
        """Before evaluation, circuit and observables must be specified
        These are not initialised so that protocol can be defined separately"""
        if circuit is None:
            raise ValueError("Specifify a circuit to evaluate")
        if observables is None:
            raise ValueError("Specifify observables to evaluate")

        if isinstance(observables, Mapping):
            is_mapping = True
            observables = dict(observables)
        else: # Allows for single observable input 
            is_mapping = False
            observables = {"observable": observables}

        # First allocate shots to each observable 
        self.allocate_observable_shots(observables)
        results = {}
        for label, observable in observables.items():
            num_allocated_shots = self.observable_shots[label]

            # For variance shot allocation, run its own wrapper function 
            # That wrapper function does variance sampling automatically 
            if self.shot_allocation_mode == "Var":
                results[label] = v_expectation(circuit, observable,
                                               n_shots=num_allocated_shots,
                                               n_trial_shots=self.trial_shots,
                                               grouping=self.grouping, method="vmsa")
                continue
            
            grouped_terms = measurement_basis_rotations(observable, grouping=self.grouping)
            method = {"Uniform": "u", "Coeff": "c"}[self.shot_allocation_mode]
            shot_allocation = allocate_shots(grouped_terms, num_allocated_shots, method=method)
            
            results[label] = expectation_from_samples(circuit, observable,
                                                      n_shots=num_allocated_shots,
                                                      grouping=self.grouping,
                                                      n_shots_per_pauli_term=False,
                                                      shot_allocation=shot_allocation)

        if is_mapping:
            return results
        return results["observable"] # Output one expectation value if input is a single observable 


# Future AdaptiveMatrixShotProtocol:
#
# 1. Initial rough measurement
# 2. Assemble H and S matrices
# 3. Solve generalized eigenproblem
# 4. Determine important matrix elements
# 5. Allocate more shots to important observables
# 6. Re-measure selected observables
#
# This future protocol should build on top of ShotProtocol.


# Future global batching:
#
# observables
# -> extract global Pauli terms
# -> deduplicate terms
# -> build global commuting groups
# -> execute shared measurement circuits
# -> reconstruct all observables from cached Pauli expectations
