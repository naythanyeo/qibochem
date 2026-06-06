"""Shared helpers for measurement protocols."""

from collections import defaultdict
from collections.abc import Mapping

from qibochem.driver.observables import BitmaskObservable


def normalise_observables(observables):
    """Return observables as a dictionary and validate their type."""
    if not isinstance(observables, Mapping):
        observables = {"O1": observables}

    for observable in observables.values():
        if not isinstance(observable, BitmaskObservable):
            raise TypeError("Protocols only accept BitmaskObservable inputs.")

    return observables


def format_output(full_observables_data):
    """Flatten protocol output to the most compact shape."""
    if len(full_observables_data) == 1:
        _, observable_values = list(full_observables_data.items())[0]
        if len(observable_values) == 1:
            _, value = list(observable_values.items())[0]
            return value
        return dict(observable_values)

    if all(len(observable_values) == 1 for observable_values in full_observables_data.values()):
        return {
            sample_key: list(observable_values.values())[0]
            for sample_key, observable_values in full_observables_data.items()
        }

    return dict(full_observables_data)


def get_final_state(circuit):
    """Run the circuit if necessary and return its statevector."""
    if circuit is None:
        raise ValueError("Specify a circuit to evaluate.")

    if circuit._final_state is None:
        circuit()

    return circuit._final_state.state()


def collect_global_terms(observables):
    """Collect all nonconstant bitmask terms across all observables."""
    global_terms = []

    for observable in observables.values():
        for term in observable.terms:
            if term != (0, 0, 0):
                global_terms.append(term)

    return global_terms


def transpose_group_probabilities(group_probabilities):
    """Convert {group: {sample: vector}} to {sample: {group: vector}}."""
    sample_probabilities = defaultdict(dict)

    for group_index, sampled_probabilities in group_probabilities.items():
        for sample_key, probability_vector in sampled_probabilities.items():
            sample_probabilities[sample_key][group_index] = probability_vector

    return sample_probabilities
