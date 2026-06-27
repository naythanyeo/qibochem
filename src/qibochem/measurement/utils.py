"""Shared helpers for measurement protocols."""

from collections import defaultdict
from collections.abc import Mapping

import numpy as np

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


def get_final_state(circuit_or_state):
    """Return a statevector from either a circuit or a statevector input."""
    if circuit_or_state is None:
        raise ValueError("Specify a circuit or statevector to evaluate.")

    if not hasattr(circuit_or_state, "_final_state"):
        final_state = np.asarray(circuit_or_state, dtype=complex)
        if final_state.ndim != 1:
            raise ValueError("Statevector input must be one-dimensional.")
        return final_state

    if circuit_or_state._final_state is None:
        circuit_or_state()

    return circuit_or_state._final_state.state()


def collect_global_unique_terms(observables):
    global_terms = []
    seen_terms = set() # For comparison by hashing

    for observable in observables.values():
        for term in observable.terms:
            if term == (0, 0, 0) or term in seen_terms:
                continue

            seen_terms.add(term)
            global_terms.append(term)

    return global_terms