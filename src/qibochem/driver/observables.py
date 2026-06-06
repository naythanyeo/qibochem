"""
Lightweight observable class object for storage of observables 
In general using the hamiltonian as a observable is very heavy and impractical for repeated calculations 
For the general hamiltonian objects, usage of sympy is also very heavy and expensive 
This lightweight observable class is built on bitmask tuples which save memory and computational time 
Protocol class is also built on top of this observable class, ie all evaluations will use this bitmasks 
format to input observables and evaluate them. In general hamiltonian can exist in various forms but 
the efficient form for comparison and calculations is bitmasks, which significnantly out perform sympy objects

General structure of observable 
constant: complex number 
terms: {(bitmask x, bitmask y, bitmask z): coeff}

Here we first include helper functions to convert the qubit form of observables into the bitmask form

"""

from dataclasses import dataclass
from collections import defaultdict

@dataclass(slots=True)
class BitmaskObservable:
    constant: complex
    terms: dict

def qubit_term2bitmask(qubit_term):
    """
    Helper function to convert qubit terms into bitmasks format
    Convert qubit operator key (from open fermion) into bitmasks 
    Input: ((0, "Y"), (1, "X")), (int, str)

    Output: (x_mask, y_mask, z_mask, weight)
    """
    x_mask = y_mask = z_mask = 0

    for qubit_index, pauli_term in qubit_term:
        if pauli_term == "X":
            x_mask |= 1 << qubit_index
        elif pauli_term == "Y":
            y_mask |= 1 << qubit_index
        elif pauli_term == "Z":
            z_mask |= 1 << qubit_index
        else:
            raise ValueError(f"Unknown Pauli: {pauli_term}")

    return x_mask, y_mask, z_mask

def qubit_operator2observable(qubit_operator):
    """
    Function to convert an OpenFermion QubitOperator object into bitmasks observables 
    Current qibochem qubit hamiltonian object is a OpenFermion QubitOperator object 
    It has attributes of .terms which will return individual qubit terms 
    This function converts that object into the general BitmaskObservable class which is default used
    for all forms of evaluations and calculations (rotations/evaluations/groupings) as its faster
    """
    constant = 0
    terms = defaultdict(complex)
    for qubit_term, coeff in qubit_operator.terms.items():
        if qubit_term == ():
            constant += coeff
        else:
            bitmask = qubit_term2bitmask(qubit_term)
            terms[bitmask] += coeff
    return BitmaskObservable(constant = constant, terms = terms)