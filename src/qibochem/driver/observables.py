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
Also helper functions to multiply terms
"""

from dataclasses import dataclass
from collections import defaultdict

@dataclass(slots=True)
class BitmaskObservable:
    constant: complex
    terms: dict

    def scale(self, scalar, threshold=0.0):
        """Return a new observable multiplied by a scalar."""
        constant = scalar * self.constant
        terms = {
            term: scalar * coeff
            for term, coeff in self.terms.items()
            if abs(scalar * coeff) > threshold
        }
        if abs(constant) <= threshold:
            constant = 0.0

        return BitmaskObservable(constant=constant, terms=terms)

def qubit_term2bitmask(qubit_term, n_qubits=None):
    """
    Helper function to convert qubit terms into bitmasks format
    Convert qubit operator key (from open fermion) into bitmasks 
    Input: ((0, "Y"), (1, "X")), (int, str)

    Output: (x_mask, y_mask, z_mask)
    NOTE: Following qibochem convention, the bitmasks are ordered such that X0 becomes 1000, Y1 because 0100... 
    Ie, the qubit counting starts from the left, not right. 
    """
    if n_qubits is None:
        n_qubits = max((qubit_index for qubit_index, _ in qubit_term), default=-1) + 1

    x_mask = y_mask = z_mask = 0

    for qubit_index, pauli_term in qubit_term:
        bit = 1 << (n_qubits - 1 - qubit_index)
        if pauli_term == "X":
            x_mask |= bit
        elif pauli_term == "Y":
            y_mask |= bit
        elif pauli_term == "Z":
            z_mask |= bit
        else:
            raise ValueError(f"Unknown Pauli: {pauli_term}")

    return x_mask, y_mask, z_mask

def qubit_operator2observable(qubit_operator, n_qubits=None):
    """
    Function to convert an OpenFermion QubitOperator object into bitmasks observables 
    Current qibochem qubit hamiltonian object is a OpenFermion QubitOperator object 
    It has attributes of .terms which will return individual qubit terms 
    This function converts that object into the general BitmaskObservable class which is default used
    for all forms of evaluations and calculations (rotations/evaluations/groupings) as its faster
    """
    if n_qubits is None:
        n_qubits = max(
            (qubit_index + 1 for qubit_term in qubit_operator.terms for qubit_index, _ in qubit_term),
            default=0,
        )

    constant = 0
    terms = defaultdict(complex)
    for qubit_term, coeff in qubit_operator.terms.items():
        if qubit_term == ():
            constant += coeff
        else:
            bitmask = qubit_term2bitmask(qubit_term, n_qubits=n_qubits)
            terms[bitmask] += coeff
    return BitmaskObservable(constant = constant, terms = terms)


"""
Here we define some simple bitmask operations between pauli gates 
For addition between masks, a simple | OR function will suffice
For multiplication between bitmask, we need to construct a multiplication map

Helper function here will allow for multiplication of bitmask terms 
"""

def multiply_bitmask_terms(a, b):
    """
    Multiply two Pauli terms in bitmask form.

    Args:
        a, b: (x_mask, y_mask, z_mask)

    Returns:
        ((x_mask, y_mask, z_mask), phase)
    """
    ax, ay, az = a
    bx, by, bz = b

    a_occ = ax | ay | az
    b_occ = bx | by | bz

    only_ax = ax & ~b_occ
    only_ay = ay & ~b_occ
    only_az = az & ~b_occ

    only_bx = bx & ~a_occ
    only_by = by & ~a_occ
    only_bz = bz & ~a_occ

    rx = only_ax | only_bx | (ay & bz) | (az & by)
    ry = only_ay | only_by | (ax & bz) | (az & bx)
    rz = only_az | only_bz | (ax & by) | (ay & bx)

    plus_i_count = (
        (ax & by).bit_count() +  # X Y = iZ
        (ay & bz).bit_count() +  # Y Z = iX
        (az & bx).bit_count()    # Z X = iY
    )

    minus_i_count = (
        (ay & bx).bit_count() +  # Y X = -iZ
        (az & by).bit_count() +  # Z Y = -iX
        (ax & bz).bit_count()    # X Z = -iY
    )

    phase = (1j) ** (plus_i_count - minus_i_count)

    return (rx, ry, rz), phase


def multiply_bitmask_observables(left, right, threshold=0.0):
    """
    Multiply two BitmaskObservable objects.

    Identity contributions are stored in ``constant`` to preserve the
    BitmaskObservable convention. Callers that need explicit identity keys can
    emit ``(0, 0, 0): constant`` at the output boundary.
    """
    constant = left.constant * right.constant
    terms = defaultdict(complex)

    for term, coeff in left.terms.items():
        terms[term] += coeff * right.constant

    for term, coeff in right.terms.items():
        terms[term] += left.constant * coeff

    for left_term, left_coeff in left.terms.items():
        for right_term, right_coeff in right.terms.items():
            term, phase = multiply_bitmask_terms(left_term, right_term)
            coeff = left_coeff * right_coeff * phase
            if term == (0, 0, 0):
                constant += coeff
            else:
                terms[term] += coeff

    filtered_terms = {
        term: coeff
        for term, coeff in terms.items()
        if abs(coeff) > threshold
    }
    if abs(constant) <= threshold:
        constant = 0.0

    return BitmaskObservable(constant=constant, terms=filtered_terms)


def add_bitmask_observables(left, right, threshold=0.0):
    """Add two BitmaskObservable objects."""
    constant = left.constant + right.constant
    terms = defaultdict(complex)

    for term, coeff in left.terms.items():
        terms[term] += coeff

    for term, coeff in right.terms.items():
        terms[term] += coeff

    filtered_terms = {
        term: coeff
        for term, coeff in terms.items()
        if abs(coeff) > threshold
    }
    if abs(constant) <= threshold:
        constant = 0.0

    return BitmaskObservable(constant=constant, terms=filtered_terms)
