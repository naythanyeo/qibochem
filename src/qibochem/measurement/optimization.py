"""
Functions for optimising the measurement cost of obtaining the expectation value
"""

import networkx as nx
import sympy as sp
from qibo import gates
from qibo.symbols import X, Y, Z
from sympy.core.numbers import One


def term_to_string(term):
    """
    Convert a single Pauli term (:class:`sympy.Expr`) to its string representation. Drops the coefficient and will not
    check if input is a float!!
    """
    return " ".join(str(_x) for _x in term.args if isinstance(_x, (X, Y, Z))) if term.args else str(term)


def check_terms_commutativity(term1: str, term2: str, qubitwise: bool):
    """
    Check if terms 1 and 2 are mutually commuting. The 'qubitwise' flag determines if the check is for general
    commutativity (False), or the stricter qubitwise commutativity.

    Args:
        term1/term2: Strings representing a single Pauli term. E.g. "X0 Z1 Y3". Obtained from a Qibo SymbolicTerm as
        ``" ".join(factor.name for factor in term.factors)``.
        qubitwise (bool): Determines if the check is for general commutativity, or the stricter qubitwise commutativity

    Returns:
        bool: Do terms 1 and 2 commute?
    """
    # Get a list of common qubits for each term
    common_qubits = {_term[1:] for _term in term1.split() if _term[0] != "I"} & {
        _term[1:] for _term in term2.split() if _term[0] != "I"
    }
    if not common_qubits:
        return True
    # Get the single Pauli operators for the common qubits for both Pauli terms
    term1_ops = [_op for _op in term1.split() if _op[1:] in common_qubits]
    term2_ops = [_op for _op in term2.split() if _op[1:] in common_qubits]
    if qubitwise:
        # Qubitwise: Compare the Pauli terms at the common qubits. Any difference => False
        return all(_op1 == _op2 for _op1, _op2 in zip(term1_ops, term2_ops))
    # General commutativity:
    # Get the number of single Pauli operators that do NOT commute
    n_noncommuting_ops = sum(_op1 != _op2 for _op1, _op2 in zip(term1_ops, term2_ops))
    # term1 and term2 have general commutativity iff n_noncommuting_ops is even
    return n_noncommuting_ops % 2 == 0

def group_commuting_terms(terms_list, qubitwise):
    """
    Groups the terms in terms_list into as few groups as possible, where all the terms in each group commute
    mutually == Finding the minimum clique cover (i.e. as few cliques as possible) for the graph whereby each node
    is a Pauli string, and an edge exists between two nodes iff they commute.

    This is equivalent to the graph colouring problem of the complement graph (i.e. edge between nodes if they DO NOT
    commute), which this function follows.

    Args:
        terms_list: List of strings. The strings should follow the output from
            ``" ".join(factor.name for factor in term.factors)``, where term is a Qibo SymbolicTerm. E.g. "X0 Z1".
        qubitwise: Determines if the check is for general commutativity, or the stricter qubitwise commutativity

    Returns:
        list: Containing groups (lists) of Pauli strings that all commute mutually
    """
    G = nx.Graph()
    # Complement graph: Add all the terms as nodes first, then add edges between nodes if they DO NOT commute
    G.add_nodes_from(terms_list)
    G.add_edges_from(
        (term1, term2)
        for _i1, term1 in enumerate(terms_list)
        for _i2, term2 in enumerate(terms_list)
        if _i2 > _i1 and not check_terms_commutativity(term1, term2, qubitwise)
    )
    # Solve using Greedy Colouring on NetworkX
    sorted_groups = nx.coloring.greedy_color(G)
    group_ids = set(sorted_groups.values())
    # Sort results so that test results will be replicable
    term_groups = sorted(
        sorted(group for group, group_id in sorted_groups.items() if group_id == _id) for _id in group_ids
    )
    return term_groups


def qwc_measurement_gates(expression):
    """
    Get the list of (basis rotation) measurement gates to be added to the circuit. The measurements from the resultant
    circuit can then be used to obtain the expectation values of ALL the terms in expression directly.

    Args:
        expression (sympy.Expr): Group of Pauli terms that all mutually commute with each other qubitwise

    Returns:
        list: Measurement gates to be appended to the Qibo circuit
    """
    m_gates, _m_gates = {}, {}
    # Single Pauli operator
    if not expression.args:
        return [gates.M(expression.target_qubit, basis=type(expression.gate))]
    # Either a single Pauli term or a sum of Pauli terms
    for term in expression.args:
        # Term should either be a single Pauli operator or a Pauli string
        if isinstance(term, (X, Y, Z)):
            _m_gates = {term.target_qubit: gates.M(term.target_qubit, basis=type(term.gate))}
        else:
            _m_gates = {
                pauli_op.target_qubit: gates.M(pauli_op.target_qubit, basis=type(pauli_op.gate))
                for pauli_op in term.args
                if hasattr(pauli_op, "target_qubit") and m_gates.get(pauli_op.target_qubit) is None
            }
        m_gates = {**m_gates, **_m_gates}
    return list(m_gates.values())


def qwc_measurements(terms):
    """
    Sort out a list of Hamiltonian terms into separate groups of mutually qubitwise commuting terms, and returns the
    grouped terms along with their associated measurement gates

    Args:
        terms: List of (sympy.Expr, coeff) terms, without constant terms

    Returns:
        list: List of two-tuples, with each tuple given as (sorted_ham, [`list of measurement gates`]), where
            sorted_ham is a :class:`qibo.hamiltonians.SymbolicHamiltonian`
    """
    # Build dictionary with keys = string representation of the terms, values = corresponding (sympy.Expr, term coeff)
    ham_terms = {term_to_string(term): (term, coeff) for term, coeff in terms}
    term_groups = group_commuting_terms(ham_terms.keys(), qubitwise=True)
    return [
        (
            sum(ham_terms[term][1] * ham_terms[term][0] for term in term_group),  # Original expression: coeff*term
            qwc_measurement_gates(sum(ham_terms[term][0] for term in term_group)),  # No coeff for qwc_measurement_gates
        )
        for term_group in term_groups
    ]

"""
New function for fast commuting grouping 
the previous method of creating graphs can probably find optimal solution but currently 
it also does greedy colouring which is also not optimal, and a lot slower 
"""


def _qwc_mask(term):
    """
    Convert one SymPy Pauli term into bitmasks.
    Input: SymPy Pauli term, e.g. X0*Z2*Y5

    Output: (x_mask, y_mask, z_mask, weight)

    Bitmask meaning:
        X0*Z2*Y5 -> x_mask has bit 0, z_mask has bit 2, y_mask has bit 5.
    Can use strings to compare but we are having many many terms comparing with each other
    especially for larger H, so using bitmasks to compare is the most basic and fastest
    """
    x_mask = y_mask = z_mask = 0
    factors = term.args if term.args else (term,)

    for factor in factors:
        if not isinstance(factor, (X, Y, Z)):
            continue
        # Represent the qubit index as a bitstring with a 1, eg X0 --> 0b001, X1 --> 0b010
        bit = 1 << factor.target_qubit
        # Add on the bit to the corresponding mask
        if isinstance(factor, X):
            x_mask |= bit
        elif isinstance(factor, Y):
            y_mask |= bit
        else:
            z_mask |= bit
    # Count number of terms for sorting so later can do largest first grouping
    weight = (x_mask | y_mask | z_mask).bit_count()
    return x_mask, y_mask, z_mask, weight


def _qwc_compatible(group_mask, term_mask):
    """
    Check whether a term can be added to a QWC group.
    Input: group_mask / term_mask: (x_mask, y_mask, z_mask, weight)

    Compatibility rule:
        A new term cannot introduce a different Pauli basis on any qubit
        that is already fixed by the group.
    """
    group_x, group_y, group_z, _ = group_mask
    term_x, term_y, term_z, _ = term_mask
    # Basically saying that each bit cannot have a new term with different operator
    # Eg group_x is 0b001(X0), group_y is 0b010(Y1), then ok, but if group_y is 0b011 (Y0Y1)
    # then they cannot match because 0th qubit conflicts
    return not (
        group_x & (term_y | term_z)
        or group_y & (term_x | term_z)
        or group_z & (term_x | term_y)
    )


def qwc_fast_measurements(terms):
    """
    Greedily pack Pauli terms into qubitwise-commuting measurement groups.
    Input: List of (sympy.Expr, coeff) terms, without constant terms

    Internal data format:
        term_data = [(term, coeff, term_mask), ...]
        groups = [
            {
                "mask": (x_mask, y_mask, z_mask, weight),
                "terms": [(term, coeff), ...],
            },
            ...
        ]

    Largest-first ordering places longer Pauli strings first, to try to be as 
    close to optimal grouping as possible while still being fast
    Output format:
        [(group_expression, measurement_gates), ...]
    """
    # Convert every term into bitmasks for comparison later
    term_data = [(term, coeff, _qwc_mask(term)) for term, coeff in terms]
    # Sort by the weights so largest groups get grouped first
    term_data.sort(key=lambda data: (-data[2][3], str(data[0])))
    # Loop through all the terms and put them into groups, check compatibility then 
    # update each group mask when new term is added
    groups = []
    for term, coeff, term_mask in term_data:
        # For every term, check all the existing groups, if it doesnt exist then make new one
        for group in groups:
            if _qwc_compatible(group["mask"], term_mask):
                # Define the new group mask terms
                group_x, group_y, group_z, _ = group["mask"]
                term_x, term_y, term_z, _ = term_mask
                x_mask = group_x | term_x
                y_mask = group_y | term_y
                z_mask = group_z | term_z

                group["mask"] = (x_mask, y_mask, z_mask, 
                                 (x_mask | y_mask | z_mask).bit_count()) # Weight
                group["terms"].append((term, coeff))
                break
        else:
            # Create new group if not compatible
            groups.append({"mask": term_mask, "terms": [(term, coeff)]})

    result = []
    # Reconstruct the grouping expressions in to the symbolic groups
    for group in groups:
        x_mask, y_mask, z_mask, _ = group["mask"]
        basis_mask = x_mask | y_mask | z_mask
        measurement_gates = []

        for qubit in range(basis_mask.bit_length()):
            bit = 1 << qubit
            if x_mask & bit:
                measurement_gates.append(gates.M(qubit, basis=gates.X))
            elif y_mask & bit:
                measurement_gates.append(gates.M(qubit, basis=gates.Y))
            elif z_mask & bit:
                measurement_gates.append(gates.M(qubit, basis=gates.Z))

        group_expression = sp.Add(*(coeff * term for term, coeff in group["terms"]))
        result.append((group_expression, measurement_gates))

    return result


def measurement_basis_rotations(hamiltonian_or_terms, grouping=None):
    """
    Split up and sort the Hamiltonian terms to get the basis rotation gates to be applied to a quantum circuit for the
    respective (group of) terms in the Hamiltonian

    Args:
        hamiltonian_or_terms: Hamiltonian of interest, or a list of (sympy.Expr, coeff) terms
        grouping (str): Whether or not to group Hamiltonian terms together, i.e. use the same set of measurements to get
            the expectation values of a group of terms simultaneously. Default value of ``None`` will not group any
            terms together. ``"qwc"`` uses graph colouring, while ``"qwc_fast"`` uses largest-first greedy QWC basis
            packing. Both return the measurement gates associated with each group of terms

    Returns:
        list: List of two-tuples; the first item in the tuple is a group of Pauli terms (:class:`sympy.Expr`), and the
        second is a list of measurement gates (:class:`qibo.gates.M`) that can be used to get the expectation value
        for the corresponding expression.
    """
    # First convert hamiltonian into list of terms
    if hasattr(hamiltonian_or_terms, "form"):
        terms = [
            (term, coeff)
            for term, coeff in hamiltonian_or_terms.form.as_coefficients_dict().items()
            if term != 1 and not isinstance(term, One)
        ]
    else:
        terms = list(hamiltonian_or_terms)

    result = []
    if grouping is None:
        result += [(coeff * term, qwc_measurement_gates(term)) for term, coeff in terms]
    elif grouping == "qwc":
        result += qwc_measurements(terms)
    elif grouping == "qwc_fast":
        result += qwc_fast_measurements(terms)
    else:
        raise NotImplementedError("Not ready yet!")
    return result
