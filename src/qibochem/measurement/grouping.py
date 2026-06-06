"""
Functions for optimising the measurement cost of obtaining the expectation value
"""


"""
New function for fast commuting grouping 
the previous method of creating graphs can probably find optimal solution but currently 
it also does greedy colouring which is also not optimal, and a lot slower 
Edit to input bitmask function directly, no more SYMPY obejects 

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
