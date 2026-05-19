"""
Utility functions that can be used by different ansatzes
"""


def mp2_amplitude(excitation, orbital_energies, tei):
    r"""
    Calculate the MP2 guess amplitude for a single UCC circuit: 0.0 for a single excitation.
        for a double excitation (In SO basis): :math:`t_{ij}^{ab} = (g_{ijab} - g_{ijba}) / (e_i + e_j - e_a - e_b)`

    Args:
        excitation: Iterable of spin-orbitals representing a excitation. Must have either 2 or 4 elements exactly,
            representing a single or double excitation respectively.
        orbital_energies: eigenvalues of the Fock operator, i.e. orbital energies
        tei: Two-electron integrals in MO basis and second quantization notation

    Returns:
        MP2 guess amplitude (float)
    """
    # Checks validity of excitation argument
    assert len(excitation) % 2 == 0 and len(excitation) // 2 <= 2, f"{excitation} must have either 2 or 4 elements"
    # If single excitation, can just return 0.0 directly
    if len(excitation) == 2:
        return 0.0

    # Convert orbital indices to be in MO basis
    mo_orbitals = [orbital // 2 for orbital in excitation]
    # Numerator: g_ijab - g_ijba
    g_ijab = (
        tei[tuple(mo_orbitals)]  # Can index directly using the MO TEIs
        if (excitation[0] + excitation[3]) % 2 == 0 and (excitation[1] + excitation[2]) % 2 == 0
        else 0.0
    )
    g_ijba = (
        tei[tuple(mo_orbitals[:2] + mo_orbitals[2:][::-1])]  # Reverse last two terms
        if (excitation[0] + excitation[2]) % 2 == 0 and (excitation[1] + excitation[3]) % 2 == 0
        else 0.0
    )
    numerator = g_ijab - g_ijba
    # Denominator is directly from the orbital energies
    denominator = sum(orbital_energies[mo_orbitals[:2]]) - sum(orbital_energies[mo_orbitals[2:]])
    return numerator / denominator


def generate_excitations(order, excite_from, excite_to, conserve_spin=True):
    """
    Generate all possible excitations between a list of occupied and virtual orbitals

    Args:
        order: Order of excitations, i.e. 1 == single, 2 == double
        excite_from: Iterable of integers
        excite_to: Iterable of integers
        conserve_spin: ensure that the total electronic spin is conserved

    Return:
        List of lists, e.g. [[0, 1]]
    """
    # If order of excitation > either number of electrons/orbitals, return list of empty list
    if order > min(len(excite_from), len(excite_to)):
        return [[]]

    # Generate all possible excitations first
    from itertools import combinations  # pylint: disable=C0415

    all_excitations = [
        [*_from, *_to] for _from in combinations(excite_from, order) for _to in combinations(excite_to, order)
    ]
    # Filter out the excitations if conserve_spin set
    if conserve_spin:
        # Not sure if this filtering is exhaustive; might not remove some redundant excitations?
        all_excitations = [
            _ex
            for _ex in all_excitations
            if sum(_ex) % 2 == 0 and (sum(_i % 2 for _i in _ex[:order]) == sum(_i % 2 for _i in _ex[order:]))
        ]
    return all_excitations


def sort_excitations(excitations):
    """
    Sorts a list of excitations according to some common-sense and empirical rules (see below).
        The order of the excitations must be the same throughout.

    Sorting order:
    1. (For double excitations only) All paired excitations between the same MOs first
    2. Pair up excitations between the same MOs, e.g. (0, 2) and (1, 3)
    3. Then count upwards from smallest MO

    Args:
        excitations: List of iterables, each representing an excitation; e.g. [[1, 5], [0, 4]]

    Returns:
        List of excitations after sorting
    """
    # Check that all excitations are of the same order and <= 2
    order = len(excitations[0]) // 2
    if order > 2:
        raise NotImplementedError("Can only handle single and double excitations!")
    assert all(len(_ex) // 2 == order for _ex in excitations), "Cannot handle excitations of different orders!"

    # Define variables for the while loop
    copy_excitations = [list(_ex) for _ex in excitations]
    result = []
    prev = []

    # No idea how I came up with this, but it seems to work for double excitations
    def sorting_fn(x):
        # Default sorting is OK for single excitations
        return sum((order + 1 - _i) * abs(x[2 * _i + 1] // 2 - x[2 * _i] // 2) for _i in range(0, order))

    # Make a copy of the list of excitations, and use it populate a new list iteratively
    while copy_excitations:
        if not prev:
            # Take out all pair excitations first
            pair_excitations = [
                _ex
                for _ex in copy_excitations
                # Indices of the electrons/holes must be consecutive numbers
                if sum(abs(_ex[2 * _i + 1] // 2 - _ex[2 * _i] // 2) for _i in range(0, order)) == 0
            ]
            while pair_excitations:
                pair_excitations = sorted(pair_excitations)
                ex_to_remove = pair_excitations.pop(0)
                if ex_to_remove in copy_excitations:
                    # 'Move' the first pair excitation from copy_excitations to result
                    index = copy_excitations.index(ex_to_remove)
                    result.append(copy_excitations.pop(index))

            # No more pair excitations, only remaining excitations should have >=3 MOs involved
            # Sort the remaining excitations
            copy_excitations = sorted(copy_excitations, key=sorting_fn if order != 1 else None)
        else:
            # Check to see for excitations involving the same MOs as prev
            _from = prev[:order]
            _to = prev[order:]
            # Get all possible excitations involving the same MOs
            new_from = [_i + 1 if _i % 2 == 0 else _i - 1 for _i in _from]
            new_to = [_i + 1 if _i % 2 == 0 else _i - 1 for _i in _to]
            same_mo_ex = [sorted(list(_f) + list(_t)) for _f in (_from, new_from) for _t in (_to, new_to)]
            # Remove the excitations with the same MOs from copy_excitations
            while same_mo_ex:
                same_mo_ex = sorted(same_mo_ex)
                ex_to_remove = same_mo_ex.pop(0)
                if ex_to_remove in copy_excitations:
                    # 'Move' the first entry of same_mo_index from copy_excitations to result
                    index = copy_excitations.index(ex_to_remove)
                    result.append(copy_excitations.pop(index))
            prev = None
            continue
        if copy_excitations:
            # Remove the first entry from the sorted list of remaining excitations and add it to result
            prev = copy_excitations.pop(0)
            result.append(prev)
    return result


# UCCSD Excitation Generator
def generate_UCCSD_excitations(n_elec, n_orbs):
    param_excitations = {}  

    double_excitations = sort_excitations(
        generate_excitations(2, range(n_elec), range(n_elec, n_orbs))
    )
    single_excitations = sort_excitations(
        generate_excitations(1, range(n_elec), range(n_elec, n_orbs))
    )

    for index, excitation in enumerate(double_excitations):
        param_excitations[f"d{index}"] = [tuple(excitation)]
    for index, excitation in enumerate(single_excitations):
        param_excitations[f"s{index}"] = [tuple(excitation)]

    return param_excitations

# UCCSD Singlet Excitation Generator
def generate_UCCSDSinglet_excitations(n_elec, n_orbs):
    param_excitations = {}

    double_excitations = sort_excitations(
        generate_excitations(2, range(n_elec), range(n_elec, n_orbs))
    )
    single_excitations = sort_excitations(
        generate_excitations(1, range(n_elec), range(n_elec, n_orbs))
    )

    def _spin2spatial(spin_orbitals):
        return tuple(orb // 2 for orb in spin_orbitals)
    
    # For singles excitations, group all the same term together 
    singlet_pairs = {}
    for excitation in single_excitations:
        spatial_pair = _spin2spatial(excitation)
        if spatial_pair not in singlet_pairs:
            singlet_pairs[spatial_pair] = []
        singlet_pairs[spatial_pair].append(tuple(excitation))

    # KIV function, might modify generate_excitaions 
    # For UCCSDSinglet current the spin orbital grouping cannot be straight forwardly implemented 
    # Generate_excitations outputs 0, 3, 5, 6 but not 0, 3, 6, 5 for example 
    # Here to preserve spin 0>6, 3>5 which will be different from say 1, 2, 5, 6 which is 1>5 2>6
    # KIV relook to modify generate_excitations so that the output has all combinations 
    # But also preserve spin in the same excitation order for doubles 
    singlet_doubles = {}
    for excitation in double_excitations:
        excitation = tuple(excitation)
        direct_pairs = [(excitation[0], excitation[2]), (excitation[1], excitation[3])]
        cross_pairs = [(excitation[0], excitation[3]), (excitation[1], excitation[2])]

        if all(pair in singlet_pairs.get(_spin2spatial(pair), []) for pair in direct_pairs):
            spatial_pair = tuple(sorted(_spin2spatial(pair) for pair in direct_pairs))
        else:
            spatial_pair = tuple(sorted(_spin2spatial(pair) for pair in cross_pairs))

        if spatial_pair not in singlet_doubles:
            singlet_doubles[spatial_pair] = []
        singlet_doubles[spatial_pair].append(excitation)

    for index, excitations in enumerate(singlet_doubles.values()):
        param_excitations[f"sd{index}"] = excitations
    for index, excitations in enumerate(singlet_pairs.values()):
        param_excitations[f"ss{index}"] = excitations

    return param_excitations


def ansatz2param_excitations(ansatz_name, n_elec, n_orbs):
        if ansatz_name.upper() == "UCCSD":
            return generate_UCCSD_excitations(n_elec, n_orbs)
        elif ansatz_name.upper() == "UCCSDSINGLET":
            return generate_UCCSDSinglet_excitations(n_elec, n_orbs)
        raise ValueError(f"Unknown ansatz_type: {ansatz_name}")