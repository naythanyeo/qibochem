import numpy as np
from itertools import product, combinations
"""
EXCITATION GENERATION FUNCTIONS 
These set of excitation generators enumerate all possibilities so it is easier to build generalised ansatz too 
First layer generates ALL excitations 
Subsequent defines filter and grouping functions that can be used to restrict ansatz 
In general an Ansatz will then be constructed from a combination of these functions 
"""

# Generate All Excitations
def generate_excitations(rank, n_orbs): 
    """
    Function to generate ALL possible excitations for a particular rank
    First groups them by sets with combinations to prevent overlap 
    Then find all possible excitations, including O->V, O->O, V->V, V->O
    Outputs: 
    Singles: ((1,), (2,)), ((1,), (3,))... 
    Doubles: ((1, 2), (3, 4)), ((1, 2), (3, 5)) ... 
    Generally: ((Excite from Sets), (Excite to Sets))
    """
    index_sets = list(combinations(range(n_orbs), r=rank))
    excitations = list(product(index_sets, repeat = 2))
    return excitations

def filter_OV_transition(unfiltered_list, n_elec, n_orbs):
    """
    Filters to keep only O->V transitions
    Used for non generalised ansatz
    """
    occupied = set(range(n_elec))
    virtual = set(range(n_elec, n_orbs))
    filtered_list = [transition for transition in unfiltered_list 
                     if set(transition[0]).issubset(occupied) 
                     and set(transition[1]).issubset(virtual)]
    return filtered_list

def filter_spin(unfiltered_list):
    """
    Filters to keep only spin conserved transitions (m = 0) 
    Different from spin-adapt --> just counts total spin of destroyed and created
    Used for most ansatz 
    """
    def sum_spin(index_list):
        return sum(i % 2 for i in index_list)
    filtered_list = [transition for transition in unfiltered_list 
                     if sum_spin(transition[0]) == sum_spin(transition[1])]
    return filtered_list
    
def filter_paired(unfiltered_list):
    """
    Filters to only keep paired doubles, ie every doubles term in transition must be
    from the same spatial orbital (1a 1b) ok, (1a 2b) reject 
    Only make sense for EVEN ranks here because ODD ranks will just kill everything
    """
    def is_made_of_pairs(index_tuple):
        spatial_to_spins = {}
        for i in index_tuple:
            spatial = i // 2
            spin = i % 2
            spatial_to_spins.setdefault(spatial, set()).add(spin)
        return all(spins == {0, 1} for spins in spatial_to_spins.values())

    filtered_list = [transition for transition in unfiltered_list
                     if is_made_of_pairs(transition[0])
                     and is_made_of_pairs(transition[1])]
    return filtered_list

def group_spin_adapt(unfiltered_list):
    def spin2spatial(index_lists):
        from_index = [i // 2 for i in index_lists[0]]
        to_index = [i // 2 for i in index_lists[1]]
        return [from_index, to_index]
    unique_keys = set([spin2spatial(transition) for transition in unfiltered_list])
    sorted_groups = [[transition for transition in unfiltered_list
                      if spin2spatial(transition) == key] 
                      for key in unique_keys]
    return sorted_groups

def group_spin_adapt(unfiltered_list):
    """
    Group excitations by spatial hole pattern and spatial particle pattern.
    Used for UCCSDSinglet-style parameter tying.
    """
    def spin2spatial(transition):
        holes, excited = transition
        spatial_holes = tuple(sorted(i // 2 for i in holes))
        spatial_excited = tuple(sorted(a // 2 for a in excited))
        return spatial_holes, spatial_excited

    groups = {}
    for transition in unfiltered_list:
        key = spin2spatial(transition)
        groups.setdefault(key, []).append(transition)

    return list(groups.values())

def flatten_excitation(excitation):
    """
    Flattens excitations from ((hole), (excitation)), eg ((0, 1), (2, 3)) to (hole, excitation) (0, 1, 2, 3)
    to match the input parameters of UCC_Circuit
    """
    holes, particles = excitation
    return tuple(holes) + tuple(particles)

def sort_excitations(excitations_list):
    """
    Previous sorting method used chemical rules to sort excitations, but not too practical
    for many excitations. New sorting function sorts a list of grouped excitations in lexicographic order 
    because it is more practical. Sorting is just to ensure engineering consistency.

    INPUT:
        A list of grouped excitations:
        [[(0, 1, 2, 3), (0, 1, 4, 5)], [(0, 1, 5, 6), (0, 1, 7, 8)], ... ]
    OUTPUT:
        Same list, but sorted by lexicographic token order 
    * Groups are first sorted lexicographically so this process will be deterministic 
    """
    # Flattens the grouped excitations
    def group_token(group):
        canonical_group = sorted(group)
        token = []
        for excitation in canonical_group:
            token.extend(excitation)
        return tuple(token)
    return sorted(excitations_list, key=group_token)

"""
Other utility functions for ansatzes
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
    # Guards added against denominator and amplitude to catch nan or inf values
    denominator = sum(orbital_energies[mo_orbitals[:2]]) - sum(orbital_energies[mo_orbitals[2:]])
    if abs(denominator) < 1e-12:
        return 0.0
    amplitude = numerator / denominator
    if np.isnan(amplitude):
        return 0.0
    return amplitude