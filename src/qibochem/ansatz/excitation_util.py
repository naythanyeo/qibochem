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
    Function to generate ALL possible excitations for a particular rank (including 0->O and V->V)
    First groups them by sets with combinations

    This general function find all possible excitations, including O->O and V->V

    However, this filters out all repeated or duplicated excitations, ie 0->0 for eg or 1->1 will be filtered out
    For doubles also, if (0, 1) -> (0, 2) will not be allowed (with isdisjoint)

    A second filter is also in place which only keeps one set of excitation directions. Because UCC
    ansatz fundamentally does e^(T - T+), so the conjugate is already repeated. So for eg if you have 
    a0 a1 a2+ a3+, it will give identical operations if you do a2 a3 a0+ a1+ after the conjugate has been
    subtracted (up to a sign). However, the subspace will be the same. Therefore, we keep only one copy. 
    For standardisation, we just keep the lower to higher transition, so enforce that the "holes" is 
    smaller than "particles". This will follow python lexicographic sorting and keeps a unique copy. 

    Last part will sort all the pairs by spin order, so all the alpha electrons first then beta electrons 
    This just defines the convention clearly, because before combinations only allows for unique lists, eg
    0->1 and not 1->0. But this step chooses to fix the up spin electrons first, which allows for more 
    consistent and robust sorting later on.

    Outputs: 
    Singles: ((1,), (2,)), ((1,), (3,))... 
    Doubles: ((1, 2), (3, 4)), ((1, 2), (3, 5)) ... 
    Generally: ((Excite from Sets (Holes))), (Excite to Sets (Particles)))
    """
    def sort_by_spin(pair):
        return tuple(sorted(pair, key=lambda i: i % 2))
    index_sets = list(combinations(range(n_orbs), r=rank))
    excitations = list(excitation for excitation in 
                       product(index_sets, repeat = 2) 
                       if (set(excitation[0]).isdisjoint(excitation[1]) # First filter to remove duplicate terms
                       and excitation[0] < excitation[1]))# Second filter to remove the conjugate terms
    spin_sorted_excitations = [(sort_by_spin(hole), sort_by_spin(particle)) 
                               for (hole, particle) in excitations]
    return spin_sorted_excitations


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


def filter_cross_excitations(unfiltered_list):
    """
    Used to filter out cross excitations, only retaining parallel excitations for multi excitations
    This is in place to restrict the generalised ansatz slightly based on Inquanto Convention
    These kind only allowed if OV is not restricted. 
    NOTE: Not totally sure if this restriction is needed for generalised ansatz, because it is spin 
    conserviving, and it is not the conjugate of anything. However, Inquanto seems to restrict this 
    to only parallel track excitations. Overall this filter will cut down the Generalised ansatz 
    significantly so it also helps with circuit depth. 
    
    0a 2b -> 1a 1b (X) Not allowed because alpha is 0->1 (UP) while beta is 2->1 (DOWN)
    0a 1b -> 1a 2b (√) Allowed because both alpha and beta excites upwards

    Following the previous convention that the upwards conjugate term is kept, here we restrict all
    excitations such that only upwards excitations are allowed.
    """
    filtered_list = []
    for (hole, particle) in unfiltered_list: 
        if all(hole[i] <= particle[i] for i in range(len(hole))):
            filtered_list.append((hole, particle))

    return filtered_list


def filter_spin(unfiltered_list):
    """
    Filters to keep only spin conserved transitions (m = 0) 
    Different from spin-adapt --> makes sure that each spin of every hole matches the corresponding particle term
    Should be used for all ansatz 
    """
    filtered_list = []
    for (hole, particle) in unfiltered_list: 
        if all(hole[i]%2 == particle[i]%2 for i in range(len(hole))):
            filtered_list.append((hole, particle))

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
    """
    Group excitations for UCCSDSinglet-style parameter tying.
    Each spin-orbital excitation is converted into spatial hole-particle transitions by 
    pairing holes and particles position-wise. For eg,

    (h0, h1) -> (p0, p1) transforms into
    ((h0 // 2, p0 // 2), (h1 // 2, p1 // 2))

    We use this form because sorting or grouping by (hole) (pair) ends up with lots of edge
    cases in terms of the sorting order. You cannot sort it trivially because then cases like
    12 -> 34 vs 12 -> 43 (SPATIAL)
    which represent different matched excitation channels collapse to the same state.
    
    At the same time you cannot also just not sort the (hole) (pair) form because symmetry terms like
    00 -> 23 00 -> 32 (SPATIAL)
    are the same excitations, will not group together. 

    Hence we match every excitation pair to each other, then group them that way. 
    This method accounts for all the symmetry issues from before. 
    """
    
    def spin2spatial_key(transition):
        holes, particles= transition
        spatial_lines = [(h // 2, p // 2)
                         for h, p in zip(holes, particles)]
        return tuple(sorted(spatial_lines))

    groups = {}
    for transition in unfiltered_list:
        key = spin2spatial_key(transition)
        groups.setdefault(key, []).append(transition)

    # Sort the groups itself before returning by holes first then particles
    # This sorting is done so that later on sorting with ucc uses first term, so its consistent
    sorted_groups = [sorted(group, key = lambda excitation: (excitation[0], 
                                                             excitation[1]))
                     for group in list(groups.values())]
    return sorted_groups