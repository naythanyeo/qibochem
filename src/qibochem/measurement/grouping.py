"""Bitmask-based grouping of qubitwise-commuting Pauli terms."""


def _qwc_compatible(group_mask, term_mask):
    """Check whether adding ``term_mask`` creates a same-qubit basis conflict."""
    group_x, group_y, group_z = group_mask
    term_x, term_y, term_z = term_mask

    return not (
        group_x & (term_y | term_z)
        or group_y & (term_x | term_z)
        or group_z & (term_x | term_y)
    )


def _update_group_mask(group_mask, term_mask):
    """Merge one compatible term into a QWC group basis mask."""
    group_x, group_y, group_z = group_mask
    term_x, term_y, term_z = term_mask
    return group_x | term_x, group_y | term_y, group_z | term_z



def group_bitmask_terms(unique_terms, largest_first=True):
    """
    Greedily pack bitmask Pauli terms into QWC groups.
    Returns:
        dict:
            {
                group_mask: [term_1, term_2, ...],
                ...
            }
        where group_mask is the final measurement basis mask for the group.

    Here we first store a conflict map called basis_index, a list of 3 dictionaries 
    basis_index = x_basis, y_basis, z_basis 
    x_basis = {qubit_n: groups_with_x}
    whereby both qubit_n and groups_with_x are binary form 

    Both of them are represented as binary form so that union is faster 
    for eg, x_basis = {0001: 0101, 0010: 1000} means that
    groups 0 and 2 have qubit 0 occupied with x, and group 3 has qubit 1 occupied with x 

    So to combine bitsets to find all the groups with xyz we can do OR operations 
    """

    # First sort the terms if there are 
    if largest_first:
        unique_terms.sort(
            key=lambda term: (term[0] | term[1] | term[2]).bit_count(),
            reverse=True,
        )
    def _iter_bits(bit_mask):
        while bit_mask:
            bit = bit_mask & -bit_mask # Select the first bit 
            yield bit
            bit_mask ^= bit # XOR to remove the latest bit
    
    def _update_basis_index(group_bit):
        for bit in _iter_bits(x_mask):
            basis_index[0][bit] = basis_index[0].get(bit, 0) | group_bit
        for bit in _iter_bits(y_mask):
            basis_index[1][bit] = basis_index[1].get(bit, 0) | group_bit
        for bit in _iter_bits(z_mask):
            basis_index[2][bit] = basis_index[2].get(bit, 0) | group_bit

    # Use list first as its mutable
    basis_index = [{}, {}, {}]
    groups = []
    all_groups_bit = 0
    for term in unique_terms:
        # Term is a bitmask (x_mask, y_mask, z_mask)
        # First check if there are conflicts
        x_mask, y_mask, z_mask = term
        conflict_x, conflict_y, conflict_z = 0, 0, 0
        for bit in _iter_bits(x_mask):
            groups_with_y = basis_index[1].get(bit, 0)
            groups_with_z = basis_index[2].get(bit, 0)
            conflict_x |= groups_with_y | groups_with_z
        for bit in _iter_bits(y_mask):
            groups_with_x = basis_index[0].get(bit, 0)
            groups_with_z = basis_index[2].get(bit, 0)
            conflict_y |= groups_with_x | groups_with_z
        for bit in _iter_bits(z_mask):
            groups_with_x = basis_index[0].get(bit, 0)
            groups_with_y = basis_index[1].get(bit, 0)
            conflict_z |= groups_with_x | groups_with_y
        conflict_groups = conflict_x | conflict_y | conflict_z
        
        """
        all_group_bits is a binary like 1111111
        ~conflict_groups will give the remaining group index that are not conflicting 
        doing valid_groups & -valid_groups gives the first valid group bit 
        valid_groups is something like 1010100 for eg, then first_valid_group_bit will be 100
        bit_length will give 3, so subtract 1 because 100 would represent qubit 2 
        """
        valid_groups = all_groups_bit & ~conflict_groups
        
        # IF no group
        if valid_groups == 0:
            # Add on to the all_groups_bits
            group_bit = 1 << len(groups)
            all_groups_bit |= group_bit
            groups.append([term, [term]])
            # Update the cache map
            _update_basis_index(group_bit=group_bit)

        # If got groups
        else:
            first_valid_group_bit = valid_groups & -valid_groups
            first_valid_group_index = first_valid_group_bit.bit_length() - 1

            group = groups[first_valid_group_index]
            group_mask, group_terms = group
            groups[first_valid_group_index][0] = _update_group_mask(group_mask, term)
            group_terms.append(term)

            _update_basis_index(group_bit=first_valid_group_bit)

    return {group_mask: group_terms 
            for group_mask, group_terms in groups}



        


