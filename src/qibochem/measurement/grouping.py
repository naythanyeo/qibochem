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


def group_bitmask_terms(terms, largest_first=True):
    """
    Greedily pack bitmask Pauli terms into QWC groups.

    Args:
        terms: List of ``(x_mask, y_mask, z_mask)`` Pauli terms.
        largest_first: If ``True``, pack higher-weight terms first. If
            ``False``, preserve first-seen unique term order. Default use largest first for 
            better and more reliable grouping

    Returns:
        ``(groups, term2group_map)``
        ``groups`` is a list of QWC group basis masks  
        ``term2group_map`` maps each unique term to its group index.
    """
    unique_terms = []
    seen_terms = set()

    for term in terms:
        term = tuple(int(mask) for mask in term)
        if term == (0, 0, 0) or term in seen_terms:
            continue
        unique_terms.append(term)
        seen_terms.add(term)

    # Sort by all the largest terms first
    if largest_first:
        unique_terms.sort(
            key=lambda term: ((term[0] | term[1] | term[2]).bit_count(), term), reverse = True
        )

    groups = []
    term2group_map = {}

    for term in unique_terms:
        for group_index, group_mask in enumerate(groups):
            if _qwc_compatible(group_mask, term):
                groups[group_index] = _update_group_mask(group_mask, term)
                term2group_map[term] = group_index
                break
        else:
            term2group_map[term] = len(groups)
            groups.append(term)

    return groups, term2group_map
