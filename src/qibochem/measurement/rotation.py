"""
Rotation functions that does rotation onto states for measurement 
Before, this was done by adding circuit gates and re-evaluating them
However, more efficiently, we just use bitmasks to directly rotate the final_state 

Similar to expectations.py, but before we were applying X and Y gates which were easy to do 
just by having XOR + AND functions, and recording phase shifts
Here we need to implement instead H and S gates to rotate the original final_state 
into the Z basis entirely. 

The input of this will be a group_term, which has the same format as before, as well
as the final_state of a circuit, which will be rotated. 
So for the helper functions here will rotate 1 group term 

final_state: vector of complex coefficient amplitudes
group_term: (bitmask_x, bitmask_y, bitmask_z)

Output: Vector with rotated coefficient amplitudes 
"""

import numpy as np


def _iter_mask_bits(mask):
    """Yield one active qubit bit at a time: 0b101 -> 0b001, 0b100."""
    while mask:
        bit = mask & -mask # -mask flips every bit and adds 1. AND of both yields the lowest bit
        yield bit # Returns but remembers position in loop
        mask ^= bit # XOR to remove the bit


def _apply_sdg_mask(state, sdg_mask):
    """
    Apply Sdg to every qubit in sdg_mask.

    Sdg|0> = |0>
    Sdg|1> = -i|1>

    For each basis index, the total phase is (-i) raised to the number of
    occupied qubits selected by sdg_mask.
    """
    if sdg_mask == 0:
        return state

    basis_states = np.arange(state.size)
    occupied_counts = np.bitwise_count(basis_states & sdg_mask).astype(int)
    return state * ((-1j) ** occupied_counts)


def _apply_h_single_qubit(state, bit):
    """
    Applying H is problematic because it gives a superposition state for every H 
    H|0> = (|0> + |1>) / sqrt(2)
    H|1> = (|0> - |1>) / sqrt(2)

    For each pair in a "statevector slice":
        |...0...> amplitude a0
        |...1...> amplitude a1
    Those terms are consecutive numbers in final_state vector 

    The superposition of acting H on both consecutive bitstrings gives
    H|...0...> + H|...1...> = a0(|...0...> + |...1...> ) / sqrt(2) + 
                              a1(|...0...> - |...1...> ) / sqrt(2) 
    So H gives:
        new0 = (a0 + a1) / sqrt(2)
        new1 = (a0 - a1) / sqrt(2)
    """
    basis_states = np.arange(state.size)
    zero_indices = basis_states[(basis_states & bit) == 0]
    one_indices = zero_indices | bit
    # Get the zero and one amplitudes a0 and a1
    zero_amplitudes = state[zero_indices].copy()
    one_amplitudes = state[one_indices].copy()
    # Make copy so original state is not disturbed
    rotated = state.copy()
    # Vectorised form of the new amplitudes addition with numpy settling all the element operations
    rotated[zero_indices] = (zero_amplitudes + one_amplitudes) / np.sqrt(2) 
    rotated[one_indices] = (zero_amplitudes - one_amplitudes) / np.sqrt(2)
    return rotated


def _apply_h_mask(state, h_mask):
    """Apply H to every qubit selected by h_mask."""
    rotated = state
    # First split up the h_mask by bitstring then apply H gate one by one 
    for bit in _iter_mask_bits(h_mask):
        rotated = _apply_h_single_qubit(rotated, bit)
    return rotated


def rotate_basis(final_state, group_term):
    """
    Rotate final_state into the computational basis for measuring group_term.

    X terms need H.
    Y terms need Sdg then H.
    Z terms need no rotation.

    Args:
        final_state: 1D complex statevector.
        group_term: ``(x_mask, y_mask, z_mask)`` bitmask group.

    Returns:
        Rotated 1D complex statevector with the same shape as final_state.
    """
    x_mask, y_mask, _ = group_term

    sdg_mask = y_mask # Y gates need the S dagger gates
    h_mask = x_mask | y_mask # Both X and Y will need H gate 

    rotated = _apply_sdg_mask(final_state, sdg_mask)
    rotated = _apply_h_mask(rotated, h_mask)
    return rotated
