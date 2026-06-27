"""Fast statevector evolution for UCC Pauli rotations."""

import numpy as np


def get_hf_bit_state(n_qubits, n_elec):
    """Return the Hartree-Fock reference statevector."""
    basis_index = 0
    for qubit in range(n_elec):
        basis_index |= 1 << (n_qubits - 1 - qubit)

    state = np.zeros(2**n_qubits, dtype=complex)
    state[basis_index] = 1.0
    return state


def get_pauli_action(bitmask, basis_states):
    """Return the cached index shift and phase for one Pauli bitmask."""
    x_mask, y_mask, z_mask = bitmask
    flipped_states = basis_states ^ (x_mask | y_mask)

    z_sign = (-1) ** np.bitwise_count(basis_states & z_mask).astype(np.int8)
    y_sign = (-1) ** np.bitwise_count(basis_states & y_mask).astype(np.int8)
    y_phase = (-1j) ** y_mask.bit_count()
    phase_shift = (z_sign * y_sign * y_phase).astype(np.complex64)

    # If the basis state is small then this saves RAM by storing as int32
    # For practical purposes up to 8e8o will fit in int32
    # Phase shift is complex so remain as complex64
    if basis_states.size <= np.iinfo(np.uint32).max:
        flipped_states = flipped_states.astype(np.uint32, copy=False)

    return flipped_states, phase_shift


def apply_ucc_rotations(state, theta_vector, rotations):
    """Apply precomputed UCC Pauli rotations to a statevector."""
    evolved_state = np.asarray(state, dtype=complex).copy()

    for param_index, flipped_states, phase_shift, angle_coeff in rotations:
        angle = angle_coeff * theta_vector[param_index]
        if angle == 0:
            continue

        pauli_state = evolved_state[flipped_states]
        evolved_state = (
            np.cos(angle) * evolved_state  + 
            1j * np.sin(angle) * pauli_state * phase_shift
        )

    return evolved_state
