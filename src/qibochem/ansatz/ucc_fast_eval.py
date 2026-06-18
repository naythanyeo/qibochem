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


def apply_ucc_rotations(state, theta_vector, rotations):
    """Apply precomputed UCC Pauli rotations to a statevector."""
    evolved_state = np.asarray(state, dtype=complex).copy()
    basis_states = np.arange(evolved_state.size)

    for param_index, bitmask, angle_coeff in rotations:
        angle = angle_coeff * theta_vector[param_index]
        if angle == 0:
            continue

        x_mask, y_mask, z_mask = bitmask
        flip_mask = x_mask | y_mask
        flipped_states = basis_states ^ flip_mask

        z_sign = (-1) ** np.bitwise_count(basis_states & z_mask).astype(int)
        y_sign = (-1) ** np.bitwise_count(basis_states & y_mask).astype(int)
        y_phase = (-1j) ** y_mask.bit_count()
        pauli_state = evolved_state[flipped_states] * z_sign * y_sign * y_phase

        evolved_state = np.cos(angle) * evolved_state + 1j * np.sin(angle) * pauli_state

    return evolved_state
