"""GENERAL HELPER FUNCTIONS"""
from collections.abc import Mapping

import numpy as np
from numpy import linalg as la


def assemble_hs(values):
    if not values:
        raise ValueError("Cannot assemble matrices from empty values.")

    dim = max(max(i, j) for _, i, j in values) + 1
    H = np.zeros((dim, dim), dtype=complex)
    S = np.zeros((dim, dim), dtype=complex)

    for (label, i, j), value in values.items():
        if label == "H":
            matrix = H
        elif label == "S":
            matrix = S
        else:
            raise ValueError("QSE matrix label must be 'H' or 'S'.")

        matrix[i, j] = value
        if i != j:
            matrix[j, i] = np.conj(value)

    return H, S


def assemble_matrix_outputs(values):
    if not isinstance(values, Mapping):
        raise TypeError("values must be a mapping.")

    if not values:
        raise ValueError("Cannot assemble matrices from empty values.")

    first_value = list(values.values())[0]
    if not isinstance(first_value, Mapping):
        return assemble_hs(values)

    H = {}
    S = {}
    for sample_key, sample_values in values.items():
        H[sample_key], S[sample_key] = assemble_hs(sample_values)
    return H, S


def solve_generalised_eigeneqn(S, H, threshold=1e-6):
        """
        Performs a single run of the QSE protocol
        """
        # Solve generalized eigenvalue problem: H_LR C = S_LR C E
        # Since S_LR can still be ill-conditioned, we diagonalize S_LR first
        s_evals, s_evecs = la.eigh(S)

        # Keep only eigenvectors of S where eigenvalue is > threshold
        valid_idx = s_evals > threshold
        if not np.any(valid_idx):
            # TODO: Make error message nicer
            raise ValueError(
                "No valid QSE subspace found after overlap matrix diagonalization. "
                "The excitation_threshold or eigenvalue_threshold might be too strict, "
                "or the reference state provides no valid excitations."
            )

        s_evals = s_evals[valid_idx]
        s_evecs = s_evecs[:, valid_idx]  # Projection matrix P

        # S^{-1/2}
        s_inv_half = s_evecs @ np.diag(1.0 / np.sqrt(s_evals))

        # Form the orthogonalized Hamiltonian H' = S^{-1/2}^T H S^{-1/2}
        h_prime = s_inv_half.T.conj() @ H @ s_inv_half

        eigenvalues, c_prime = la.eigh(h_prime)

        # Transform back to original basis C = S^{-1/2} C'
        eigenvectors = s_inv_half @ c_prime

        return eigenvalues, eigenvectors, len(s_evals)
