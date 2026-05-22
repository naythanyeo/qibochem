"""GENERAL HELPER FUNCTIONS"""
import numpy as np
from numpy import linalg as la


def assemble_matrix(values, dim=None):
        """Assemble a dense Hermitian matrix from upper-triangular values."""
        if not values and dim is None:
                raise ValueError("dim must be specified when values is empty.")

        if dim is None:
                dim = max(max(i, j) for i, j in values) + 1

        matrix = np.zeros((dim, dim), dtype=complex)
        for (i, j), value in values.items():
                if i >= dim or j >= dim:
                        raise ValueError("Matrix index is outside the requested dimension.")
                matrix[i, j] = value
                if i != j:
                        matrix[j, i] = np.conj(value)
        return matrix


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




"""FUNCTINO FOR BOOTSTRAP SAMPLING PROTOCOL HERE
READ IN H AND S SAMPLES and a LIST OF SAMPLE VALUES
OUTPUT BOOTSTRAPPED MATRICES"""
