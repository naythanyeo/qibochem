"""
Helper functions for obtaining and transforming the molecular Hamiltonian
"""

from functools import reduce
import sympy as sp
import openfermion
from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian


def _fermionic_hamiltonian(oei, tei, constant):
    """
    Build molecular Hamiltonian as an InteractionOperator using the 1-/2- electron integrals

    Args:
        oei: 1-electron integrals in the MO basis
        tei: 2-electron integrals in 2ndQ notation and MO basis
        constant: Nuclear-nuclear repulsion, and inactive Fock energy if HF embedding used

    Returns:
        Molecular Hamiltonian as an InteractionOperator
    """
    oei_so, tei_so = openfermion.ops.representations.get_tensors_from_integrals(oei, tei)
    # tei_so already multiplied by 0.5, no need to include in InteractionOperator
    return openfermion.InteractionOperator(constant, oei_so, tei_so)


def _qubit_hamiltonian(fermion_hamiltonian, ferm_qubit_map):
    """
    Converts the molecular Hamiltonian to a QubitOperator

    Args:
        fermion_hamiltonian: Molecular Hamiltonian as a InteractionOperator/FermionOperator
        ferm_qubit_map: Which Fermion->Qubit mapping to use

    Returns:
        qubit_operator : Molecular Hamiltonian as a QubitOperator
    """
    # Map the fermionic molecular Hamiltonian to a QubitHamiltonian
    if ferm_qubit_map == "jw":
        q_hamiltonian = openfermion.jordan_wigner(fermion_hamiltonian)
    elif ferm_qubit_map == "bk":
        q_hamiltonian = openfermion.bravyi_kitaev(fermion_hamiltonian)
    else:
        raise KeyError("Unknown fermion->qubit mapping!")
    q_hamiltonian.compress()  # Remove terms with v. small coefficients
    return q_hamiltonian


def _qubit_to_symbolic_hamiltonian(q_hamiltonian, n_qubits=None):
    """
    Converts a OpenFermion QubitOperator to a Qibo SymbolicHamiltonian

    Args:
        q_hamiltonian: QubitOperator

    Returns:
        qibo.hamiltonians.SymbolicHamiltonian
    """
    # Cache Qibo Pauli symbols so repeated X/Y/Z operators on the same qubit
    # are not rebuilt for every Pauli string.
    # Eg X0Y0, X0Y1 --> X(0) symbolic is only being built 1
    pauli_symbols = {
        (pauli_op, qubit): getattr(symbols, pauli_op)(qubit)
        for pauli_string in q_hamiltonian.terms
        for qubit, pauli_op in pauli_string
    }

    # Build each Pauli string as one SymPy product, then give all terms to
    # SymPy at once. This avoids slow repeated expression canonicalization
    # from Python's incremental sum(...).
    symbolic_terms = [
        sp.Mul(
            coeff,
            *(pauli_symbols[(pauli_op, qubit)] for qubit, pauli_op in pauli_string),
        )
        for pauli_string, coeff in q_hamiltonian.terms.items()
    ]

    symbolic_ham = sp.Add(*symbolic_terms)
    return SymbolicHamiltonian(symbolic_ham, nqubits=n_qubits)
