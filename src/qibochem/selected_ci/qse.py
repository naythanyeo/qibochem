"""Quantum Subspace Expansion (QSE)."""

from __future__ import annotations

import numpy as np
import openfermion

from qibochem.driver.hamiltonian import (
    _qubit_hamiltonian,
    _qubit_to_symbolic_hamiltonian,
)
from qibochem.selected_ci.utils import assemble_matrix

"""
GENERAL EXCITATION OPERATORS
Here we define a few QSE Excitation Operator Generators
These generators take in a mol object and return a list of excitation operators 
Mol object is input because sometimes number of electrons will be used for some generators 
"""

def generate_general_singles(mol) -> list[openfermion.FermionOperator]:
        """Generate one-body excitation operators + the identity."""
        n_active_orbs = mol.n_active_orbs if mol.n_active_orbs is not None else mol.nso
        operators = [
            openfermion.FermionOperator(f"{2*_i}^ {2*_j}") + openfermion.FermionOperator(f"{2*_i+1}^ {2*_j+1}")
            for _i in range(n_active_orbs // 2)
            for _j in range(n_active_orbs // 2)
        ]
        # operators = [openfermion.FermionOperator("")]
        # for i in range(nso):
        #     for j in range(nso):
        #         if self.config.conserve_spin and (i % 2) != (j % 2):
        #             continue
        #         operators.append(openfermion.FermionOperator(f"{i}^ {j}"))
        return operators


"""
QSE COMPUTABLE OBJECT 
This object manages the QSE protocol for building the matrix 
Main changes from before is that now the protocols used for various strategies are 
implemented into protocol.py, so this computable object is more focused on managing
the QSE-specific logic only. Abit easier to read and control imo. 

Initialised with molecule only, configuration is removed 

Things like n_shots will be initialised with PROTOCOL, so the logic is kept separate 
OUTPUT: OBSERVABLE (H) and OVERLAP (S) matrices only 

For future implementation? 
Treats the QSE computable as a generic projected observable computable 
By default, the observable is hamiltonian (FOR NOW), but subsequently if you have a 
different observable like S^2, can initialise QSE observable with other observables instead 
Not too hard to implement but need a general observable class to manages this
"""
class QSE_Computable:
    def __init__(self, molecule, excitation_generator=None, ferm_qubit_map="jw"):
        """
        Quantum Subspace Expansion (QSE) manager.
        Args:
            molecule: `qibochem.driver.Molecule` instance.
            excitation_generator: Function that generates a list of excitaiton operators
        """
        if ferm_qubit_map not in ("jw", "bk"):
            raise ValueError("ferm_qubit_map must be 'jw' or 'bk'.")

        self.molecule = molecule
        self.excitation_generator = excitation_generator or generate_general_singles
        self.ferm_qubit_map = ferm_qubit_map
        self.operators = None  # List of fermion excitation operators
        # h and s data store the operator strings and coefficients 
        self.s_data = None
        self.h_data = None
    

    def collate_hs_matrix_info(self):
        """Constructs the Hamiltonian and terms for each of the S/H matrix elements"""
        if self.operators is None:
            self.operators = self.excitation_generator(self.molecule)

        n_active_orbs = self.molecule.n_active_orbs if self.molecule.n_active_orbs is not None else self.molecule.nso
        dim = len(self.operators)

        # Store information about H/S in two dictionaries first
        self.s_data = {(_i, _j): dict() for _i in range(dim) for _j in range(dim) if _i <= _j}
        self.h_data = {(_i, _j): dict() for _i in range(dim) for _j in range(dim) if _i <= _j}

        # Populate the Hamiltonians corresponding to each matrix element in S/H
        for mat_data, operator in zip((self.s_data, self.h_data), (1.0, self.molecule.hamiltonian("f"))):
            for element in mat_data.keys():
                mat_data[element] = _qubit_to_symbolic_hamiltonian(
                    _qubit_hamiltonian(
                        openfermion.hermitian_conjugated(self.operators[element[0]])
                        * operator
                        * self.operators[element[1]],
                        self.ferm_qubit_map,
                    ),
                    n_active_orbs,
                )
        # NOTE: Previously stored constant and terms separately but protocol already does this during measurement 
        # For future implementaiton: make this molecule_hamiltonian a generic observable 
        # Eg now it calls mol.hamiltonian("f"). If other operators implemented this can change to 
        # calling mol.spin_projection("f") or mol.spin_squared("f") etc 
        # Depending on observable calss, might have more functions to call this projection also

    def run_qse(self, circuit, protocol) -> tuple(np.ndarray, np.ndarray):
        """
        Run the QSE protocol using the given circuit.

        Args:
            circuit: Qibo circuit used to prepare the reference state (e.g. from VQE).
            statevector: Optional exact statevector. If None and n_shots is None,
                         it is automatically computed via circuit().state().

        Returns:
            H and S matrices as numpy arrays
        """
        # First define the excitation operators 
        self.operators = self.excitation_generator(self.molecule)

        # Update the H and S observables  
        self.collate_hs_matrix_info()
        H_values = protocol.evaluate(circuit, self.h_data)
        S_values = protocol.evaluate(circuit, self.s_data)

        H = assemble_matrix(H_values)
        S = assemble_matrix(S_values)

        return H, S
