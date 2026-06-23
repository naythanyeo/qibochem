"""
Quantum Subspace Expansion (QSE).
New reformatted to just use operator strings instead of symbolic hamiltonian
Old sympy objects actually have a ton of overhead which is very slow
Just build as openfermion objects
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product

import numpy as np
from pathlib import Path
import openfermion

from qibochem.driver.observables import (
    BitmaskObservable,
    multiply_bitmask_observables,
    qubit_operator2observable,
)
from qibochem.selected_ci.utils import assemble_matrix_outputs, pickle2dict, dict2pickle

"""
GENERAL EXCITATION OPERATORS
Here we define a few QSE Excitation Operator Generators
These generators take in an excitation_params dictionary and return a list of excitation operators.
"""
def generate_general_singles(excitation_params: dict) -> list[openfermion.FermionOperator]:
    """Generate all spin-orbital one-body operators a_p^ a_q.
    Here general singles contains unrestricted excitations"""
    n_spin_orbs = excitation_params["n_orbs"]
    operators = [openfermion.FermionOperator(f"{_i}^ {_j}")
                 for _i in range(n_spin_orbs)
                 for _j in range(n_spin_orbs)]
    return operators

def generate_singlet_singles(excitation_params: dict) -> list[openfermion.FermionOperator]:
    """Generate spin-adapt one-body operators
    Loops through spatial orbitals instead to pair the terms
    INCLUDES the number operator here"""
    n_active_orbs = excitation_params["n_orbs"]
    operators = [openfermion.FermionOperator(f"{2*_i}^ {2*_j}") + openfermion.FermionOperator(f"{2*_i+1}^ {2*_j+1}")
            for _i in range(n_active_orbs // 2)
            for _j in range(n_active_orbs // 2)]
    return operators

def generate_triplet_singles(excitation_params: dict) -> list[openfermion.FermionOperator]:
    """Generate one-body triplet excitation operators for a chosen spin projection.
    Similar to singlet singles, but the triplets sign is reversed for m=0
    For m=+-1, the excitations dont need to be paired"""
    ms = excitation_params["spin_projection"]
    if ms not in (0, 1, -1, "all"):
        raise ValueError("ms must be 0, 1, -1, or 'all'.")

    n_active_orbs = excitation_params["n_orbs"]
    n_spatial_orbs = n_active_orbs // 2

    def triplet_operator(_i, _j, _ms):
        if _ms == 0: # m=0 case, flip the sign of operator pairs
            return (openfermion.FermionOperator(f"{2 * _i}^ {2 * _j}") - 
                    openfermion.FermionOperator(f"{2 * _i + 1}^ {2 * _j + 1}")) / np.sqrt(2.0)
        # The case for m not 0 
        return openfermion.FermionOperator(f"{2 * _i + int(_ms < 0)}^ {2 * _j + int(_ms > 0)}")

    if ms == "all":
        return [triplet_operator(_i, _j, _ms)
                for _i in range(n_spatial_orbs)
                for _j in range(n_spatial_orbs)
                for _ms in (0, 1, -1)]

    return [triplet_operator(_i, _j, ms)
            for _i in range(n_spatial_orbs)
            for _j in range(n_spatial_orbs)]


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
@dataclass
class QSE_Computable:
    molecule: object
    # Excitations
    excitation_generator: Callable
    spin_projection: int | str = 0 # Can be 0 -1 1 or "all"
    excitation_params: dict | None = field(default=None, init=False)
    operators: list | None = field(default=None, init=False)
    # Params
    ferm_qubit_map: str = "jw"
    map_threshold: float = 1e-12
    # Caching
    s_cache_path: str | None = None
    h_cache_path: str | None = None
    # Storing observables
    s_data: dict | None = field(default=None, init=False)
    h_data: dict | None = field(default=None, init=False)
    

    def __post_init__(self):
        """Validate QSE inputs and define the excitation generator parameters."""
        if self.excitation_generator is None:
            raise ValueError("excitation_generator must be specified.")

        if self.ferm_qubit_map not in ("jw", "bk"):
            raise ValueError("ferm_qubit_map must be 'jw' or 'bk'.")

        n_elec = self.molecule.n_active_e if self.molecule.n_active_e is not None else self.molecule.nelec
        n_orbs = self.molecule.n_active_orbs if self.molecule.n_active_orbs is not None else self.molecule.nso

        self.excitation_params = {
            "n_elec": n_elec,
            "n_orbs": n_orbs,
            "spin_projection": self.spin_projection,
        }
    


    def _map_ferm2bitmask(self, projected):
        """
        Function to map projected terms to observable class based on mapping choice
        After fermionic to qubit mapping is done, it is compressed, then mapped again
        to the bitmask form with the helper function
        """
        if self.ferm_qubit_map == "jw":
            q_op = openfermion.jordan_wigner(projected)
        elif self.ferm_qubit_map == "bk":
            q_op = openfermion.bravyi_kitaev(projected)
        else:
            raise ValueError("ferm_qubit_map must be 'jw' or 'bk'.")
        
        # Use a threshold smaller than main threhsold for map construction because the map terms can be
        # summed up across hpq and hpqrs terms, so we keep terms 2 orders of magnitude smaller to be safe
        q_op.compress(abs_tol=self.map_threshold * 1e-2)
        # Last stage map the qubit operator to a bitmask observable
        return qubit_operator2observable(q_op, n_qubits=self.excitation_params["n_orbs"])

    def _get_Ei_Ej(self, i, j):
        ferm_Ei_dag = openfermion.hermitian_conjugated(self.operators[i])
        ferm_Ej = self.operators[j]
        bitmask_Ei_dag = self._map_ferm2bitmask(ferm_Ei_dag)
        bitmask_Ej = self._map_ferm2bitmask(ferm_Ej)
        return bitmask_Ei_dag, bitmask_Ej

    def _get_S_ij(self, i, j):
        """
        Get a term of Sij
        """
        bitmask_Ei_dag, bitmask_Ej = self._get_Ei_Ej(i, j)
        projected_S = multiply_bitmask_observables(bitmask_Ei_dag, bitmask_Ej, 
                                                   threshold=self.map_threshold)
        return projected_S

    def _get_H_ij_direct(self, i, j, ferm_hamiltonian):
        """
        Get a term of Hij directly from the hamilltonian
        """
        bitmask_Ei, bitmask_Ej = self._get_Ei_Ej(i, j)
        bitmask_ham = self._map_ferm2bitmask(ferm_hamiltonian)
        left_prod = multiply_bitmask_observables(bitmask_Ei, bitmask_ham, 
                                                 threshold=self.map_threshold * 1e-2) # intermediate threshold
        projected_ham = multiply_bitmask_observables(left_prod, bitmask_Ej, 
                                                            threshold=self.map_threshold)
        return projected_ham

    def _prepare_s_data(self):
        """
        Load S Data from cache path is its available
        If not build S data and optionally cache it
        """
        # First check if S cache map exist
        if self.s_cache_path is not None:
            path = Path(self.s_cache_path)
            if path.exists():
                self.s_data = pickle2dict(self.s_cache_path)
                return
            
        # If not build in the s_matrix
        dim = len(self.operators)
        for i, j in product(range(dim), repeat=2):
            if i > j: 
                continue # Only build upper triangle
            self.s_data[(i, j)] = self._get_S_ij(i, j)

        # Optional caching if the cache path is provided
        if self.s_cache_path is not None:
            dict2pickle(self.s_data, self.s_cache_path)


    def _prepare_h_data(self):
        """
        Load H data from cache path if its available
        If not build H data from fermionic hamiltonian and optionally cache it
        """
        # First check if H cache map exist
        if self.h_cache_path is not None:
            path = Path(self.h_cache_path)
            if path.exists():
                self.h_data = pickle2dict(self.h_cache_path)
                return
            
        # If not build the h_matrix
        dim = len(self.operators)
        ferm_hamiltonian = self.molecule.hamiltonian("f")
        for i, j in product(range(dim), repeat=2):
            if i > j:
                continue # Only build upper triangular
            H_ij = self._get_H_ij_direct(i, j, ferm_hamiltonian)
            self.h_data[(i, j)] = H_ij

        # Optional caching if the cache path is provided
        if self.h_cache_path is not None:
            dict2pickle(self.h_data, self.h_cache_path)


    def run_qse(self, circuit, protocol) -> tuple[np.ndarray | dict, np.ndarray | dict]:
        """
        Run the QSE protocol using the given circuit.

        Args:
            circuit: Qibo circuit used to prepare the reference state (e.g. from VQE).
            statevector: Optional exact statevector. If None and n_shots is None,
                         it is automatically computed via circuit().state().

        Returns:
            H and S matrices as arrays, or dictionaries of sampled matrices.
        """
        # First define the excitation operators.
        self.operators = self.excitation_generator(self.excitation_params)

        # Update the H and S observables if they were not done before
        if self.h_data is None:
            self.h_data = defaultdict()
            self._prepare_h_data()
        if self.s_data is None:
            self.s_data = defaultdict()
            self._prepare_s_data()

        # Group the H and S observables together into one dictionary so protocol evaluates it at once
        # This is for global commuting terms to be implemented
        qse_observables = {("H", *element): observable
                            for element, observable in self.h_data.items()}
        qse_observables.update({("S", *element): observable
                                 for element, observable in self.s_data.items()})

        H, S = assemble_matrix_outputs(protocol.evaluate(circuit, qse_observables))

        return H, S
