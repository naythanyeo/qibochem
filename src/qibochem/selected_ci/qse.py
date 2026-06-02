"""Quantum Subspace Expansion (QSE)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import openfermion
import sympy as sp
from qibo import symbols
from qibo.hamiltonians import SymbolicHamiltonian

from qibochem.driver.hamiltonian import _qubit_hamiltonian, _qubit_to_symbolic_hamiltonian
from qibochem.selected_ci.utils import assemble_matrix_outputs

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
    excitation_generator: Callable
    spin_projection: int | str = 0
    ferm_qubit_map: str = "jw"
    cache_qse_matrix: bool = True
    excitation_map: dict = None # Cached QSE excitation map --> RETURN and CACHE THIS
    map_threshold: float = 1e-12

    operators: list | None = field(default=None, init=False)
    s_data: dict | None = field(default=None, init=False)
    h_data: dict | None = field(default=None, init=False)
    excitation_params: dict | None = field(default=None, init=False)

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
    
    
    def _labeled_hamiltonian(self):
        """
        Function to generate a labeled hamiltonian form to cache the QSE terms 
        For every molecule, can re-use the QSE map by reconstructing from OEI and TEI terms
        instead of performing repeated JW conversion multiple times. 
        For consistency, input n_orbs is the number of spin orbitals
        Output will be a list of tuples in the form [(label, operator), ...]
        
        Spin rules: for OEI the spin of both must be same because h doesnt act on spinnors, 
        so spin-orbitlas with opposite spin will automatically be orthogonal 
        For TEI, g = int (Xp(1)Xq(2) 1/r12 Xr(2)Xs(1)) so p and s must have same spin, 
        and q and r must have same spin, otherwise the term is 0.
        """
        n_spin_orbitals = self.excitation_params["n_orbs"]
        labeled_hamiltonian = [
            (("constant",), openfermion.FermionOperator((), 1.0))
        ]
        # OEI terms
        for p, q in product(range(n_spin_orbitals), repeat=2):
            if p%2 != q%2: 
                continue # Skip all the spin flip OEI terms because they are 0
            label = ("oei", p , q)
            # Create p and destroy q
            operator = openfermion.FermionOperator(((p, 1), (q, 0)), 1) 
            labeled_hamiltonian.append((label, operator))
        # TEI terms
        for p, q, r, s in product(range(n_spin_orbitals), repeat=4):
            if p%2 != s%2 or q%2 != r%2:
                continue # Skip all the spin flip TEI terms because they are 0
            label = ("tei", p , q, r, s)
            # Create p and q, destroy r and s
            operator = openfermion.FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)), 1)
            labeled_hamiltonian.append((label, operator))
        return labeled_hamiltonian

    def _map_projected_terms(self, projected):
        """
        Function to map projected terms to qubit terms based on mapping choice
        There is already a mapping function fermionic_to_qubit but this is more direct
        and takes in just the operators rather than the whole fermionic hamiltonian object
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
        return q_op.terms

    def _build_excitation_map(self):
        """
        Stores excitation map for both H and S in self.excitation_map 
        For S, the map is just S directly in qubit form since its the same for all molecules
        For H, the map is stored in terms of OEI and TEI labels 
        H[i, j] = {pauli_string: {label: prefactor,....}}
        label is either ("oei", p, q) or ("tei", p, q, r, s) or ("constant",)
        """
        labeled_hamiltonian = self._labeled_hamiltonian() # [(("oei", p, q), a_p^ a_q), (("tei", p, q, r, s), a_p^ a_q^ a_r a_s), ...]
        dim = len(self.operators)
        self.excitation_map = {"H": {}, "S": {}}
        # Loop through the i, j terms for all excitation operators
        for i, j in product(range(dim), repeat=2):
            if i > j:
                continue # Only build upper triangle, will mirror for lower triangle later
            # Projection terms
            Ei_dag = openfermion.hermitian_conjugated(self.operators[i])
            Ej = self.operators[j]
            # S map
            s_ferm = Ei_dag * Ej
            self.excitation_map["S"][(i, j)] = dict(self._map_projected_terms(s_ferm))
            # H map
            h_map = defaultdict(lambda: defaultdict(complex))
            # Loop through all the labeled hamiltonian terms
            for label, op in labeled_hamiltonian:
                projected = Ei_dag * op * Ej
                if not projected.terms:
                    continue
                for pauli_string, prefactor in self._map_projected_terms(projected).items():
                    if abs(prefactor) > self.map_threshold:
                        h_map[pauli_string][label] += prefactor
            # Convert default dict to regular dictionary so pickle can work
            self.excitation_map["H"][(i, j)] = {
                pauli_string: dict(label_map)
                for pauli_string, label_map in h_map.items()
            }

    def _pauli_terms_to_symbolic(self, pauli_terms):
        """
        Convert cached OpenFermion Pauli-string terms to SymbolicHamiltonian.
        Almost the same as _qubit_to_symbolic_hamiltonian, but takes in a dict of pauli terms 
        instead of a QubitOperator. TBH can convert to qubit_hamiltonian object first but redefining 
        it here saves some overhead cost of building a full operator.
        """
        pauli_symbols = {
            (pauli_op, qubit): getattr(symbols, pauli_op)(qubit)
            for pauli_string in pauli_terms
            for qubit, pauli_op in pauli_string
        }
        symbolic_terms = [
            sp.Mul(
                coeff,
                *(pauli_symbols[(pauli_op, qubit)] for qubit, pauli_op in pauli_string),
            )
            for pauli_string, coeff in pauli_terms.items()
            if abs(coeff) > self.map_threshold # Here we use the main map threshold
        ]
        symbolic_expr = sp.Add(*symbolic_terms) if symbolic_terms else sp.Integer(0)
        return SymbolicHamiltonian(symbolic_expr, nqubits=self.excitation_params["n_orbs"])
    
    def _reconstruct_HS_from_map(self):
        """
        Function to reconstruct the H and S matrix from the excitation map for a given molecule
        The excitation map stores in qubit representation, but H and S data will be in symbolic form
        For S, just need to convert map from qubit to symbolic form 
        For H, need to reconstruct the qubit terms first from the OEI and TEI terms,
        then convert into symbolic form 
        """
        # First get all the OEI and TEI terms needed
        # This part is almost copying molecule.hamiltonian("f")
        oei = self.molecule.oei if self.molecule.embed_oei is None else self.molecule.embed_oei
        tei = self.molecule.tei if self.molecule.embed_tei is None else self.molecule.embed_tei
        constant = 0.0 if self.molecule.inactive_energy is None else self.molecule.inactive_energy
        constant += self.molecule.e_nuc

        oei = self.molecule._filter_array(oei, self.map_threshold)
        tei = self.molecule._filter_array(tei, self.map_threshold)
        oei_so, tei_so = openfermion.ops.representations.get_tensors_from_integrals(oei, tei)

        # Mini helper function to match the qubit operator labels to constant, OEI and TEI values
        def integral_value(label):
            if label[0] == "constant":
                return constant
            if label[0] == "oei":
                _, p, q = label
                return oei_so[p, q]
            if label[0] == "tei":
                _, p, q, r, s = label
                return tei_so[p, q, r, s]
            raise ValueError(f"Unknown Hamiltonian label: {label}")
        # S data just reconstruct into symbolic form
        for element, s_terms in self.excitation_map["S"].items():
            self.s_data[element] = self._pauli_terms_to_symbolic(s_terms)
        # H data will map the constant, OEI and TEI terms first then convert to symbolic form
        for element, h_map in self.excitation_map["H"].items():
            h_terms = {}
            for pauli_string, label_map in h_map.items():
                coeff = sum(
                    prefactor * integral_value(label)
                    for label, prefactor in label_map.items()
                )
                if abs(coeff) > self.map_threshold:
                    h_terms[pauli_string] = coeff
            self.h_data[element] = self._pauli_terms_to_symbolic(h_terms)

    def _has_collated_matrix_info(self):
        """
        Check whether H/S observables have already been built.
        """
        return (
            self.h_data is not None
            and self.s_data is not None
            and all(isinstance(observable, SymbolicHamiltonian) for observable in self.h_data.values())
            and all(isinstance(observable, SymbolicHamiltonian) for observable in self.s_data.values())
        )


    def _collate_hs_matrix_direct(self):
        dim = len(self.operators)
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
                    self.excitation_params["n_orbs"],
                )

    def collate_hs_matrix_info(self):
        """
        Main function to build the H and S matrix of observables. 
        Supports caching of the projected H and S terms in terms of OEI and TEI terms
        Builds a labeled hamiltonian first, then reconstruct for each molecule quickly 
        Cached hamiltonian saves time for many molecules
        For every active space and excitation generator, there will be 1 excitation map 
        that can be re-used across molecules and ansatz, saves the heavy lifting of JW

        Cached hamiltonian will be labeled in the form ... 
        
        If only single molecule and no caching wanted, then can use direct mode instead
        which will directly compute the H and S matrix from fermionic Hamiltonian
        """
        if self._has_collated_matrix_info():
            return
        if self.excitation_map is not None: 
            self.cache_qse_matrix = True
        if self.operators is None:
            self.operators = self.excitation_generator(self.excitation_params)
        dim = len(self.operators)
        # Build the dictionaries in s_data and h_data 
        self.s_data = {(_i, _j): dict() for _i in range(dim) for _j in range(dim) if _i <= _j}
        self.h_data = {(_i, _j): dict() for _i in range(dim) for _j in range(dim) if _i <= _j}
        # If its false, then build via direct method and store self.s_data and self.h_data
        if self.cache_qse_matrix is False:
            self._collate_hs_matrix_direct()
            return  
        # In this case, if caching is desired, then it will build the H S observable map from the excitation mpa first
        # If the excitaiton map is not yet built, then construct it 
        elif self.excitation_map is None:
            self._build_excitation_map()
        # From the excitation map, reconstruct H and S based on OEI and TEI terms
        self._reconstruct_HS_from_map()
        # All the terms stored in self.h_data and self.s_data



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
        # First define the excitation operators 
        if self.operators is None:
            self.operators = self.excitation_generator(self.excitation_params)

        # Update the H and S observables  
        if not self._has_collated_matrix_info():
            self.collate_hs_matrix_info()
        H_values = protocol.evaluate(circuit, self.h_data)
        S_values = protocol.evaluate(circuit, self.s_data)

        H = assemble_matrix_outputs(H_values)
        S = assemble_matrix_outputs(S_values)

        return H, S
