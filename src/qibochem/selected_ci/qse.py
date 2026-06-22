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
    excitation_generator: Callable
    spin_projection: int | str = 0 # Can be 0 -1 1 or "all"
    ferm_qubit_map: str = "jw"

    map_threshold: float = 1e-12

    s_cache_path: str | None = None
    h_cache_path: str | None = None
    h_map_cache_path: str | None = None


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
    
    
    def _get_labeled_hamiltonian(self):
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

    def _reconstruct_H_ij_from_map(self, H_map):
        """
        Function to reconstruct the H matrix from a H_map for a given molecule
        The excitation map stores in qubit representation, but H data will be in bitmask observable form.
        Reconstruct the qubit terms first from the OEI and TEI terms, then convert into bitmask observable form.
        
        H_map has the form {label: observables}
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

        h_constant = 0.0
        terms = defaultdict(complex)
        # Doesnt use the default bitmask observable add function because of overhead cost
        for label, template_observable in H_map:
            integral = integral_value(label)
            h_constant += integral * template_observable.constant
            for term, coeff in template_observable.terms.items():
                terms[term] += integral * coeff
        # Re build the bitmask observable back at the end
        # Filter away the small terms lower than map threshold
        H_compressed = BitmaskObservable(constant=h_constant if abs(h_constant) > self.map_threshold else 0.0,
                                         terms={term: coeff for term, coeff in terms.items()
                                            if abs(coeff) > self.map_threshold})
        return H_compressed
    
    def _get_Ei_Ej(self, i, j):
        ferm_Ei_dag = openfermion.hermitian_conjugated(self.operators[i])
        ferm_Ej = self.operators[j]
        bitmask_Ei_dag = self._map_ferm2bitmask(ferm_Ei_dag)
        bitmask_Ej = self._map_ferm2bitmask(ferm_Ej)
        return bitmask_Ei_dag, bitmask_Ej

    def _get_S_ij(self, i, j):
        """
        Get the Ei and Ej first
        Do bitmask multiplication 
        """
        bitmask_Ei, bitmask_Ej = self._get_Ei_Ej(i, j)
        return multiply_bitmask_observables(bitmask_Ei, bitmask_Ej, threshold=self.map_threshold)
    
    def _get_H_map_ij(self, i, j, labeled_hamiltonian):
        """
        Get a term of the excitaiton map of H
        """
        bitmask_Ei, bitmask_Ej = self._get_Ei_Ej(i, j)
        labeled_projected_hamiltonian = {}
        for label, ferm_ham in labeled_hamiltonian:
            bitmask_ham = self._map_ferm2bitmask(ferm_ham)
            left_prod = multiply_bitmask_observables(bitmask_Ei, bitmask_ham, 
                                                     threshold=self.map_threshold * 1e-2)
            label_projected_ham = multiply_bitmask_observables(left_prod, bitmask_Ej, 
                                                                threshold=self.map_threshold * 1e-2)
            labeled_projected_hamiltonian[label] = label_projected_ham
        return labeled_projected_hamiltonian

    def _get_H_ij_direct(self, i, j, ferm_hamiltonian):
        """
        Get a term of Hij directly from the hamilltonian
        """
        bitmask_Ei, bitmask_Ej = self._get_Ei_Ej(i, j)
        bitmask_ham = self._map_ferm2bitmask(ferm_hamiltonian)
        left_prod = multiply_bitmask_observables(bitmask_Ei, bitmask_ham, 
                                                 threshold=self.map_threshold * 1e-2)
        projected_ham = multiply_bitmask_observables(left_prod, bitmask_Ej, 
                                                            threshold=self.map_threshold * 1e-2)
        return projected_ham

    def _prepare_s_data(self):
        """
        Load S Datat from cache path is its available
        If not build S data and optionally cache it
        """
        if self.s_cache_path is not None:
            path = Path(self.s_cache_path)
            if path.exists():
                self.s_data = pickle2dict(self.s_cache_path)
                return
        # If not build in the s_matrix
        dim = len(self.operators)
        for i, j in product(range(dim), repeat=2):
            if i > j: 
                continue # Only build upper triangle, will mirror for lower triangle later
            self.s_data[(i, j)] = self._get_S_ij(i, j)
        # Optional caching if the cache path is provided
        if self.s_cache_path is not None:
            dict2pickle(self.s_data, self.s_cache_path)


    def _prepare_h_data(self):
        """
        Function main goal is to build self.h_data
        First checks if the molecule specific map is provided 
        Also checks if the individual excitation map is provided, cache excitation maps if given
        If no cache paths are provided, then directly builds H_data from hamiltonian isntead of labeled hamiltonian
        """
        # First check if the molecule specific H map is provided
        if self.h_cache_path is not None:
            path = Path(self.h_cache_path)
            if path.exists():
                self.h_data = pickle2dict(self.h_cache_path)
                return
        
        dim = len(self.operators)
        # If it doesnt exist, then check if the caching of excitation map is wanted
        if self.h_map_cache_path is not None:
            """
            Here we build the individual elements from the cache because the RAM can quickly explode 
            The individual hamiltonian terms are huge for large active space, so the cached files are huge also
            Loading in the full excitation map before compression is not RAM safe
            It is definitely slower because more overhead of constantly referencing zip file but good trade off for RAM

            In this case h_map_cache_path expected is a folder path
            """
            labeled_hamiltonian = self._get_labeled_hamiltonian()
            for i, j in product(range(dim), repeat=2):
                if i > j:
                    continue # Only build upper triangular
                # Check if the file exists
                map_ij_path = Path(self.h_map_cache_path+f"/H_{i}_{j}.pkl")
                if map_ij_path.exists():
                    h_observable_map = pickle2dict(map_ij_path)
                # If it doesnt exist, then make the map and immediately cache it
                else:
                    h_observable_map = self._get_H_map_ij(i, j, labeled_hamiltonian)
                    dict2pickle(h_observable_map, map_ij_path)
                
                # Use the map to reconstruct the H_ij
                H_ij = self._reconstruct_H_ij_from_map(h_observable_map)
                self.h_data[(i, j)] = H_ij

        # If excitation map is not provided, then directly build H data
        else:
            ferm_hamiltonian = self.molecule.hamiltonian("f")
            for i, j in product(range(dim), repeat=2):
                if i > j:
                    continue # Only build upper triangular
                H_ij = self._get_H_ij_direct(i, j, ferm_hamiltonian)
                self.h_data[(i, j)] = H_ij

        # Cache the hamiltonian if path is provided
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
        # This can be reused, IE the QSE computable can be reused for same active space and molecule for 
        # different ansatz etc. So if re-used, the terms will not be re-collated
        if self.h_data is None or self.s_data is None:
            self.h_data, self.s_data = defaultdict(), defaultdict()
            self._prepare_s_data()
            self._prepare_h_data()

        # Group the H and S observables together into one dictionary so protocol evaluates it at once
        # This is for global commuting terms to be implemented
        qse_observables = {("H", *element): observable
                            for element, observable in self.h_data.items()}
        qse_observables.update({("S", *element): observable
                                 for element, observable in self.s_data.items()})

        H, S = assemble_matrix_outputs(protocol.evaluate(circuit, qse_observables))

        return H, S
