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
import openfermion

from qibochem.driver.observables import (
    BitmaskObservable,
    multiply_bitmask_observables,
    qubit_operator2observable,
)
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

    def _map_operator_to_observable(self, projected):
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
        return qubit_operator2observable(q_op)

    def _build_excitation_map(self):
        """
        Stores excitation map for both H and S in self.excitation_map 
        For S, the map is just S directly in bitmask observable form
        For H, the map is stored in terms of OEI and TEI labels 
        H[i, j] = {label: bitmask_observable}
        label is either ("oei", p, q) or ("tei", p, q, r, s) or ("constant",)
        """
        labeled_hamiltonian = self._labeled_hamiltonian() # [(("oei", p, q), a_p^ a_q), (("tei", p, q, r, s), a_p^ a_q^ a_r a_s), ...]
        dim = len(self.operators)
        self.excitation_map = {"H": {}, "S": {}}

        # Map all reusable factors to bitmask once, then do the expensive products in bitmask form.
        e_observables = [self._map_operator_to_observable(operator)
                         for operator in self.operators]
        edag_observables = [self._map_operator_to_observable(openfermion.hermitian_conjugated(operator))
                            for operator in self.operators]
        bitmask_labeled_hamiltonian = [(label, self._map_operator_to_observable(operator))
                                       for label, operator in labeled_hamiltonian]

        # Each matrix element is of the form E_dag P Ej, so here we cache P*Ej and reuse it 
        right_products = []
        for Ej in e_observables:
            right_products_j = []
            for label, h_observable in bitmask_labeled_hamiltonian:
                right_product = multiply_bitmask_observables(h_observable, Ej, threshold=self.map_threshold*1e-2)
                # Use intermediate thresold above so that the final threshold is not affected
                if right_product.constant or right_product.terms: # Remove all the 0 terms
                    right_products_j.append((label, right_product))
            right_products.append(right_products_j)

        # Build the full map now from all the cached terms
        for i, j in product(range(dim), repeat=2):
            if i > j:
                continue # Only build upper triangle, will mirror for lower triangle later

            s_observable = multiply_bitmask_observables(edag_observables[i], e_observables[j], 
                                                        threshold=self.map_threshold*1e-2)
            # Store each labelled Hamiltonian contribution as its own observable template.
            h_map = {}
            for label, right_product in right_products[j]:
                projected_observable = multiply_bitmask_observables(edag_observables[i], right_product, 
                                                                    threshold=self.map_threshold*1e-2)
                h_map[label] = projected_observable
                
            self.excitation_map["S"][(i, j)] = s_observable
            self.excitation_map["H"][(i, j)] = h_map

    def _reconstruct_HS_from_map(self):
        """
        Function to reconstruct the H and S matrix from the excitation map for a given molecule
        The excitation map stores in qubit representation, but H and S data will be in bitmask observable form.
        For S, just need to convert map from qubit to bitmask observable form.
        For H, need to reconstruct the qubit terms first from the OEI and TEI terms,
        then convert into bitmask observable form.
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

        # S is already molecule-independent and stored directly as observables.
        self.s_data = dict(self.excitation_map["S"])

        # H templates are weighted by molecule-specific OEI/TEI/constant values.
        self.h_data = {}
        for element, h_map in self.excitation_map["H"].items():
            h_constant = 0.0
            terms = defaultdict(complex)
            # Doesnt use the default bitmask observable add function because of overhead cost
            for label, template_observable in h_map.items():
                integral = integral_value(label)
                h_constant += integral * template_observable.constant
                for term, coeff in template_observable.terms.items():
                    terms[term] += integral * coeff
            # Re build the bitmask observable back at the end
            # Filter away the small terms lower than map threshold
            self.h_data[element] = BitmaskObservable(
                constant=h_constant if abs(h_constant) > self.map_threshold else 0.0,
                terms={term: coeff for term, coeff in terms.items()
                       if abs(coeff) > self.map_threshold}
            )

    def collate_hs_matrix_info(self):
        """
        Main function to build the H and S matrix of observables. 
        If the excitation map is not yet built, then build it first 
        After that reconstruct the HS into self.H_data and self.S_data with the reconstruct function
        """
        if self.operators is None:
            self.operators = self.excitation_generator(self.excitation_params)

        if self.excitation_map is None:
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
        # First define the excitation operators.
        if self.operators is None:
            self.operators = self.excitation_generator(self.excitation_params)

        # Update the H and S observables if they were not done before
        # This can be reused, IE the QSE computable can be reused for same active space and molecule for 
        # different ansatz etc. So if re-used, the terms will not be re-collated
        if self.h_data is None or self.s_data is None:
            self.collate_hs_matrix_info()

        # Group the H and S observables together into one dictionary so protocol evaluates it at once
        # This is for global commuting terms to be implemented
        qse_observables = {("H", *element): observable
                            for element, observable in self.h_data.items()}
        qse_observables.update({("S", *element): observable
                                 for element, observable in self.s_data.items()})

        H, S = assemble_matrix_outputs(protocol.evaluate(circuit, qse_observables))

        return H, S
