from qibochem.ansatz.ucc import UCCAnsatz
import numpy as np
from dataclasses import dataclass, field
from qibochem.ansatz.ucc_util import excitation2qubit_observable
from openfermion.linalg import get_sparse_operator
from qibochem.driver.hamiltonian import _qubit_hamiltonian
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csc_matrix, lil_matrix
import itertools

'''
New class for general UPS and tUPS. 
Object inherits from previous UCCAnsatz with added properties and functions for faster evaluation.
'''

@dataclass
class UPSAnsatz(UCCAnsatz):
    use_projection: bool = False
    ref_bitstring: str | None = None
    spin_preserving: bool = False # if S_z is preserved using spin adapted or paired operators
    perfect_pair: bool = False
    oo_layers: str | int = 0

    def __post_init__(self):
        self.n_spat = self.mol.norb
        self.n_spin = self.mol.nso
        self.n_elec = self.mol.nelec
        self.n_active_elec = self.mol.n_active_e
        self.n_active_spat = self.mol.n_active_orbs // 2
        self.n_active_spin = self.mol.n_active_orbs
        self.n_alpha = self.mol.nalpha
        self.n_beta = self.mol.nbeta
        self.N = 2 ** self.n_active_spin # size of Fock space
        if self.oo_layers == 'full':
            self.oo_layers = self.n_active_spat // 2
        super().__post_init__()

        if self.use_mat_mul:
            if self.use_projection:
                self._initialise_projection_mat()

            self.hf_ref = self._initialise_reference()

            self._generate_operator_matrix()

            self._initialise_hamiltonian()

            self.theta_vector = np.array([self.initial_params[name] for name in self.param_names])

            self._update_mat_mul(self.theta_vector)


    def _initialise_reference(self):
        sv = np.zeros(self.N, dtype=complex)
        if self.ref_bitstring is not None:
            sv_idx = int(self.ref_bitstring,2)
        else:
            bitstring = '1' * self.n_active_elec + '0' * (self.n_active_spin - self.n_active_elec)
            sv_idx = int(bitstring,2)
        sv[sv_idx] = 1.0

        if self.use_projection:
            sv = self.proj_mat.T @ sv

        return sv

    def _initialise_hamiltonian(self):
        if not self.perfect_pair:
            self.h_mat = get_sparse_operator(self.mol.hamiltonian('qubit', ferm_qubit_map=self.ferm_qubit_map))
        else:
            perm = [0,3,1,4,2,5]
            oei = self.mol.oei
            tei = self.mol.tei
            oei_pp = oei[np.ix_(perm, perm)]
            tei_pp = tei[np.ix_(perm, perm, perm, perm)]
            self.h_mat = get_sparse_operator(self.mol.hamiltonian('qubit', ferm_qubit_map=self.ferm_qubit_map, oei=oei_pp, tei=tei_pp))
        if self.use_projection:
            self.h_mat = self.proj_mat.T @ (self.h_mat @ self.proj_mat)

    def _generate_operator_matrix(self):
        self.operator_mat_dict = {}
        for name, weighted_excitations in self.param_excitations.items():
            qubit_op = excitation2qubit_observable(weighted_excitations[0],
                                                    ferm_qubit_map=self.ferm_qubit_map)
            for weighted_excitation in weighted_excitations[1:]:
                qubit_op += excitation2qubit_observable(weighted_excitation,
                                                        ferm_qubit_map=self.ferm_qubit_map)
            if self.use_projection:
                self.operator_mat_dict[name[1:4]] = self.proj_mat.T @ get_sparse_operator(qubit_op, n_qubits=self.n_active_spin) @ self.proj_mat
            else:
                self.operator_mat_dict[name[1:4]] = get_sparse_operator(qubit_op, n_qubits=self.n_active_spin)

    def _update_mat_mul(self, theta_vector):
        if self.use_projection:
            N = self.proj_N
        else:
            N = self.N

        self.wfn = self.hf_ref.copy()

        grad_wfn = np.zeros((N, self.dim))
        for j in range(grad_wfn.shape[1]):
            grad_wfn[:,j] = self.hf_ref.copy()

        for idx, name in enumerate(self.param_names):
            grad_wfn = expm_multiply(self.operator_mat_dict[name[1:4]] * theta_vector[idx], grad_wfn)
            grad_wfn[:,idx] = self.operator_mat_dict[name[1:4]] @ grad_wfn[:,idx]
            self.wfn = expm_multiply(self.operator_mat_dict[name[1:4]] * theta_vector[idx], self.wfn)
        
        self.gradient = 2 * np.conj(grad_wfn).T @ self.h_mat @ self.wfn 
        self.energy = np.conj(self.wfn).T @ (self.h_mat @ self.wfn)


    # @property
    # def energy(self):
    #     E = np.conj(self.wfn) @ (self.h_mat @ self.wfn)
    #     return E

    @property
    def dim(self):
        return len(self.param_names)

    def _get_fast_mat_mul_energy(self, x):
        self._update_mat_mul(x)
        return (self.energy, self.gradient)

    def _initialise_projection_mat(self):
        bitstrings = []

        # get combinations of allowed bitstrings
        alpha_positions = [i for i in range(0,self.n_active_spin,2)]
        beta_positions = [i for i in range(1,self.n_active_spin,2)]

        for odd_choice in itertools.combinations(alpha_positions, 3):
            for even_choice in itertools.combinations(beta_positions, 3):
                s = ['0'] * self.n_active_spin

                for i in odd_choice:
                    s[i] = '1'

                for i in even_choice:
                    s[i] = '1'

                bitstrings.append(int("".join(s),2))
        bitstrings.sort()
        # construct projector matrix and dimension of reduced space
        self.proj_mat = lil_matrix((self.N, len(bitstrings)))
        for j, idx in enumerate(bitstrings):
            self.proj_mat[idx,j] = 1
        self.proj_mat = self.proj_mat.tocsc()
        self.proj_N = len(bitstrings)


class Ansatz_tUPS(UPSAnsatz):
    def __init__(self, mol, layers=1, use_first_singles=True,**kwargs):
        self.layers = layers
        self.use_first_singles = use_first_singles
        super().__init__(mol, **kwargs)

        
    def excitations(self):
        param_excitations = {}
        for l in range(self.layers):
            # defining k_10, k_32, k_54, ... k_pq. where q is even 
            # 1st half layer of a tups layer
            for p in range(2, self.n_orbs, 4):
                q = p-2
                # spin adapted singles set 1
                param_excitations[f'{l+1}s{p//2}{q//2}-1'] = [(1.0,((q,),(p,))),(1.0,((q+1,),(p+1,)))]
                # paired doubles
                param_excitations[f'{l+1}d{p//2}{q//2}-2'] = [(1.0,((q,q+1),(p,p+1)))] # -1?
                # spin adapted singles set 2
                param_excitations[f'{l+1}s{p//2}{q//2}-3'] = [(1.0,((q,),(p,))),(1.0,((q+1,),(p+1,)))]
            # defining k_21, k_43, k_65, ... k_pq. where q is odd 
            # 2nd half layer of a tups layer
            for q in range(2, self.n_orbs-2, 4):
                p = q+2
                # spin adapted singles set 1
                param_excitations[f'{l+1}s{p//2}{q//2}-1'] = [(1.0,((q,),(p,))),(1.0,((q+1,),(p+1,)))]
                # paired doubles
                param_excitations[f'{l+1}d{p//2}{q//2}-2'] = [(1.0,((q,q+1),(p,p+1)))] # -1?
                # spin adapted singles set 2
                param_excitations[f'{l+1}s{p//2}{q//2}-3'] = [(1.0,((q,),(p,))),(1.0,((q+1,),(p+1,)))]

        # orbital optimisation layers
        for x in range(self.oo_layers):
            # 1st half layer of oo layer
            for p in range(2, self.n_orbs, 4):
                q = p-2
                # spin adapted singles
                param_excitations[f'{x+1}s{p//2}{q//2}-1oo'] = [(1.0,((q,),(p,))),(1.0,((q+1,),(p+1,)))]

            # 2nd half layer of oo layer
            for q in range(2, self.n_orbs-2, 4):
                p = q+2
                # spin adapted singles
                param_excitations[f'{x+1}s{p//2}{q//2}-1oo'] = [(1.0,((q,),(p,))),(1.0,((q+1,),(p+1,)))]
        return param_excitations

