from qibochem.ansatz.ucc import UCCAnsatz
import numpy as np
from dataclasses import dataclass, field
from qibochem.ansatz.ucc_util import excitation2qubit_observable
from openfermion.linalg import get_sparse_operator
from openfermion import FermionOperator, jordan_wigner
from qibo.optimizers import optimize
from qibochem.driver.hamiltonian import _qubit_hamiltonian
from qibochem.measurement.protocol import StateVectorProtocol
from scipy.sparse.linalg import expm_multiply
from scipy.linalg import expm
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
    mo_perm: list | None = None

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
            self.hf_ref = self._initialise_reference()


            if self.use_projection:
                self._initialise_projection_mat()
                self.hf_ref = self.proj_mat.T @ self.hf_ref

            self._generate_operator_matrix()

            self._initialise_hamiltonian()

            self.theta_vector = np.array([self.initial_params[name] for name in self.param_names])

            self._update_mat_mul(self.theta_vector)

            self._initialise_rdm_ops()


    def _initialise_reference(self):
        '''Generate statevector reference from given bitstring or just from the HF reference order.
        Number of alpha and beta electrons used for generation of the projection matrix later.
        '''
        sv = np.zeros(self.N, dtype=complex)
        if self.ref_bitstring is not None:
            sv_idx = int(self.ref_bitstring,2)
        else:
            self.ref_bitstring = '1' * self.n_active_elec + '0' * (self.n_active_spin - self.n_active_elec)
            sv_idx = int(self.ref_bitstring,2)
        sv[sv_idx] = 1.0

        self.n_active_alpha = sum(1 for idx, bit in enumerate(self.ref_bitstring) if idx % 2 == 0 and bit == '1')
        self.n_active_beta = sum(1 for idx, bit in enumerate(self.ref_bitstring) if idx % 2 == 1 and bit == '1')

        return sv

    def _initialise_hamiltonian(self):
        '''Function to create a the molecular hamiltonian in the form of a sparse matrix.
        Perfect pairing reorders the molecular coefficient indices as given by mo_perm.
        Using the projection matrix reduces the dimension of the matrix from the Fock space to the 
        smaller Hilbert space for faster computation.
        '''
        if not self.perfect_pair:
            self.h_mat = get_sparse_operator(self.mol.hamiltonian('qubit', ferm_qubit_map=self.ferm_qubit_map))
        else:
            if self.mo_perm is None:
                self.mo_perm = [0 for _ in range(self.n_active_spat)]
                i = 0
                for _ in range(0,self.n_active_spat,2):
                    self.mo_perm[_] = i
                    i += 1

                for _ in range(self.n_active_spat-1-self.n_active_spat % 2, 0, -2):
                    self.mo_perm[_] = i
                    i += 1
            if len(self.mo_perm) != self.n_active_spat:
                raise ValueError("Length of permutation list does not equal the number of spatial orbitals!")

            oei = self.mol.embed_oei
            tei = self.mol.embed_tei
            oei_pp = oei[np.ix_(self.mo_perm, self.mo_perm)]
            tei_pp = tei[np.ix_(self.mo_perm, self.mo_perm, self.mo_perm, self.mo_perm)]
            self.h_mat = get_sparse_operator(self.mol.hamiltonian('qubit', ferm_qubit_map=self.ferm_qubit_map, oei=oei_pp, tei=tei_pp))
        if self.use_projection:
            self.h_mat = self.proj_mat.T @ (self.h_mat @ self.proj_mat)

    def _generate_operator_matrix(self):
        '''Function to generate the excitation operator matrices for multiplication later during
        the update step.
        Using the projection matrix reduces the dimension of each of the operator matrices for faster calculation.'''
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
        '''Function to update wave function, energy and gradient during optimisation.
        Using the analytical gradient speeds up the optimisation as using the numerical gradient is slower.'''

        if self.use_projection: # use smaller dimension if projected
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
        
        self.gradient = 2 * np.real(np.conj(grad_wfn).T @ self.h_mat @ self.wfn)
        self.energy = np.real(np.conj(self.wfn).T @ (self.h_mat @ self.wfn))


    @property
    def dim(self):
        '''returns number of parameters'''
        return len(self.param_names)

    def _get_fast_mat_mul_energy(self, x):
        '''loss function called during optimisation'''
        self._update_mat_mul(x)
        return (self.energy, self.gradient)

    def _initialise_projection_mat(self):
        '''initialise projection matrix for reducing dimension from Fock space 
        to the smaller Hillbert space with constant quantum numbers (particle, spin).
        Speeds up the matrix multiplication step'''
        bitstrings = []

        if self.spin_preserving:
            # get combinations of allowed bitstrings that conserves spin and particle number
            alpha_positions = [i for i in range(0,self.n_active_spin,2)]
            beta_positions = [i for i in range(1,self.n_active_spin,2)]

            for odd_choice in itertools.combinations(alpha_positions, self.n_active_alpha):
                for even_choice in itertools.combinations(beta_positions, self.n_active_beta):
                    s = ['0'] * self.n_active_spin

                    for i in odd_choice:
                        s[i] = '1'

                    for i in even_choice:
                        s[i] = '1'
                    bitstrings.append(int("".join(s),2))
        else:
            # get combinations of allowed bitstrings that conserves only particle number
            perm_str = '1'*self.n_active_elec + '0'*(self.n_active_spin-self.n_active_elec)
            perms = tuple(set(itertools.permutations(perm_str)))
            bitstrings = [''.join(x) for x in perms]
        bitstrings.sort()
        # construct projector matrix and dimension of reduced space
        self.proj_mat = lil_matrix((self.N, len(bitstrings)))
        for j, idx in enumerate(bitstrings):
            self.proj_mat[idx,j] = 1
        self.proj_mat = self.proj_mat.tocsc()
        self.proj_N = len(bitstrings)

    def _ansatz_active_orbitals(self):
        if not self.perfect_pair:
            return self.mol.active
        return [self.mol.active[i] for i in self.mo_perm]

    def _initialise_rdm_ops(self):
        self.o_rdm_ops = {}
        self.t_rdm_ops = {}
        chi = {}
        for p in range(self.n_active_spat):
            for q in range(self.n_active_spat):
                # 1-RDM
                ferm_op = FermionOperator(f"{p*2}^ {q*2}") + FermionOperator(f"{p*2+1}^ {q*2+1}")
                qubit_op = jordan_wigner(ferm_op)
                op_mat = get_sparse_operator(qubit_op, n_qubits=self.n_active_spin)
                if self.use_projection:
                    op_mat = self.proj_mat.T @ op_mat @ self.proj_mat
                self.o_rdm_ops[p,q] = op_mat
                # 2-RDM
                for tau in range(2):
                    for sigma in range(2):
                        ferm_op = FermionOperator(f"{q*2+tau} {p*2+sigma}") 
                        qubit_op = jordan_wigner(ferm_op)         
                        op_mat = get_sparse_operator(qubit_op, n_qubits=self.n_active_spin)
                        if self.use_projection:
                            op_mat = op_mat @ self.proj_mat
                        chi[p,q,tau,sigma] = op_mat

        for p in range(self.n_active_spat):
            for q in range(self.n_active_spat):
                for r in range(self.n_active_spat):
                    for s in range(self.n_active_spat):
                        rdm_op = csc_matrix((self.proj_N, self.proj_N),dtype=complex)
                        for tau in range(2):
                            for sigma in range(2):
                                rdm_op += chi[p,q,tau,sigma].conj().T @ chi[r,s,tau,sigma]
                        self.t_rdm_ops[p,q,r,s] = rdm_op

    def get_spat_1rdm(self):
        self.o_rdm = np.zeros([self.n_active_spat,self.n_active_spat])
        for p in range(self.n_active_spat):
            for q in range(self.n_active_spat):
                self.o_rdm[p][q] = np.conj(self.wfn).T @ self.o_rdm_ops[p,q] @ self.wfn 

    def get_spat_2rdm(self):
        self.t_rdm = np.zeros([self.n_active_spat,self.n_active_spat,self.n_active_spat,self.n_active_spat])
        for p in range(self.n_active_spat):
            for q in range(self.n_active_spat):
                for r in range(self.n_active_spat):
                    for s in range(self.n_active_spat):
                        self.t_rdm[p][q][r][s] = np.vdot(self.wfn, self.t_rdm_ops[p,q,r,s] @ self.wfn)

    def _energy_from_rdms(self):
        oei = self.mol.embed_oei
        tei = self.mol.embed_tei
        if self.perfect_pair:
            permutation = self.mo_perm
            oei = oei[np.ix_(permutation, permutation)]
            tei = tei[np.ix_(
                permutation,
                permutation,
                permutation,
                permutation,
            )]

        constant = 0.0 if self.mol.inactive_energy is None else self.mol.inactive_energy
        constant += self.mol.e_nuc
        energy = np.einsum("pq,qp", oei, self.o_rdm) + 0.5* np.einsum("pqsr, pqrs",tei,self.t_rdm) + constant
        self.energy = energy
        return energy

    def _active_fock_matrix(self):
        active_fock = np.zeros([self.n_spat,self.n_spat])
        for p in range(self.n_spat):
            for q in range(self.n_spat):
                for v, V in enumerate(self.mol.active):
                    for w, W in enumerate(self.mol.active):
                        active_fock[p][q] += self.o_rdm[v][w] * (self.mol.tei[p][V][W][q] - 0.5 * self.mol.tei[p][V][q][W])

        return active_fock


    def _auxiliary_q_matrix(self):
        q_matrix = np.zeros([self.n_active_spat, self.n_spat])
        for v in range(self.n_active_spat):
            for m in range(self.n_spat):
                for w, W in enumerate(self.mol.active):
                    for x, X in enumerate(self.mol.active):
                        for y, Y in enumerate(self.mol.active):
                            q_matrix[v][m] += self.t_rdm[v][w][x][y] * self.mol.tei[m][W][Y][X]
        return q_matrix


    def _generalised_fock_matrix(self):
        generalised_fock = np.zeros([self.n_spat, self.n_spat])
        inactive_fock = self.mol._inactive_fock_matrix(self.mol.frozen)
        active_fock = self._active_fock_matrix()
        q_matrix = self._auxiliary_q_matrix()

        for I in self.mol.frozen:
            for n in range(self.n_spat):
                generalised_fock[I][n] = 2 * (inactive_fock[n][I] + active_fock[n][I])

        for v, V in enumerate(self.mol.active):
            for n in range(self.n_spat):
                generalised_fock[V][n] = q_matrix[v][n]
                for w, W in enumerate(self.mol.active):
                    generalised_fock[V][n] += inactive_fock[n][W] * self.o_rdm[v][w]
        return generalised_fock

    def orbital_gradient(self, generalised_fock):
        g = 2 * (generalised_fock.T - generalised_fock)
        return g

    def orbital_optimisation_step(self, threshold=1e-7, max_iter=10000):
        self.get_spat_1rdm()
        self.get_spat_2rdm()
        eta = 0.3
        oo_converged = False
        i = 0
        while not oo_converged and i < max_iter:
            F = self._generalised_fock_matrix()
            g = self.orbital_gradient(F)
            self.mol.ca = self.mol.ca @ expm(-eta*g)

            self.mol.hf_embedding(
                active=self.mol.active,
                frozen=self.mol.frozen,
            )
            print(self._energy_from_rdms())
            if np.linalg.norm(g) < 1e-4:
                oo_converged = True
            i += 1
        prev_energy = self.energy
        self._initialise_hamiltonian()
        self.energy = np.conj(self.wfn).T @ (self.h_mat @ self.wfn)
        if 0 < prev_energy - self.energy < threshold:
            converged = True
        else:
            converged = False
        # print(self.mol.ca)
        return False

    def _pack_orbital_gradient(self, gradient):
        """Convert an antisymmetric orbital-gradient matrix to a 1D array."""
        gradient = np.asarray(gradient)

        if gradient.ndim != 2 or gradient.shape[0] != gradient.shape[1]:
            raise ValueError("Orbital gradient must be a square 2D matrix.")

        indices = np.triu_indices(gradient.shape[0], k=1)
        return gradient[indices]

    def _unpack_orbital_gradient(self, vector, n_orbitals):
        """Convert independent upper-triangle values to an antisymmetric matrix."""
        vector = np.asarray(vector)
        indices = np.triu_indices(n_orbitals, k=1)

        if vector.size != len(indices[0]):
            raise ValueError(
                f"Expected {len(indices[0])} values, got {vector.size}."
            )

        gradient = np.zeros(
            (n_orbitals, n_orbitals),
            dtype=vector.dtype,
        )
        gradient[indices] = vector
        gradient[(indices[1], indices[0])] = -vector

        return gradient

    # def _orbital_gradient_vector(self, params, C0):
    #     kappa = self._
    #     F = self._generalised_fock_matrix()
    #     g = self.orbital_gradient(F)
    #     g_vector = self._pack_orbital_gradient(g)
    #     return g_vector

    def _orbital_objective(self, params, C0):
        kappa = self._unpack_orbital_gradient(params, self.n_spat)

        self.mol.ca = C0 @ expm(kappa)

        self.mol.hf_embedding(
            active=self.mol.active,
            frozen=self.mol.frozen
        )
        self.energy = self._energy_from_rdms()
        return self.energy

    def run_oo(self, method="L-BFGS-B", callback=None):
        self.get_spat_1rdm()
        self.get_spat_2rdm()
        initial_params = np.zeros(self.n_spat * (self.n_spat - 1) // 2)
        C0 = self.mol.ca.copy()
        energy, optimised_vector, extra = optimize(
            self._orbital_objective,
            initial_params,
            args=(C0,),
            method=method,
            callback=callback,
            jac=None
        )


    def check_numerical_orb_gradient(self):
        C0 = self.mol.ca.copy()

        p, q = 2, 0
        eps = 1e-5

        K = np.zeros((self.n_spat, self.n_spat))
        K[p, q] = 1.0
        K[q, p] = -1.0

        # + epsilon
        self.mol.ca = C0 @ expm(-eps * K)
        self.mol.hf_embedding(
            active=self.mol.active,
            frozen=self.mol.frozen
        )
        self._initialise_hamiltonian()

        E_plus = np.vdot(
            self.wfn,
            self.h_mat @ self.wfn
        ).real

        # - epsilon
        self.mol.ca = C0 @ expm(eps * K)
        self.mol.hf_embedding(
            active=self.mol.active,
            frozen=self.mol.frozen
        )
        self._initialise_hamiltonian()

        E_minus = np.vdot(
            self.wfn,
            self.h_mat @ self.wfn
        ).real

        numerical_grad = (E_plus - E_minus) / (2 * eps)
        print(numerical_grad)
        self.mol.ca = C0

        



class Ansatz_tUPS(UPSAnsatz):
    def __init__(self, mol, layers=1, use_first_singles=True,**kwargs):
        self.layers = layers
        self.use_first_singles = use_first_singles
        super().__init__(mol, spin_preserving=True, **kwargs)


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

