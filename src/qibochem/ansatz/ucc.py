"""
Circuit representing the Unitary Coupled Cluster ansatz in quantum chemistry
"""

from dataclasses import dataclass, field

import numpy as np
import openfermion
from qibo import Circuit
from qibo.optimizers import optimize

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.ansatz.excitation_util import (generate_excitations, filter_OV_transition, filter_paired, 
                                             filter_spin, group_spin_adapt, filter_cross_excitations)
from qibochem.ansatz.ucc_util import (ucc_circuit, mp2_amplitude, excitation2qubit_observable)
from qibochem.driver.observables import qubit_operator2observable

"""
Use a UCCAnsatz class instead to create the UCC ansatz circuit and run VQE optimisation
This class does not use qibo.VQE so that the circuit parameters can be better constrained 
More overhead expected compared to just optimising the circuit parameters but important 
for accurate ansatz construction. 
The class uses ucc_circuit and above helper functions to build and optimise VQE circuit
Ansatz construction is more easily done here also by defining param_excitations
"""

@dataclass
class UCCAnsatz:
    mol: object
    final_params: dict | None = None
    ferm_qubit_map: str = "jw"
    trotter_steps: int = 1
    include_hf: bool = True
    use_mp2_guess: bool = True
    param_excitations: dict = field(init=False)
    param_map: dict = field(init=False)

    def __post_init__(self):
        # Follow the active space definitions from the mol object
        self.n_elec = (self.mol.nelec
                       if self.mol.n_active_e is None
                       else self.mol.n_active_e)
        # SPIN ORBITALS
        self.n_orbs = (self.mol.nso 
                       if self.mol.n_active_orbs is None 
                       else self.mol.n_active_orbs)
        """
        Here the param_excitations is a unique dictionary that maps to each ansatz 
        param_excitaitons format {"s0": [((0,), (2,)), "s1": [((1,), (3,))....]}
        parm_map is a dictionary that maps each parameter to the coefficients of the 
        corresponding CIRCUIT parameters.
        """

        self.param_excitations = self.excitations() # Ansatz specific excitations, defined at subclass 
        self.param_map = self._get_param_map()
        self.param_names = list(self.param_excitations.keys())

        # Define initial parameters, if mp2 is false then default to zeros 
        # Here the parameters are stored as a dictionary {s0: 0.3, s1: 0.2...}
        if self.use_mp2_guess:
            self.initial_params = {name: mp2_amplitude(excitations[0], self.mol.eps, self.mol.tei)
                                   for name, excitations in self.param_excitations.items()}
        else:
            self.initial_params = {name: 0.0 for name in self.param_names}

        # Build the initial circuit with initial parameters
        # After optimisation, then final parameters will be set and final_circuit will be built 
        self.circuit = self._build_circuit(self.initial_params)
        self.final_circuit = None

        # Here, if the final params is already set, then final_circuit will be built 
        # If input final_params, ansatz object will treat it as optimised coefficients 
        if self.final_params is not None:
            self._set_params(self.final_params)
            self.final_circuit = self.circuit.copy(deep=True)
        # Here if you set the final parameters, should be able to call directly 
        # VQE_circuit = UCC_Ansatz.final_circuit --> Pass this circuit into QSE / others 

    def excitations(self):
        raise NotImplementedError(
            "Cannot call UCCAnsatz directly. Use a concrete ansatz class such as UCCSD, UCCGSD, or UCCSDSinglet."
        )

    def _generate_ansatz_excitations(self, rank, generalised, spin_conserve, paired, spin_adapt,
                                     parallel_excitations=True):
        """
        Helper function for self.excitations()
        This function allows for the mix and match of different anstaz filters for each subclass
        """
        excitations = generate_excitations(rank, self.n_orbs)
        if not generalised:
            excitations = filter_OV_transition(excitations, self.n_elec, self.n_orbs)
        elif parallel_excitations==True: # Default restriction to match Inquanto for restricted generalised ansatz
            excitations = filter_cross_excitations(excitations)
        if spin_conserve:
            excitations = filter_spin(excitations)
        if paired:
            excitations = filter_paired(excitations)
        if spin_adapt:
            grouped_excitations = group_spin_adapt(excitations)
        else:
            # Group the excitations regardless to preserve data structure
            grouped_excitations = [[excitation] for excitation in excitations] 
        # Sort the groups by the FIRST group term, holes first, then particles
        sorted_groups = sorted(grouped_excitations,
                               key = lambda group_excitation: (group_excitation[0][0], # Sort by holes
                                                               group_excitation[0][1])) # Sort by particles
        rank_map = {1: "s", 2: "d", 3: "t", 4: "q"}
        label = (f"{rank_map[rank]}"
                 f"{'g' if generalised else ''}"
                 f"{'s' if spin_adapt else ''}"
                 f"{'p' if paired else ''}")
        # Label the excitations from before with standardise labels
        return {f"{label}{count}": excitation
                for count, excitation in enumerate(sorted_groups)}

    def _build_circuit(self, param_values):
        # Default should be true to include the HF state 
        if self.include_hf:
            circuit = hf_circuit(self.n_orbs, self.n_elec, ferm_qubit_map=self.ferm_qubit_map)
        else:
            circuit = Circuit(self.n_orbs)
        # Add on the Gates for every ANSATZ Parameter 
        for name in self.param_names:
            theta = param_values[name]
            for excitation in self.param_excitations[name]:
                circuit += ucc_circuit(self.n_orbs, excitation, theta=theta,
                                       trotter_steps=self.trotter_steps,
                                       ferm_qubit_map=self.ferm_qubit_map)
        return circuit

    # Function to map the param_excitations into the corresponding CIRCUIT parameters
    # Outputs the coefficients of each circuit parameter relative to the ansatz parameters
    def _get_param_map(self):
        """
        Function to map the param_excitations into the corresponding CIRCUIT parameters
        Outputs the coefficients of each circuit parameter relative to the ansatz parameters
        """
        param_map = {}
        for name, excitations in self.param_excitations.items():
            param_map[name] = []
            # Excitations can be a list of excitations with grouped paramaeters (tied together) for spin adapt ansatz
            for excitation in excitations:
                # Convert excitation into qubit operator
                qubit_ucc_operator = excitation2qubit_observable(excitation, ferm_qubit_map=self.ferm_qubit_map)
                for _ in range(self.trotter_steps):
                    for raw_pauli_string in qubit_ucc_operator.get_operators():
                        ((_pauli_ops, coeff),) = raw_pauli_string.terms.items()
                        gate_coeff = np.real(-2.0 * (-1.0j * coeff) / self.trotter_steps)
                        param_map[name].append(gate_coeff)
        return param_map

    # Get the circuit parameters given parameters in the dictionary form
    def _get_circuit_parameters(self, param_values):
        circuit_params = []
        for name in self.param_names:
            theta = param_values[name]
            for coeff in self.param_map[name]:
                circuit_params.append(coeff * theta)
        return circuit_params

    # Update circuit 
    def _set_params(self, param_values):
        # Param_values is a DICTIONARY of ANSATZ parameters 
        circuit_parameters = self._get_circuit_parameters(param_values) # Maps to circuit parameters via param_map
        self.circuit.set_parameters(circuit_parameters)
    
    # Convert vector into parameter dictionary 
    def _vector2params(self, theta_vector):
        return {name: theta for name, theta in zip(self.param_names, theta_vector)}

    # Function for the optimiser to reconstruct the circuit and get expectation value of the hamiltonian
    def _get_energy(self, theta_vector):
        self._set_params(self._vector2params(theta_vector))
        return self.protocol.evaluate(self.circuit, self.hamiltonian)
    

    """
    Function to run VQE optimisation
    Build on top of qibo.optimize function
    Finds the optimal parameters and constructs the final circuit
    """

    def run_vqe(self, protocol, method="BFGS", **optimizer_kwargs):
        # Get the bitmask hamiltonian from qubit hamiltonian to run VQE
        self.hamiltonian = qubit_operator2observable(self.mol.hamiltonian("qubit", ferm_qubit_map=self.ferm_qubit_map))
        # Convert the initial parameters (dictionary) into a vector form for the optimizer
        initial_vector = np.array([self.initial_params[name] for name in self.param_names])
        # The vector that optimize uses is length equal to number of ANSATZ parameters 
        # The variable circuit_params contains the FULL CIRCUIT parameters 
        self.protocol = protocol # Set self attribute protocol for get_energy to run
        vqe_energy, optimised_vector, extra = optimize(self._get_energy, initial_vector, 
                                                       method=method, **optimizer_kwargs)
        # Convert the outut optimised vector back into parameter dictionary form
        self.final_params = self._vector2params(optimised_vector)
        # Set the circuit parameters to optimised parameters and build final circuit
        self._set_params(self.final_params)
        self.final_circuit = self.circuit.copy()
        self.vqe_energy = vqe_energy
        self.vqe_result = extra

        return vqe_energy, self.final_params, self.final_circuit
    
"""
ALL UCC ANSATZ SUBCLASSES
"""
class Ansatz_UCCSD(UCCAnsatz):
    def excitations(self):
        singles_excitations = self._generate_ansatz_excitations(rank=1, generalised=False, spin_conserve=True, 
                                                                paired=False, spin_adapt=False)
        doubles_excitations = self._generate_ansatz_excitations(rank=2, generalised=False, spin_conserve=True, 
                                                                paired=False, spin_adapt=False)
        return {**singles_excitations, **doubles_excitations}

class Ansatz_UCCSDSinglet(UCCAnsatz):
    def excitations(self):
        singles_excitations = self._generate_ansatz_excitations(rank=1, generalised=False, spin_conserve=True, 
                                                                paired=False, spin_adapt=True)
        doubles_excitations = self._generate_ansatz_excitations(rank=2, generalised=False, spin_conserve=True, 
                                                                paired=False, spin_adapt=True)
        return {**singles_excitations, **doubles_excitations}

class Ansatz_UCCGSD(UCCAnsatz):
    def excitations(self):
        singles_excitations = self._generate_ansatz_excitations(rank=1, generalised=True, spin_conserve=True, 
                                                                paired=False, spin_adapt=False)
        doubles_excitations = self._generate_ansatz_excitations(rank=2, generalised=True, spin_conserve=True, 
                                                                paired=False, spin_adapt=False)
        return {**singles_excitations, **doubles_excitations}

class Ansatz_UCCD(UCCAnsatz):
    def excitations(self):
        doubles_excitations = self._generate_ansatz_excitations(rank=2, generalised=False, spin_conserve=True, 
                                                                paired=False, spin_adapt=False)
        return doubles_excitations

class Ansatz_UCCDSinglet(UCCAnsatz):
    def excitations(self):
        doubles_excitations = self._generate_ansatz_excitations(rank=2, generalised=False, spin_conserve=True, 
                                                                paired=False, spin_adapt=True)
        return doubles_excitations

class Ansatz_kUpCCGSDSinglet(UCCAnsatz):
    def __init__(self, mol, k=1, **kwargs):
        self.k = k
        super().__init__(mol, **kwargs)

    def excitations(self):
        param_excitations = {}
        for iteration in range(self.k):
            singles_excitations = self._generate_ansatz_excitations(rank=1, generalised=True, spin_conserve=True, 
                                                                paired=False, spin_adapt=True)
            doubles_excitations = self._generate_ansatz_excitations(rank=2, generalised=True, spin_conserve=True, 
                                                                paired=True, spin_adapt=True)
            for key, value in singles_excitations.items():
                param_excitations[f"{key}_k{iteration}"] = value
            for key, value in doubles_excitations.items():
                param_excitations[f"{key}_k{iteration}"] = value
        return param_excitations