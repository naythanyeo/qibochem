"""
Circuit representing the Unitary Coupled Cluster ansatz in quantum chemistry
"""

from dataclasses import dataclass, field

import numpy as np
import openfermion
from qibo import Circuit, gates
from qibo.optimizers import optimize

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.ansatz.excitation_util import (generate_excitations, mp2_amplitude, filter_OV_transition, 
                                             filter_paired, filter_spin, group_spin_adapt, flatten_excitation,
                                             sort_excitations, filter_unique_generalised)


def expi_pauli(n_qubits, pauli_string, theta):
    """
    Build circuit representing exp(i*theta*pauli_string)

    Args:
        n_qubits: No. of qubits in the quantum circuit
        pauli_string: String in the format: ``"X0 Z1 Y3 X11"``
        theta: Real number

    Returns:
        circuit: Qibo Circuit object representing exp(i*theta*pauli_string)
    """
    # Split pauli_string into the old p_letters format
    pauli_ops = sorted(((int(_op[1:]), _op[0]) for _op in pauli_string.split()), key=lambda x: x[0])
    n_pauli_ops = len(pauli_ops)

    # Convert theta into a real number for applying with a RZ gate
    rz_parameter = -2.0 * theta

    # Generate the list of basis change gates using the pauli_ops list
    basis_changes = []
    for qubit, pauli_op in pauli_ops:
        if pauli_op == "Y":
            basis_changes.append(gates.S(qubit).dagger())
        if pauli_op not in ("I", "Z"):
            basis_changes.append(gates.H(qubit))

    # Build the circuit
    circuit = Circuit(n_qubits)
    # 1. Change to X/Y where necessary
    circuit.add(basis_changes)
    # 2. Add CNOTs to all pairs of qubits in pauli_ops, starting from the last letter
    circuit.add(gates.CNOT(pauli_ops[_i][0], pauli_ops[_i - 1][0]) for _i in range(n_pauli_ops - 1, 0, -1))
    # 3. Add RZ gate to last element of pauli_ops
    circuit.add(gates.RZ(pauli_ops[0][0], rz_parameter))
    # 4. Add CNOTs to all pairs of qubits in pauli_ops
    circuit.add(gates.CNOT(pauli_ops[_i + 1][0], pauli_ops[_i][0]) for _i in range(n_pauli_ops - 1))
    # 5. Change back to the Z basis
    circuit.add(_gate.dagger() for _gate in reversed(basis_changes))
    return circuit


def ucc_circuit(n_qubits, excitation, theta=0.0, trotter_steps=1, ferm_qubit_map=None):
    r"""
    Circuit corresponding to the unitary coupled-cluster ansatz for a single excitation

    Args:
        n_qubits (int): Number of qubits in the quantum circuit
        excitation (list): Iterable of orbitals involved in the excitation; must have an even number of elements
            E.g. ``[0, 1, 2, 3]`` represents the excitation of electrons in orbitals ``(0, 1)`` to ``(2, 3)``
        theta (float): UCC parameter. Defaults to 0.0
        trotter_steps (int): Number of Trotter steps; i.e. number of times the UCC ansatz is applied
            with :math:`\theta = \theta` / ``trotter_steps``. Default: 1
        ferm_qubit_map (str): Fermion-to-qubit transformation. Default is Jordan-Wigner (``"jw"``).

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit corresponding to a single UCC excitation
    """
    # Check size of orbitals input
    n_orbitals = len(excitation)
    assert n_orbitals % 2 == 0, f"{excitation} must have an even number of items"
    # Reverse sort orbitals to get largest-->smallest
    sorted_orbitals = sorted(excitation, reverse=True)

    # Define default mapping
    if ferm_qubit_map is None:
        ferm_qubit_map = "jw"

    # Define the UCC excitation operator corresponding to the given list of orbitals
    fermion_op_str_template = f"{(n_orbitals//2)*'{}^ '}{(n_orbitals//2)*'{} '}"
    fermion_operator_str = fermion_op_str_template.format(*sorted_orbitals)
    # Build the FermionOperator and make it unitary
    fermion_operator = openfermion.FermionOperator(fermion_operator_str)
    ucc_operator = fermion_operator - openfermion.hermitian_conjugated(fermion_operator)

    # Map the FermionOperator to a QubitOperator
    if ferm_qubit_map == "jw":
        qubit_ucc_operator = openfermion.jordan_wigner(ucc_operator)
    elif ferm_qubit_map == "bk":
        qubit_ucc_operator = openfermion.bravyi_kitaev(ucc_operator)
    else:
        raise KeyError("Fermon-to-qubit mapping must be either 'jw' or 'bk'")
    
    # Apply the qubit_ucc_operator 'trotter_steps' times:
    assert trotter_steps > 0, f"{trotter_steps} must be > 0!"
    circuit = Circuit(n_qubits)
    for _i in range(trotter_steps):
        # Use the get_operators() generator to get the list of excitation operators
        for raw_pauli_string in qubit_ucc_operator.get_operators():
            # Convert each operator into a string and get the associated coefficient
            ((pauli_ops, coeff),) = raw_pauli_string.terms.items()  # Unpack the single-item dictionary
            pauli_string = " ".join(f"{pauli_op[1]}{pauli_op[0]}" for pauli_op in pauli_ops)
            # Build the circuit and add it on
            _circuit = expi_pauli(
                n_qubits, pauli_string, -1.0j * coeff * theta / trotter_steps
            )  # Divide imag. coeff by 1.0j
            circuit += _circuit
    return circuit

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
    # Maybe modify in the future to allow users to add in own excitation parameters
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
        param_excitaitons format {"s0": [(0, 2), "s1": [(1, 3)....]}
        parm_map is a dictionary that maps each parameter to the coefficients of the 
        corresponding excitations in the circuit.
        """

        self.param_excitations = self.excitations()
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

    def _generate_ansatz_excitations(self, rank, generalised, spin_conserve, paired, spin_adapt):
        excitations = generate_excitations(rank, self.n_orbs)
        if generalised:
            excitations = filter_unique_generalised(excitations)
        else:
            excitations = filter_OV_transition(excitations, self.n_elec, self.n_orbs)
        if spin_conserve:
            excitations = filter_spin(excitations)
        if paired:
            excitations = filter_paired(excitations)
        if spin_adapt:
            grouped_excitations = group_spin_adapt(excitations)
        else:
            # Group the excitations regardless to preserve data structure
            grouped_excitations = [[excitation] for excitation in excitations] 
        # Flatten excitations to pass into UCC_Circuit 
        flattened_grouped_excitations = [[flatten_excitation(excitation) for excitation in group]
                                            for group in grouped_excitations]
        # Sort the groups 
        sorted_flattened_groups = sort_excitations(flattened_grouped_excitations)
        rank_map = {1: "s", 2: "d", 3: "t", 4: "q"}
        label = (f"{rank_map[rank]}"
                 f"{'g' if generalised else ''}"
                 f"{'s' if spin_adapt else ''}"
                 f"{'p' if paired else ''}")
        # Sort the excitations 
        return {f"{label}{count}": excitation
                for count, excitation in enumerate(sorted_flattened_groups)}

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
        param_map = {}
        for name, excitations in self.param_excitations.items():
            param_map[name] = []
            # Excitations can be a list of excitations with grouped paramaeters (tied together) for spin adapt ansatz
            for excitation in excitations:
                # Excitation is one flattened list of excitations 
                n_orbitals = len(excitation)
                sorted_orbitals = sorted(excitation, reverse=True)
                # Create the anti hermitian operator string 
                fermion_op_str_template = f"{(n_orbitals // 2) * '{}^ '}{(n_orbitals // 2) * '{} '}"
                fermion_operator_str = fermion_op_str_template.format(*sorted_orbitals)
                fermion_operator = openfermion.FermionOperator(fermion_operator_str)
                ucc_operator = fermion_operator - openfermion.hermitian_conjugated(fermion_operator)
                if self.ferm_qubit_map == "jw":
                    qubit_ucc_operator = openfermion.jordan_wigner(ucc_operator)
                elif self.ferm_qubit_map == "bk":
                    qubit_ucc_operator = openfermion.bravyi_kitaev(ucc_operator)
                else:
                    raise KeyError("Fermon-to-qubit mapping must be either 'jw' or 'bk'")
                # Double check this for other trotter steps ?? 
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
        self.hamiltonian = self.mol.hamiltonian("sym", ferm_qubit_map=self.ferm_qubit_map)
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
                                                                paired=False, spin_adapt=False)
            doubles_excitations = self._generate_ansatz_excitations(rank=2, generalised=True, spin_conserve=True, 
                                                                paired=True, spin_adapt=False)
            for key, value in singles_excitations.items():
                param_excitations[f"{key}_k{iteration}"] = value
            for key, value in doubles_excitations.items():
                param_excitations[f"{key}_k{iteration}"] = value
        return param_excitations