"""
Circuit representing the Unitary Coupled Cluster ansatz in quantum chemistry
"""

from dataclasses import dataclass, field

import numpy as np
import openfermion
from qibo import Circuit, gates

from qibochem.ansatz.hf_reference import hf_circuit
from qibochem.ansatz.util import generate_excitations, mp2_amplitude, sort_excitations


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
    # 3. Change back to the Z basis
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


def ucc_ansatz(
    molecule,
    excitation_level=None,
    excitations=None,
    thetas=None,
    trotter_steps=1,
    ferm_qubit_map=None,
    include_hf=True,
    use_mp2_guess=True,
):
    r"""
    Convenience function for buildng a circuit corresponding to the UCC ansatz with multiple excitations for a
    given :class:`qibochem.driver.Molecule`. If no excitations are given, it defaults to returning the full UCCSD
    circuit ansatz.

    Args:
        molecule (:class:`qibochem.driver.Molecule`): The molecule of interest.
        excitation_level (str): Include excitations up to how many electrons, i.e. ``"S"`` or ``"D"``.
            Ignored if ``excitations`` argument is given. Default: ``"D"``, i.e. double excitations
        excitations (list): List of excitations (e.g. ``[[0, 1, 2, 3], [0, 1, 4, 5]]``) used to build the
            UCC circuit. Overrides the ``excitation_level`` argument
        thetas (list): Parameters for the excitations. Default value depends on the ``use_mp2_guess`` argument.
        trotter_steps (int): Number of Trotter steps; i.e. number of times the UCC ansatz is applied with
            :math:`\theta = \theta` / ``trotter_steps``. Default: 1
        ferm_qubit_map (str): fermion-to-qubit transformation. Default: Jordan-Wigner (``"jw"``)
        include_hf (bool): Whether or not to start the circuit with a Hartree-Fock circuit. Default: ``True``
        use_mp2_guess (bool): Whether to use MP2 amplitudes or a numpy zero array as the initial guess parameter.
            Default: ``True``, will use the MP2 amplitudes as the initial guess parameters

    Returns:
        :class:`qibo.models.circuit.Circuit`: Circuit corresponding to an UCC ansatz
    """
    # Get the number of electrons and spin-orbitals from the molecule argument
    n_elec = molecule.nelec if molecule.n_active_e is None else molecule.n_active_e
    n_orbs = molecule.nso if molecule.n_active_orbs is None else molecule.n_active_orbs

    # Define the excitation level to be used if no excitations given
    if excitations is None:
        excitation_levels = ("S", "D", "T", "Q")
        if excitation_level is None:
            excitation_level = "D"
        else:
            # Check validity of input
            assert (
                len(excitation_level) == 1 and excitation_level.upper() in excitation_levels
            ), "Unknown input for excitation_level"
            # Note: Probably don't be too ambitious and try to do 'T'/'Q' at the moment...
            if excitation_level.upper() in ("T", "Q"):
                raise NotImplementedError("Cannot handle triple and quadruple excitations!")
        # Get the (largest) order of excitation to use
        excitation_order = excitation_levels.index(excitation_level.upper()) + 1

        # Generate and sort all the possible excitations
        excitations = []
        for order in range(excitation_order, 0, -1):  # Reversed to get higher excitations first
            excitations += sort_excitations(generate_excitations(order, range(0, n_elec), range(n_elec, n_orbs)))
    else:
        # Some checks to ensure the given excitations are valid
        assert all(len(_ex) % 2 == 0 for _ex in excitations), "Excitation with an odd number of elements found!"

    # Check if thetas argument given, define to be all zeros if not
    # Number of thetas here should match the number of excitations when circuit is initialised
    # Number of parameters previously correspond to the otal circuit parameters 
    # that are updated and set with VQE object not when its initialised
    if thetas is None:
        if use_mp2_guess:
            thetas = np.array([mp2_amplitude(excitation, molecule.eps, molecule.tei) for excitation in excitations])
        else:
            thetas = np.zeros(len(excitations))
    else:
        # Check that number of circuit variables (i.e. thetas) matches the number of circuit parameters
        assert len(thetas) == len(excitations), "Number of input parameters doesn't match the number of circuit parameters!"

    # Build the circuit
    if include_hf:
        circuit = hf_circuit(n_orbs, n_elec, ferm_qubit_map=ferm_qubit_map)
    else:
        circuit = Circuit(n_orbs)
    for excitation, theta in zip(excitations, thetas):
        circuit += ucc_circuit(n_orbs, excitation, theta, trotter_steps=trotter_steps, ferm_qubit_map=ferm_qubit_map)
    return circuit



sample_uccsd_param_excitations = {
    "d0": [(0, 1, 2, 3)],
    "s1": [(0, 2)],
    "s2": [(1, 3)],
}


@dataclass
class UCCAnsatz:
    mol: object
    ansatz_type: str = "UCCSD"
    final_params: dict | None = None
    ferm_qubit_map: str = "jw"
    trotter_steps: int = 1
    include_hf: bool = True
    use_mp2_guess: bool = True
    param_excitations: dict = field(init=False)

    def __post_init__(self):
        # Follow the active space definitions from the mol object
        self.n_elec = (
            self.mol.nelec
            if self.mol.n_active_e is None
            else self.mol.n_active_e
        )

        self.n_orbs = (
            self.mol.nso
            if self.mol.n_active_orbs is None
            else self.mol.n_active_orbs
        )
        """
        Here the param_excitations is a unique dictionary that maps to each ansatz 
        A function _ansatz2param_excitations will be defined to generate these 
        Every new ansatz just needs to define the new rules 
        For now use a sample uccsd_param_excitation dictionary to test the workflow
        param_excitaitons format {"s0": [(0, 2), "s1": [(1, 3)....]}
        Each parameter will have a list of excitations that it maps to 
        """

        self.param_excitations = sample_uccsd_param_excitations
        self.param_names = list(self.param_excitations.keys())

        # Define initial parameters, if mp2 is false then default to zeros 
        # Here the parameters are stored as a dictionary {s0: 0.3, s1: 0.2...}
        if self.use_mp2_guess:
            self.initial_params = {
                name: mp2_amplitude(excitations[0], self.mol.eps, self.mol.tei)
                for name, excitations in self.param_excitations.items()
            }
        else:
            self.initial_params = {name: 0.0 for name in self.param_names}

        # Build the initial circuit with initial parameters
        # After optimisation, then final parameters will be set and final_circuit will be built 
        self.circuit = self._build_circuit(self.initial_params)
        self.final_circuit = None

        # Here, if the final params is already set, then final_circuit will be built 
        # If input final_params, ansatz object will treat it as optimised coefficients 
        # Important that the final parameters input must match the param_excitations
        # Meaning that finalised parameters can only be used for same ansatz type 
        if self.final_params is not None:
            self.final_circuit = self._build_circuit(self.final_params)
            # Add in a print message here to measure and print final VQE energy with final parameters? 

        # Here if you set the final parameters, should be able to call directly 
        # VQE_circuit = UCC_Ansatz.final_circuit --> Pass this circuit into QSE / others 

    def _build_circuit(self, param_values):
        # Default should be true to include the HF state 
        if self.include_hf:
            circuit = hf_circuit(self.n_orbs, self.n_elec, ferm_qubit_map=self.ferm_qubit_map)
        else:
            circuit = qibo.Circuit(self.n_orbs)
        # Add on the Gates for every ANSATZ Parameter 
        # All the excitations will be mapped 
        # Param_excitaiton will be constructed based on the ansatz 
        for name in self.param_names:
            theta = param_values[name]
            for excitation in self.param_excitations[name]:
                circuit += ucc_circuit(
                    self.n_orbs,
                    excitation,
                    theta=theta,
                    trotter_steps=self.trotter_steps,
                    ferm_qubit_map=self.ferm_qubit_map,
                )
        return circuit
    
    
