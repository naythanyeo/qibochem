
import openfermion
import numpy as np
from qibo import Circuit, gates
"""
Utility functions specific to UCC ansatz
"""

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

def excitation2qubit_observable(excitation, ferm_qubit_map='jw'):
    """
    Function to convert excitation into a qubit operator
    """
    # Group the excitation into open fermion form
    holes, particles = excitation
    fermion_operator_str = "".join(
        [f"{p}^ " for p in particles] +
        [f"{h} " for h in holes]
    )
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
    return qubit_ucc_operator

def ucc_circuit(n_qubits, excitation, theta=0.0, trotter_steps=1, ferm_qubit_map='jw'):
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
    # Check validity of excitation
    assert len(excitation[0]) == len(excitation[1]), f"{excitation} must have same number of holes and particles"
    # Get the qubit operator
    qubit_ucc_operator = excitation2qubit_observable(excitation, ferm_qubit_map)
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


def mp2_amplitude(excitation, orbital_energies, tei):
    r"""
    Calculate the MP2 guess amplitude for a single UCC circuit: 0.0 for a single excitation.
        for a double excitation (In SO basis): :math:`t_{ij}^{ab} = (g_{ijab} - g_{ijba}) / (e_i + e_j - e_a - e_b)`

    Args:
        excitation: Iterable of spin-orbitals representing a excitation. Must have either 2 or 4 elements exactly,
            representing a single or double excitation respectively.
        orbital_energies: eigenvalues of the Fock operator, i.e. orbital energies
        tei: Two-electron integrals in MO basis and second quantization notation

    Returns:
        MP2 guess amplitude (float)
    """
    # Check validity of excitation
    assert len(excitation[0]) == len(excitation[1]), f"{excitation} must have same number of holes and particles"
    # If single excitation, can just return 0.0 directly
    if len(excitation[0]) == 1:
        return 0.0
    # Convert orbital indices to be in MO basis
    mo_orbitals = [orbital // 2 for orbital in excitation]
    # Numerator: g_ijab - g_ijba
    g_ijab = (
        tei[tuple(mo_orbitals)]  # Can index directly using the MO TEIs
        if (excitation[0] + excitation[3]) % 2 == 0 and (excitation[1] + excitation[2]) % 2 == 0
        else 0.0
    )
    g_ijba = (
        tei[tuple(mo_orbitals[:2] + mo_orbitals[2:][::-1])]  # Reverse last two terms
        if (excitation[0] + excitation[2]) % 2 == 0 and (excitation[1] + excitation[3]) % 2 == 0
        else 0.0
    )
    numerator = g_ijab - g_ijba
    # Denominator is directly from the orbital energies
    # Guards added against denominator and amplitude to catch nan or inf values
    denominator = sum(orbital_energies[mo_orbitals[:2]]) - sum(orbital_energies[mo_orbitals[2:]])
    if abs(denominator) < 1e-12:
        return 0.0
    amplitude = numerator / denominator
    if np.isnan(amplitude):
        return 0.0
    return amplitude