
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
    """
    Calculate MP2 guess amplitude for one excitation.
    excitation format: (holes, particles)
    For doubles:
        holes     = (i, j)
        particles = (a, b)
    MP2:
        t_ij^ab = (g_ijab - g_ijba) / (eps_i + eps_j - eps_a - eps_b)
    """
    holes, particles = excitation
    if len(holes) != len(particles):
        raise ValueError(f"{excitation} must have same number of holes and particles.")
    # MP2 singles amplitudes are zero in canonical HF orbitals.
    if len(holes) == 1:
        return 0.0
    if len(holes) != 2:
        raise ValueError("MP2 amplitude is only implemented for singles and doubles.")

    i, j = holes
    a, b = particles

    i_mo, j_mo = i // 2, j // 2
    a_mo, b_mo = a // 2, b // 2

    # Direct term: <ij|ab>
    # By default this will be true if using spin adapt ansatz
    if (i % 2 == a % 2) and (j % 2 == b % 2):
        g_ijab = tei[i_mo, j_mo, a_mo, b_mo]
    else:
        g_ijab = 0.0
    # Exchange term: <ij|ba>
    # Only consider the exchange term if both electrons are same spin
    if (i % 2 == b % 2) and (j % 2 == a % 2):
        g_ijba = tei[i_mo, j_mo, b_mo, a_mo]
    else:
        g_ijba = 0.0
    numerator = g_ijab - g_ijba

    denominator = (orbital_energies[i_mo] + orbital_energies[j_mo] - 
                   orbital_energies[a_mo] - orbital_energies[b_mo])
    # Safeguards against div by 0
    if abs(denominator) < 1e-12:
        return 0.0
    amplitude = numerator / denominator
    if not np.isfinite(amplitude):
        return 0.0
    return amplitude