"""
Functions for obtaining the expectation value given a final state and observable 
These observables will be in the form of bitmasks rather than sympy as its faster 
Rather than using the default qibo symbolic hamiltonian, this uses a bitmask for all evaluations 

A sample pauli string would be of the form {(bitmask_x, bitmask_y, bitmask_z): coeff}
Each bitmask_key is 3 bitmask, each representing X Y and Z gates 
A bitmask is just a binary form of the qubit position 
Eg X0 Y1 X2 Z3 for a system with 4 qubits will have: 
bitmask_x = 0b0101
bitmask_y = 0b0010
bitmask_z = 0b1000 
So the 0th qubit is counted from the right 

This format makes evaluating gate operations very cheap
X gate (Flip): 
X|0> = |1> 
X|1> = |0>

Y gate (Flip and Sign, Y=iXZ): 
Y|0> = i|1> 
Y|1> = -i|0>

Z gate (Sign): 
Z|0> = |0> 
Z|1> = -|1>

So given any basis state eg |1011>, first settle all the bit flips (check X and Y)
Build flip mask (X and Y masks) --> apply XOR (^)--> Flips if X and Y gate is present 
Binary structure allows for each bit to be compared and fliped directly without repeated 
symbolic comparison and rules, directly apply XOR is faster
(If flipmask is 1, ie X or Y gate present, then flip original state, else remain original)

After that count the phases and signs with bitmask AND (&) basis. 
Count Y gate and Z gate OCCUPIED --> (-1)**count
Count Y gate total number (including unoccupied) --> i**count
"""

import qibo
import numpy as np

def bitmask_expectation(final_state, bitmasks):
    """"
    Function that calculates the expectation value of a final state from a bitmask input 
    Protocol can evaluate later on the expectation of multiple observables with coefficients 
    This function uses numpy vectorised form 

    final_state: [c1, c2, c3 ...] (numpydarray), whereby they are coefficients of the wavefunction amplitude 
    psi = c1|00> + c2|01> + c3|10> ..., each basis state is basically binary form of 012345... 
    using numpy arange forms an array thats basically just (0, 1, 2, 3 ...)
    Evaluating each basis state index with XOR and & operations directly with bitmasks because internally 
    those integers are also just binary. Use numpy vectors so it uses C backend which is probably faster

    bitmasks = (bitmask_x, bitmask_y, bitmask_z)
    """
    basis_states = np.arange(final_state.size)
    x_mask, y_mask, z_mask = bitmasks
    flip_mask = x_mask | y_mask # Select both X and Y gates
    flipped_states = basis_states ^ flip_mask # Apply XOR on every basis state 
    # Calculate the phase flips 
    z_sign = (-1) ** np.bitwise_count(basis_states & z_mask).astype(int)
    y_sign = (-1) ** np.bitwise_count(basis_states & y_mask).astype(int)
    y_phase = (-1j) ** y_mask.bit_count()
    # Add in the coefficients for the transformed state 
    transformed_state = final_state[flipped_states]
    return np.vdot(final_state, (transformed_state*z_sign*y_sign*y_phase))
