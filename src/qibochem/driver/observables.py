"""
Lightweight observable class object for storage of observables 
In general using the hamiltonian as a observable is very heavy and impractical for repeated calculations 
For the general hamiltonian objects, usage of sympy is also very heavy and expensive 
This lightweight observable class is built on bitmask tuples which save memory and computational time 
Protocol class is also built on top of this observable class, ie all evaluations will use this bitmasks 
format to input observables and evaluate them. In general hamiltonian can exist in various forms but 
the efficient form for comparison and calculations is bitmasks, which significnantly out perform sympy objects

General structure of observable 
constant: complex number 
terms: {(bitmask x, bitmask y, bitmask z): coeff}

Here we first include helper functions to convert the qubit form of observables into the bitmask form

"""

import dataclass

@dataclass(slots=True)
class Observable:
    constant: complex
    terms: dict