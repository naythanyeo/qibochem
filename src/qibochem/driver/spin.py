"""
Functions to construct Spin Operators

Sz = 1/2 (n(pa) - n(pb)), sum across all p (spatial orbitals)
Number Operator for all the alpha up spin electrons - all the beta down spin electrons, whereby
n(pa) = a(pa)+ a(pa) --> Destory then create
n(pb) = a(pb)+ a(pb) --> Destory then create

Define Raising and Lowering Operators
Raising 
S+ = Sx + iSy
S+|a> = 0, S+|b> = |a>
S+ = a(pa)+ a(pb), sum across all p (spatial orbitals) -> Destroy beta create alpha

Lowering
S- = Sx - iSy
S-|a> = |b>, S+|b> = 0
S+ = a(pb)+ a(pa), sum across all p (spatial orbitals) -> Destroy alpha create beta


S^2 = Sx^2 + Sy^2 + Sz^2
    = Sz^2 + 1/2(S+S- + S-S+)

These functions input number of SPATIAL orbitals, outputs a FERMIONIC representation of the operators
"""

import openfermion

def sz_operator(n_orbitals):
    op = openfermion.FermionOperator()

    for p in range(n_orbitals):
        alpha = 2 * p
        beta = 2 * p + 1
        # Number Operators
        op += 0.5 * openfermion.FermionOperator(((alpha, 1), (alpha, 0)))
        op += -0.5 * openfermion.FermionOperator(((beta, 1), (beta, 0)))

    return op


def _s_plus_operator(n_orbitals):
    op = openfermion.FermionOperator()

    for p in range(n_orbitals):
        alpha = 2 * p
        beta = 2 * p + 1
        # Create alpha, destroy beta
        op += openfermion.FermionOperator(((alpha, 1), (beta, 0)))

    return op

def _s_minus_operator(n_orbitals):
    op = openfermion.FermionOperator()

    for p in range(n_orbitals):
        alpha = 2 * p
        beta = 2 * p + 1
        # Create beta, destroy alpha
        op += openfermion.FermionOperator(((beta, 1), (alpha, 0)))

    return op

def s2_operator(n_orbitals):
    sz = sz_operator(n_orbitals)
    sp = _s_plus_operator(n_orbitals)
    sm = _s_minus_operator(n_orbitals)

    op = sz * sz + 0.5 * (sp * sm + sm * sp)
    op.compress() # Remove duplicate intermediate terms

    return op