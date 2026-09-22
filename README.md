# Qibochem-QSE

A research fork of [Qibochem](https://github.com/qiboteam/qibochem),
developed to investigate quantum subspace expansion (QSE).

Qibochem is a plugin for [Qibo](https://github.com/qiboteam/qibo)
for quantum chemistry simulations. The original implementation and
documentation are available from the
[upstream repository](https://github.com/qiboteam/qibochem) and
[Qibochem documentation](https://qibo.science/qibochem/stable/).

## Research-fork features
The fork inherits most features from qibochem. 
Refer to their original documentation for details on the Molecule classes and Qibo backend.
This fork mainly has the following new features:

- **UCC ansatz framework:** parameter tying implemented between excitation
  operators and their circuit rotations. More broad ansatz families implemented
- **Fast VQE evaluation:** direct statevector Pauli rotations during
  optimisation, with the optimised Qibo circuit returned afterwards.
- **QSE computables:** construct projected Hamiltonian, overlap, and
  spin matrices using configurable excitation generators.
- **Measurement protocols:** statevector expectations and finite-shot
  sampling with global qubit-wise commuting grouping across projected
  observables. This fork uses bitmasks observables as opposed to SymPy from the original implementation. 

## Installation

Requires Python 3.11–3.13. Install the research branch directly from GitHub:

```bash
python -m pip install "git+https://github.com/naythanyeo/qibochem-qse.git@main-qse"
```

The distribution is named `qibochem-qse`, but the Python import remains:

```python
import qibochem
```

## Tutorials

The following notebooks provide minimal working examples:

- [Statevector QSE](tutorials/qse_full_run.ipynb):
  prepare a UCCSD reference, solve the QSE problem, and evaluate spin
  expectations.
- [Finite-shot QSE](tutorials/qse_shot_noise.ipynb):
  sample projected matrices and compare energy errors across repeats
  and shot counts.

## Citation

Please follow the citation instructions in the
[original Qibochem documentation](https://qibo.science/qibochem/stable/)
and cite the relevant upstream work when using this fork.
