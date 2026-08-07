from pathlib import Path
import numpy as np
from pyscf import gto, scf

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data" / "output"

from qibochem.driver import Molecule
from qibochem.ansatz.ucc import Ansatz_tUPS
from qibochem.ansatz.ups import UPSAnsatz
from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.scripts.script_utils import (
    Logger,
    get_vqe_circuit,
    load_molecule,
    parse_active_space,
    read_jsonl,
)

iteration=0
def callback(param):
    global iteration
    global tups

    iteration += 1
    print(f"Iteration {iteration}")
    print("Parameters:", param)
    print("Energy:", tups._get_energy(param))

SCRIPT_DIR = Path(__file__).resolve().parent

sv_protocol = StateVectorProtocol()

num_active_e = 6
num_active_o = 7
molecule_name = 'h6'
n_layers = 1
input_dir = "./data/"
mol = load_molecule(
            SCRIPT_DIR / 'data' / f"{molecule_name}.xyz",
            num_active_e,
            num_active_o
        )

# array_text = "0.        0.        0.       -0.00697  -0.351895  0.005431  0.        0.        0.        0.000686  1.254768 -0.001847 -0.179814  0.922066  0.154887"

# initial_guess = np.fromstring(array_text, sep=' ')
tups = UPSAnsatz(mol=mol, use_random_angles=False, use_mp2_guess=False, use_projection=True)
# print(tups.energy)
tups.run_vqe(protocol=sv_protocol, fast=True, callback=callback, method='BFGS')
print(tups.param_names)
quit()
tups = Ansatz_tUPS(mol, L=n_layers, use_mp2_guess=False, use_random_angles=False, initial_angles=None)
vqe_energy, vqe_params, final_circuit = tups.run_vqe(
    sv_protocol,
    method='BFGS',
    fast=True,
    callback=callback
)
print(vqe_energy)
print(vqe_params)