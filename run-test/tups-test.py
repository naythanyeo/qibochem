from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data" / "output"

from qibochem.driver import Molecule
# from qibochem.ansatz.ucc import Ansatz_tUPS
from qibochem.ansatz.ups import Ansatz_tUPS
from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.scripts.script_utils import load_molecule

iteration=0
def callback(param):
    global iteration
    global tups

    iteration += 1
    print(f"Iteration {iteration}")
    print("Parameters:", param)
    # print("Energy:", tups.energy)

SCRIPT_DIR = Path(__file__).resolve().parent

sv_protocol = StateVectorProtocol()

num_active_e = 6
num_active_o = 6
molecule_name = 'h6'
n_layers = 1
input_dir = "./data/"
mol = load_molecule(
            SCRIPT_DIR / 'data' / f"{molecule_name}.xyz",
            num_active_e,
            num_active_o
        )


# initial_guess = np.fromstring(array_text, sep=' ')
tups = Ansatz_tUPS(mol=mol, layers=2, use_random_angles=False, use_mp2_guess=False, use_projection=True, use_mat_mul=True)

tups.run_vqe(protocol=sv_protocol, fast=True, callback=callback, method='L-BFGS-B', fast_mat_mul=True)
print(tups.param_names)
print(tups.energy)
