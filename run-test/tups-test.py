from pathlib import Path
import numpy as np


from qibochem.ansatz.ups import Ansatz_tUPS
from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.scripts.script_utils import load_molecule

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data" / "output"

np.set_printoptions(precision=5, suppress=True)

vqe_iteration=0
oo_iteration=0
def vqe_callback(param):
    global vqe_iteration
    global tups

    vqe_iteration += 1
    print(f"VQE iteration {vqe_iteration}")
    print("Parameters:", param)
    print("Energy:", tups.energy)

def oo_callback(param):
    global oo_iteration
    global tups

    oo_iteration += 1
    print(f"OO iteration {oo_iteration}")
    print("Parameters:", param)
    print("Energy:", tups.energy)


SCRIPT_DIR = Path(__file__).resolve().parent

sv_protocol = StateVectorProtocol()

num_active_e = 6
num_active_o = 6
molecule_name = 'h6'
n_layers = 1
input_dir = "./data/"
mol = load_molecule(
            SCRIPT_DIR / 'data' / '28_mols' / f"{molecule_name}.xyz",
            num_active_e,
            num_active_o,
            orbitals='canonical'
        )

# Cmo = np.loadtxt(SCRIPT_DIR / "Cmo_guess", delimiter=",")
# mol.ca = Cmo
# mol.hf_embedding(active=mol.active, frozen=mol.frozen)

ref_bitstr = '110011001100'
# ref_bitstr = '111111000000'
perm = [0,4,2,3,1,5]
perm = [0,5,1,4,2,3]
# perm = None
# perm = [0,3,1,4,2,5]
# initial_guess = np.fromstring(array_text, sep=' ')
tups = Ansatz_tUPS(mol=mol, layers=1, oo_layers=0, use_random_angles=False, use_mp2_guess=False, 
                    use_projection=True, use_mat_mul=True, perfect_pair=True, 
                    ref_bitstring=ref_bitstr, mo_perm=perm, use_small_perturb_angles=True
                    )
tups.run_oo_vqe_alternating(vqe_callback=vqe_callback, oo_callback=oo_callback, options={'gtol': 1e-6})

    # if converged is True:
    #     break
    # np.savetxt("Cmo_converged", tups.mol.ca, delimiter=',', fmt='%.5f')
# print(tups.final_circuit)
