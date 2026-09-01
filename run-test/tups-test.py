from pathlib import Path
import numpy as np


from qibochem.ansatz.ups import Ansatz_tUPS
from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.scripts.script_utils import load_molecule

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data" / "output"

np.set_printoptions(precision=5, suppress=True)

iteration=0
def callback(param):
    global iteration
    global tups

    iteration += 1
    print(f"Iteration {iteration}")
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

ref_bitstr = '110011001100'
ref_bitstr = '111111000000'
perm = [0,4,2,3,1,5]
perm = [0,5,1,4,2,3]
# perm = None
# perm = [0,3,1,4,2,5]
# initial_guess = np.fromstring(array_text, sep=' ')
tups = Ansatz_tUPS(mol=mol, layers=3, oo_layers=0, use_random_angles=False, use_mp2_guess=False, 
                    use_projection=True, use_mat_mul=True, perfect_pair=False, 
                    ref_bitstring=ref_bitstr, mo_perm=perm, use_small_perturb_angles=True
                    )
for i in range(1000):

    tups.run_vqe(protocol=sv_protocol, fast=True, callback=callback, method='POWELL', fast_mat_mul=True,options={'ftol':1e-9})
    # tups.run_oo(method="L-BFGS-B", callback=callback)
    # converged = tups.orbital_optimisation_step()
    

    # print(tups.t_rdm)
    print(tups.param_names)
    print(tups.energy)
    # if converged is True:
    #     break
# np.savetxt("Cmo_converged", tups.mol.ca, delimiter=',', fmt='%.5f')
# print(tups.final_circuit)
