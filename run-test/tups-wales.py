from pathlib import Path
import numpy as np

from qibochem.ansatz.ups import Ansatz_tUPS
from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.scripts.script_utils import load_molecule
from qibochem.interfaces.wales.input_util import setup

np.set_printoptions(precision=5, suppress=True)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "wales"

sv_protocol = StateVectorProtocol()

num_active_e = 6
num_active_o = 6
molecule_name = 'h6'
input_dir = "./data/"
mol = load_molecule(
            SCRIPT_DIR / 'data' / '28_mols' / f"{molecule_name}.xyz",
            num_active_e,
            num_active_o,
            orbitals='canonical'
        )

ref_bitstr = '110011001100' # initial bit string to specify occupancy of orbitals
# ref_bitstr = '111111000000' # initial bit string to specify occupancy of orbitals
# perm = [0,4,2,3,1,5]
perm = [0,5,1,4,2,3] # permutation to rearrange orbitals when perfect_pair==True, optional
# perm = None
# perm = [0,3,1,4,2,5]
tups = Ansatz_tUPS(mol=mol, layers=2, oo_layers=3, use_random_angles=False, use_small_perturb_angles=True, use_mp2_guess=False, 
                    use_projection=False, use_mat_mul=True, perfect_pair=True, ref_bitstring=ref_bitstr, mo_perm=perm,
                    init_ham=False, init_rdm=False,
                    )
setup(tups, path=OUTPUT_DIR, temp=0.05, tightconv=5e-7, sloppyconv=1e-7,)



