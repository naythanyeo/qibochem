import json
import re
from pathlib import Path

import numpy as np
import qibo
from qibochem.ansatz.ucc import UCCAnsatz
from qibochem.driver.molecule import Molecule
from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.selected_ci.qse import QSE_Computable, generate_singlet_singles

qibo.set_backend("qibojit", platform="cuda")

ACTIVE_SPACES = ["2e2o"]
ANSATZ = "UCCSD"


def run_qse(mol, active_space, ansatz=ANSATZ):
    """Run QSE for a molecule with given active space and ansatz.

    Returns VQE metadata and H/S matrices.
    """
    match = re.fullmatch(r"(\d+)e(\d+)o", active_space.lower())
    if match is None:
        raise ValueError("active_space must use the format '<electrons>e<orbitals>o'.")

    num_active_e = int(match.group(1))
    num_active_o = int(match.group(2))

    mol.run_pyscf()

    active_mo_start = mol.nelec // 2 - num_active_e // 2
    active_mos = list(range(active_mo_start, active_mo_start + num_active_o))
    frozen_mos = [mo for mo in range(mol.nelec // 2) if mo not in active_mos]
    mol.hf_embedding(active=active_mos, frozen=frozen_mos)

    vqe = UCCAnsatz(mol, ansatz_type=ansatz)
    vqe_energy, vqe_params, _ = vqe.run_vqe(method="BFGS")

    protocol = StateVectorProtocol()
    qse = QSE_Computable(mol, excitation_generator=generate_singlet_singles,
                         ferm_qubit_map="jw")
    h_matrix, s_matrix = qse.run_qse(vqe.final_circuit, protocol)

    return {"vqe_params": {key: float(value) for key, value in vqe_params.items()},
            "H": h_matrix, "S": s_matrix}


def main():
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "data" / "28_mols"
    output_dir = script_dir / "data" / "output"
    vqe_params_file = output_dir / "qse_results.jsonl"
    hs_file = output_dir / "qse_hs_matrices.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    xyz_files = sorted(input_dir.glob("*.xyz"))

    with (vqe_params_file.open("w") as vqe_out, hs_file.open("w") as hs_out):
        for active_space in ACTIVE_SPACES:
            for xyz_path in xyz_files:
                mol = Molecule(xyz_file=str(xyz_path), basis="sto-3g")
                result = run_qse(mol, active_space, ANSATZ)

                metadata = {"molecule": xyz_path.stem, "ansatz": ANSATZ,
                            "active_space": active_space,}
                vqe_data = {**metadata, "vqe_params": result["vqe_params"]}
                hs_data = {**metadata, "h_matrix_real": np.real(result["H"]).tolist(),
                           "h_matrix_imag": np.imag(result["H"]).tolist(),
                           "s_matrix_real": np.real(result["S"]).tolist(),
                           "s_matrix_imag": np.imag(result["S"]).tolist()}

                vqe_out.write(json.dumps(vqe_data) + "\n")
                hs_out.write(json.dumps(hs_data) + "\n")


if __name__ == "__main__":
    main()
