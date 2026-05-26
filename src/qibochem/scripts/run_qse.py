import json
import re
from functools import partial
from pathlib import Path

import numpy as np
import qibo
from qibochem.ansatz.ucc import (
    Ansatz_UCCGSD,
    Ansatz_UCCSD,
    Ansatz_UCCSDSinglet,
    Ansatz_kUpCCGSDSinglet,
)
from qibochem.driver.molecule import Molecule
from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.selected_ci.qse import (
    QSE_Computable,
    generate_singlet_singles,
    generate_triplet_singles,
)

qibo.set_backend("qibojit", platform="cuda")

# Edit this list to control the sweep.
ACTIVE_SPACES = ["2e2o", "2e3o", "4e3o", "4e4o",
                 "4e5o", "6e5o", "6e6o", "6e7o",
                 "8e7o", "8e8o"]
ANSATZ_FACTORIES = {
    "UCCSD": Ansatz_UCCSD,
    "UCCSDSinglet": Ansatz_UCCSDSinglet,
    "UCCGSD": Ansatz_UCCGSD,
    "1UpCCGSDSinglet": lambda molecule, **kwargs: Ansatz_kUpCCGSDSinglet(molecule, k=1, **kwargs),
}

# File suffix -> QSE excitation generator. Triplet output is explicitly ms=0.
QSE_EXPANSIONS = {
    "singlet": generate_singlet_singles,
    "triplet_ms0": partial(generate_triplet_singles, ms=0),
}


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as fp:
        return [json.loads(line) for line in fp if line.strip()]


def append_jsonl(path, record):
    with path.open("a") as fp:
        fp.write(json.dumps(record) + "\n")


def get_vqe_circuit(mol, molecule_name, active_space, ansatz_name, ansatz_factory, protocol, vqe_cache, vqe_params_file):
    """Return a final VQE circuit, using cached parameters when available."""
    vqe_key = (molecule_name, active_space, ansatz_name)
    if vqe_key in vqe_cache:
        print(f"{ansatz_name} {active_space} VQE parameters found, reusing")
        vqe_params = vqe_cache[vqe_key]["vqe_params"]
        ansatz = ansatz_factory(mol, final_params=vqe_params)
        return ansatz.final_circuit

    ansatz = ansatz_factory(mol)
    vqe_energy, vqe_params, final_circuit = ansatz.run_vqe(protocol, method="BFGS")
    vqe_record = {
        "molecule": molecule_name,
        "active_space": active_space,
        "ansatz": ansatz_name,
        "vqe_energy": float(vqe_energy),
        "vqe_params": {key: float(value) for key, value in vqe_params.items()},
    }
    append_jsonl(vqe_params_file, vqe_record)
    vqe_cache[vqe_key] = vqe_record
    return final_circuit


def run_qse(
    mol,
    molecule_name,
    active_space,
    ansatz_name,
    ansatz_factory,
    expansion,
    excitation_generator,
    protocol,
    vqe_cache,
    vqe_params_file,
):
    """Run cached VQE followed by one QSE expansion and return a JSONL record."""
    final_circuit = get_vqe_circuit(
        mol,
        molecule_name,
        active_space,
        ansatz_name,
        ansatz_factory,
        protocol,
        vqe_cache,
        vqe_params_file,
    )
    qse = QSE_Computable(
        mol,
        excitation_generator=excitation_generator,
        ferm_qubit_map="jw",
    )
    h_matrix, s_matrix = qse.run_qse(final_circuit, protocol)
    return {
        "molecule": molecule_name,
        "active_space": active_space,
        "ansatz": ansatz_name,
        "expansion": expansion,
        "h_matrix_real": np.real(h_matrix).tolist(),
        "h_matrix_imag": np.imag(h_matrix).tolist(),
        "s_matrix_real": np.real(s_matrix).tolist(),
        "s_matrix_imag": np.imag(s_matrix).tolist(),
    }


def main():
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "data" / "28_mols"
    output_dir = script_dir / "data" / "output"
    hs_dir = output_dir / "HS_data"
    vqe_params_file = output_dir / "VQE_Params.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    hs_dir.mkdir(parents=True, exist_ok=True)

    protocol = StateVectorProtocol()
    xyz_files = sorted(input_dir.glob("*.xyz"))

    # VQE is cached once per molecule, active space, and ansatz. The same circuit
    # is reused for singlet and triplet QSE.
    vqe_cache = {
        (record["molecule"], record["active_space"], record["ansatz"]): record
        for record in read_jsonl(vqe_params_file)
    }

    # H/S output is cached separately for each active space.
    hs_done = {}
    for active_space in ACTIVE_SPACES:
        hs_file = hs_dir / f"HS_{active_space}.jsonl"
        hs_done[active_space] = {
            (record["molecule"], record["active_space"], record["ansatz"], record["expansion"])
            for record in read_jsonl(hs_file)
        }

    for active_space in ACTIVE_SPACES:
        match = re.fullmatch(r"(\d+)e(\d+)o", active_space.lower())
        if match is None:
            raise ValueError("active_space must use the format '<electrons>e<orbitals>o'.")
        num_active_e = int(match.group(1))
        num_active_o = int(match.group(2))

        for xyz_path in xyz_files:
            molecule_name = xyz_path.stem
            mol = Molecule(xyz_file=str(xyz_path), basis="sto-3g")
            mol.run_pyscf()

            active_mo_start = mol.nelec // 2 - num_active_e // 2
            active_mos = list(range(active_mo_start, active_mo_start + num_active_o))
            frozen_mos = [mo for mo in range(mol.nelec // 2) if mo not in active_mos]
            mol.hf_embedding(active=active_mos, frozen=frozen_mos)

            for ansatz_name, ansatz_factory in ANSATZ_FACTORIES.items():
                for expansion, excitation_generator in QSE_EXPANSIONS.items():
                    hs_key = (molecule_name, active_space, ansatz_name, expansion)
                    hs_file = hs_dir / f"HS_{active_space}.jsonl"
                    if hs_key in hs_done[active_space]:
                        print(f"{ansatz_name} {active_space} {expansion} done, skipping")
                        continue

                    hs_record = run_qse(
                        mol,
                        molecule_name,
                        active_space,
                        ansatz_name,
                        ansatz_factory,
                        expansion,
                        excitation_generator,
                        protocol,
                        vqe_cache,
                        vqe_params_file,
                    )
                    append_jsonl(hs_file, hs_record)
                    hs_done[active_space].add(hs_key)


if __name__ == "__main__":
    main()
