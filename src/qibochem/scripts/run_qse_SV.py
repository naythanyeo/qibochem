import json
import pickle
import re
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

ACTIVE_SPACES = [
    "2e2o", "2e3o", "4e3o", "4e4o", "4e5o",
    "6e5o", "6e6o", "6e7o", "8e7o", "8e8o",
]

MOLECULE_NAMES = [
    "Acetamide", "Acetone", "Adenine", "Benzene", "Benzoquinone",
    "Butadiene", "Cyclopentadiene", "Cyclopropene", "Cytosine", "Ethene",
    "Formaldehyde", "Formamide", "Furan", "Hexatriene", "Imidazole",
    "Naphthalene", "Norbornadiene", "Octatetraene", "Propanamide", "Pyrazine",
    "Pyridazine", "Pyridine", "Pyrimidine", "Pyrrole", "Tetrazine",
    "Thymine", "Triazine", "Uracil",
]

FERM_QUBIT_MAP = "jw"
MAP_THRESHOLD = 1e-12

ANSATZ_FUNCTIONS = {
    "UCCSD": Ansatz_UCCSD,
    "UCCSDSinglet": Ansatz_UCCSDSinglet,
    "UCCGSD": Ansatz_UCCGSD,
    "1UpCCGSDSinglet": lambda molecule, **kwargs: Ansatz_kUpCCGSDSinglet(molecule, k=1, **kwargs),
}

QSE_EXPANSIONS = {
    "singlet": generate_singlet_singles,
    "triplet": generate_triplet_singles,
}


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as fp:
        return [json.loads(line) for line in fp if line.strip()]


def append_jsonl(path, record):
    with path.open("a") as fp:
        fp.write(json.dumps(record) + "\n")


def get_vqe_circuit(mol, molecule_name, active_space, ansatz_name, ansatz_function, SV_protocol, vqe_params_file):
    """Return final VQE circuit, using cached parameters if already available."""
    vqe_key = (molecule_name, active_space, ansatz_name)
    for record in read_jsonl(vqe_params_file):
        record_key = (record["molecule"], record["active_space"], record["ansatz"])
        if record_key == vqe_key:
            print(f"{molecule_name} {active_space} {ansatz_name}: reusing VQE parameters")
            vqe_params = record["vqe_params"]
            ansatz = ansatz_function(mol, final_params=vqe_params, ferm_qubit_map=FERM_QUBIT_MAP)
            return ansatz.final_circuit

    print(f"{molecule_name} {active_space} {ansatz_name}: running VQE")
    ansatz = ansatz_function(mol, ferm_qubit_map=FERM_QUBIT_MAP)
    vqe_energy, vqe_params, final_circuit = ansatz.run_vqe(SV_protocol, method="BFGS")
    vqe_record = {
        "molecule": molecule_name,
        "active_space": active_space,
        "ansatz": ansatz_name,
        "vqe_energy": float(vqe_energy),
        "vqe_params": {key: float(value) for key, value in vqe_params.items()},
    }
    append_jsonl(vqe_params_file, vqe_record)
    return final_circuit


def main():
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "data" / "28_mols"
    output_dir = script_dir / "data" / "output"
    hs_dir = output_dir / "HS_data"
    excitation_maps_dir = output_dir / "excitation_maps"
    vqe_params_file = output_dir / "VQE_Params.jsonl"

    output_dir.mkdir(parents=True, exist_ok=True)
    hs_dir.mkdir(parents=True, exist_ok=True)
    excitation_maps_dir.mkdir(parents=True, exist_ok=True)

    SV_protocol = StateVectorProtocol()

    sv_done = {}
    for active_space in ACTIVE_SPACES:
        sv_file = hs_dir / f"SV_HS_{active_space}.jsonl"
        sv_done[active_space] = {
            (record["molecule"], record["active_space"], record["ansatz"], record["expansion"])
            for record in read_jsonl(sv_file)
        }

    for active_space in ACTIVE_SPACES:
        all_active_space_keys = {
            (molecule_name, active_space, ansatz_name, expansion)
            for molecule_name in MOLECULE_NAMES
            for ansatz_name in ANSATZ_FUNCTIONS
            for expansion in QSE_EXPANSIONS
        }
        if all_active_space_keys.issubset(sv_done[active_space]):
            print(f"{active_space}: all SV records done, skipping")
            continue

        match = re.fullmatch(r"(\d+)e(\d+)o", active_space.lower())
        if match is None:
            raise ValueError("active_space must use the format '<electrons>e<orbitals>o'.")

        num_active_e, num_active_o = int(match.group(1)), int(match.group(2))
        sv_file = hs_dir / f"SV_HS_{active_space}.jsonl"

        for expansion, excitation_generator in QSE_EXPANSIONS.items():
            excitation_map_file = excitation_maps_dir / f"{num_active_o}o_{expansion}.pkl"
            if excitation_map_file.exists() and excitation_map_file.stat().st_size > 0:
                with excitation_map_file.open("rb") as fp:
                    map_entry = pickle.load(fp)
                excitation_map = map_entry["excitation_map"]
            else:
                excitation_map = None

            for molecule_name in MOLECULE_NAMES:
                molecule_expansion_keys = {
                    (molecule_name, active_space, ansatz_name, expansion)
                    for ansatz_name in ANSATZ_FUNCTIONS
                }
                if molecule_expansion_keys.issubset(sv_done[active_space]):
                    print(f"{molecule_name} {active_space} {expansion}: all ansatz SV records done, skipping")
                    continue

                xyz_path = input_dir / f"{molecule_name}.xyz"
                print(f"\n{active_space} {expansion} {molecule_name}")

                mol = Molecule(xyz_file=str(xyz_path), basis="sto-3g")
                mol.run_pyscf()

                active_mo_start = mol.nelec // 2 - num_active_e // 2
                active_mos = list(range(active_mo_start, active_mo_start + num_active_o))
                frozen_mos = [mo for mo in range(mol.nelec // 2) if mo not in active_mos]
                mol.hf_embedding(active=active_mos, frozen=frozen_mos)

                qse = QSE_Computable(
                    mol,
                    excitation_generator=excitation_generator,
                    spin_projection=0,
                    ferm_qubit_map=FERM_QUBIT_MAP,
                    cache_qse_matrix=True,
                    excitation_map=excitation_map,
                    map_threshold=MAP_THRESHOLD,
                )
                qse.operators = qse.excitation_generator(qse.excitation_params)

                if qse.excitation_map is None:
                    print(f"{active_space} {expansion}: building QSE map")
                    qse._build_excitation_map()
                    map_entry = {
                        "num_active_o": num_active_o,
                        "expansion": expansion,
                        "ferm_qubit_map": FERM_QUBIT_MAP,
                        "map_threshold": MAP_THRESHOLD,
                        "excitation_map": qse.excitation_map,
                    }
                    excitation_map = qse.excitation_map
                    with excitation_map_file.open("wb") as fp:
                        pickle.dump(map_entry, fp)
                else:
                    print(f"{active_space} {expansion}: reusing QSE map")

                qse.collate_hs_matrix_info()

                for ansatz_name, ansatz_function in ANSATZ_FUNCTIONS.items():
                    sv_key = (molecule_name, active_space, ansatz_name, expansion)
                    if sv_key in sv_done[active_space]:
                        print(f"{molecule_name} {active_space} {ansatz_name} {expansion}: SV done, skipping")
                        continue

                    final_circuit = get_vqe_circuit(
                        mol,
                        molecule_name,
                        active_space,
                        ansatz_name,
                        ansatz_function,
                        SV_protocol,
                        vqe_params_file,
                    )
                    h_matrix, s_matrix = qse.run_qse(final_circuit, SV_protocol)
                    sv_record = {
                        "molecule": molecule_name,
                        "active_space": active_space,
                        "ansatz": ansatz_name,
                        "expansion": expansion,
                        "h_matrix_real": np.real(h_matrix).tolist(),
                        "h_matrix_imag": np.imag(h_matrix).tolist(),
                        "s_matrix_real": np.real(s_matrix).tolist(),
                        "s_matrix_imag": np.imag(s_matrix).tolist(),
                    }
                    append_jsonl(sv_file, sv_record)
                    sv_done[active_space].add(sv_key)


if __name__ == "__main__":
    main()
