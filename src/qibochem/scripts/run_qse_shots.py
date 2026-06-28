import gc
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data" / "output"

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_dir = OUTPUT_DIR / ".matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_dir)

import qibo

from qibochem.ansatz.ucc import (
    Ansatz_UCCGSD,
    Ansatz_UCCSD,
    Ansatz_UCCSDSinglet,
    Ansatz_kUpCCGSDSinglet,
)
from qibochem.measurement.protocol import MultiShotProtocol, StateVectorProtocol
from qibochem.scripts.script_utils import (
    Logger,
    append_jsonl,
    get_vqe_circuit,
    load_molecule,
    parse_active_space,
    read_jsonl,
)
from qibochem.selected_ci.qse import (
    QSE_Computable,
    generate_singlet_singles,
    generate_triplet_singles,
)

# qibo.set_backend("qibojit", platform="cuda")

ACTIVE_SPACES = [
    "2e2o", "2e3o", "4e3o", "4e4o"
]
"""
, "4e5o",
    "6e5o", "6e6o", "6e7o", "8e7o", "8e8o",
"""

MOLECULE_NAMES = [
    "Acetamide", "Acetone", "Adenine", "Benzene", "Benzoquinone",
    "Butadiene", "Cyclopentadiene", "Cyclopropene", "Cytosine", "Ethene",
    "Formaldehyde", "Formamide", "Furan", "Hexatriene", "Imidazole",
    "Naphthalene", "Norbornadiene", "Octatetraene", "Propanamide", "Pyrazine",
    "Pyridazine", "Pyridine", "Pyrimidine", "Pyrrole", "Tetrazine",
    "Thymine", "Triazine", "Uracil",
]

SAMPLE_SIZES = (1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000)
N_REPEATS = 10
FERM_QUBIT_MAP = "jw"
MAP_THRESHOLD = 1e-12
OPTIMIZER_METHOD = "L-BFGS-B"

ANSATZ_FUNCTIONS = {
    "UCCSDSinglet": Ansatz_UCCSDSinglet,
    "UCCSD": Ansatz_UCCSD,
}

QSE_EXPANSIONS = {
    "singlet": generate_singlet_singles,
    "triplet": generate_triplet_singles,
}


def expected_sample_keys():
    return {
        f"shots_{sample_size}_{repeat}"
        for sample_size in SAMPLE_SIZES
        for repeat in range(N_REPEATS)
    }


EXPECTED_SAMPLE_KEYS = expected_sample_keys()


def parse_sample_key(sample_key):
    prefix, n_shots, repeat = sample_key.split("_")
    if prefix != "shots":
        raise ValueError(f"Unrecognised multishot sample key: {sample_key}")
    return int(n_shots), int(repeat)


def completed_shot_keys(shot_file):
    completed = set()
    old_record_counts = Counter()

    for record in read_jsonl(shot_file):
        base_key = (
            record["molecule"],
            record["active_space"],
            record["ansatz"],
            record["expansion"],
        )

        if "sample_key" in record:
            completed.add((*base_key, record["sample_key"]))
            continue

        old_key = (*base_key, record["n_shots"])
        repeat = old_record_counts[old_key]
        old_record_counts[old_key] += 1
        if repeat < N_REPEATS:
            completed.add((*base_key, f"shots_{record['n_shots']}_{repeat}"))

    return completed


def save_multishot_qse_record(path, molecule, active_space, ansatz, expansion, sample_key, H, S):
    n_shots, repeat = parse_sample_key(sample_key)
    append_jsonl(
        path,
        {
            "molecule": molecule,
            "active_space": active_space,
            "ansatz": ansatz,
            "expansion": expansion,
            "sample_key": sample_key,
            "n_shots": n_shots,
            "repeat": repeat,
            "h_matrix_real": np.real(H).tolist(),
            "h_matrix_imag": np.imag(H).tolist(),
            "s_matrix_real": np.real(S).tolist(),
            "s_matrix_imag": np.imag(S).tolist(),
        },
    )


def main():
    input_dir = SCRIPT_DIR / "data" / "28_mols"
    output_dir = OUTPUT_DIR
    hs_dir = output_dir / "HS_data"
    qse_cache_dir = output_dir / "qse_cache"
    vqe_params_file = output_dir / "VQE_Params.jsonl"
    timing_file = output_dir / "MultiShot_Timings.log"

    output_dir.mkdir(parents=True, exist_ok=True)
    hs_dir.mkdir(parents=True, exist_ok=True)
    qse_cache_dir.mkdir(parents=True, exist_ok=True)

    log = Logger(timing_file)
    sv_protocol = StateVectorProtocol()

    shot_done = {
        active_space: completed_shot_keys(hs_dir / f"MultiShot_HS_{active_space}.jsonl")
        for active_space in ACTIVE_SPACES
    }

    for active_space in ACTIVE_SPACES:
        all_active_space_keys = {
            (molecule_name, active_space, ansatz_name, expansion, sample_key)
            for molecule_name in MOLECULE_NAMES
            for ansatz_name in ANSATZ_FUNCTIONS
            for expansion in QSE_EXPANSIONS
            for sample_key in EXPECTED_SAMPLE_KEYS
        }
        if all_active_space_keys.issubset(shot_done[active_space]):
            print(f"{active_space}: all multishot records done, skipping")
            continue

        num_active_e, num_active_o = parse_active_space(active_space)
        shot_file = hs_dir / f"MultiShot_HS_{active_space}.jsonl"

        for expansion, excitation_generator in QSE_EXPANSIONS.items():
            for molecule_name in MOLECULE_NAMES:
                molecule_expansion_keys = {
                    (molecule_name, active_space, ansatz_name, expansion, sample_key)
                    for ansatz_name in ANSATZ_FUNCTIONS
                    for sample_key in EXPECTED_SAMPLE_KEYS
                }
                if molecule_expansion_keys.issubset(shot_done[active_space]):
                    print(f"{molecule_name} {active_space} {expansion}: all ansatz multishot records done, skipping")
                    continue

                print(f"\n{active_space} {expansion} {molecule_name}")
                molecule_start = time.perf_counter()
                metadata = {
                    "molecule": molecule_name,
                    "active_space": active_space,
                    "expansion": expansion,
                }

                mol = load_molecule(
                    input_dir / f"{molecule_name}.xyz",
                    num_active_e,
                    num_active_o,
                    log=log,
                    **metadata,
                )

                molecule_qse_cache_dir = qse_cache_dir / active_space / expansion / molecule_name
                qse = QSE_Computable(
                    molecule=mol,
                    excitation_generator=excitation_generator,
                    observable=None,
                    spin_projection=0,
                    ferm_qubit_map=FERM_QUBIT_MAP,
                    map_threshold=MAP_THRESHOLD,
                    h_cache_path=str(molecule_qse_cache_dir / "H.pkl"),
                    s_cache_path=str(molecule_qse_cache_dir / "S.pkl"),
                )

                guess_amplitudes = None
                for ansatz_name, ansatz_function in ANSATZ_FUNCTIONS.items():
                    ansatz_sample_keys = {
                        (molecule_name, active_space, ansatz_name, expansion, sample_key)
                        for sample_key in EXPECTED_SAMPLE_KEYS
                    }
                    ansatz_start = time.perf_counter()
                    final_circuit, guess_amplitudes = get_vqe_circuit(
                        mol,
                        molecule_name,
                        active_space,
                        ansatz_name,
                        ansatz_function,
                        sv_protocol,
                        vqe_params_file,
                        log,
                        ferm_qubit_map=FERM_QUBIT_MAP,
                        optimizer_method=OPTIMIZER_METHOD,
                        guess_amplitudes=guess_amplitudes,
                    )

                    if ansatz_sample_keys.issubset(shot_done[active_space]):
                        print(f"{molecule_name} {active_space} {ansatz_name} {expansion}: multishot done, skipping")
                        del final_circuit
                        gc.collect()
                        continue

                    start = time.perf_counter()
                    multishot_protocol = MultiShotProtocol(
                        sample_sizes=SAMPLE_SIZES,
                        n_repeats=N_REPEATS,
                    )
                    h_samples, s_samples = qse.run_qse(final_circuit, multishot_protocol)
                    log("multishot_qse_evaluate", time.perf_counter() - start, **metadata, ansatz=ansatz_name)

                    for sample_key, h_matrix in h_samples.items():
                        shot_key = (molecule_name, active_space, ansatz_name, expansion, sample_key)
                        if shot_key in shot_done[active_space]:
                            continue

                        save_multishot_qse_record(
                            shot_file,
                            molecule_name,
                            active_space,
                            ansatz_name,
                            expansion,
                            sample_key,
                            h_matrix,
                            s_samples[sample_key],
                        )
                        shot_done[active_space].add(shot_key)

                    log("ansatz_total", time.perf_counter() - ansatz_start, **metadata, ansatz=ansatz_name)
                    del final_circuit
                    del h_samples
                    del s_samples
                    gc.collect()

                log("molecule_expansion_total", time.perf_counter() - molecule_start, **metadata)
                del qse
                del mol
                del guess_amplitudes
                gc.collect()


if __name__ == "__main__":
    main()
