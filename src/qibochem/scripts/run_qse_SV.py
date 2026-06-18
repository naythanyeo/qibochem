import os
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "data" / "output"

if "MPLCONFIGDIR" not in os.environ:
    matplotlib_dir = OUTPUT_DIR / ".matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_dir)

from qibochem.ansatz.ucc import (
    Ansatz_UCCGSD,
    Ansatz_UCCSD,
    Ansatz_UCCSDSinglet,
    Ansatz_kUpCCGSDSinglet,
)
from qibochem.measurement.protocol import StateVectorProtocol
from qibochem.scripts.script_utils import (
    Logger,
    get_excitation_map,
    get_vqe_circuit,
    load_molecule,
    parse_active_space,
    read_jsonl,
    save_sv_qse_record,
)
from qibochem.selected_ci.qse import (
    QSE_Computable,
    generate_singlet_singles,
    generate_triplet_singles,
)

# qibo.set_backend("qibojit", platform="cuda")

ACTIVE_SPACES = ["2e2o", "2e3o", "4e3o", "4e4o"]

"""
"4e5o", "6e5o", "6e6o", "6e7o", "8e7o", "8e8o"
"""

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
OPTIMIZER_METHOD = "L-BFGS-B"

ANSATZ_FUNCTIONS = {
    "1UpCCGSDSinglet": lambda molecule, **kwargs: Ansatz_kUpCCGSDSinglet(molecule, k=1, **kwargs),
    "UCCSDSinglet": Ansatz_UCCSDSinglet,
    "UCCSD": Ansatz_UCCSD,
    "UCCGSD": Ansatz_UCCGSD,
}

QSE_EXPANSIONS = {
    "singlet": generate_singlet_singles,
    "triplet": generate_triplet_singles,
}


def completed_sv_keys(sv_file):
    return {
        (record["molecule"], record["active_space"], record["ansatz"], record["expansion"])
        for record in read_jsonl(sv_file)
    }


def main():
    input_dir = SCRIPT_DIR / "data" / "28_mols"
    output_dir = OUTPUT_DIR
    hs_dir = output_dir / "HS_data"
    excitation_maps_dir = output_dir / "excitation_maps"
    vqe_params_file = output_dir / "VQE_Params.jsonl"
    timing_file = output_dir / "SV_Timings.log"

    output_dir.mkdir(parents=True, exist_ok=True)
    hs_dir.mkdir(parents=True, exist_ok=True)
    excitation_maps_dir.mkdir(parents=True, exist_ok=True)

    log = Logger(timing_file)
    sv_protocol = StateVectorProtocol()

    sv_done = {
        active_space: completed_sv_keys(hs_dir / f"SV_HS_{active_space}.jsonl")
        for active_space in ACTIVE_SPACES
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

        num_active_e, num_active_o = parse_active_space(active_space)
        sv_file = hs_dir / f"SV_HS_{active_space}.jsonl"

        for expansion, excitation_generator in QSE_EXPANSIONS.items():
            excitation_map_file = excitation_maps_dir / f"{num_active_o}o_{expansion}.pkl"

            for molecule_name in MOLECULE_NAMES:
                molecule_expansion_keys = {
                    (molecule_name, active_space, ansatz_name, expansion)
                    for ansatz_name in ANSATZ_FUNCTIONS
                }
                if molecule_expansion_keys.issubset(sv_done[active_space]):
                    print(f"{molecule_name} {active_space} {expansion}: all ansatz SV records done, skipping")
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

                qse = QSE_Computable(
                    mol,
                    excitation_generator=excitation_generator,
                    spin_projection=0,
                    ferm_qubit_map=FERM_QUBIT_MAP,
                    map_threshold=MAP_THRESHOLD,
                )
                get_excitation_map(
                    qse,
                    excitation_map_file,
                    log,
                    **metadata,
                )

                guess_amplitudes = None
                for ansatz_name, ansatz_function in ANSATZ_FUNCTIONS.items():
                    sv_key = (molecule_name, active_space, ansatz_name, expansion)
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

                    if sv_key in sv_done[active_space]:
                        print(f"{molecule_name} {active_space} {ansatz_name} {expansion}: SV done, skipping")
                        continue

                    start = time.perf_counter()
                    h_matrix, s_matrix = qse.run_qse(final_circuit, sv_protocol)
                    log("qse_evaluate", time.perf_counter() - start, **metadata, ansatz=ansatz_name)

                    save_sv_qse_record(
                        sv_file,
                        molecule_name,
                        active_space,
                        ansatz_name,
                        expansion,
                        h_matrix,
                        s_matrix,
                    )
                    sv_done[active_space].add(sv_key)
                    log("ansatz_total", time.perf_counter() - ansatz_start, **metadata, ansatz=ansatz_name)

                log("molecule_expansion_total", time.perf_counter() - molecule_start, **metadata)


if __name__ == "__main__":
    main()
