import json
import re
import time
from datetime import datetime

import numpy as np

from qibochem.driver.molecule import Molecule
from qibochem.ansatz.ucc_util import params2amplitudes


class Logger:
    def __init__(self, path):
        self.path = path

    def __call__(self, stage, seconds, **metadata):
        metadata_text = " ".join(f"{key}={value}" for key, value in metadata.items())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.path.open("a") as fp:
            fp.write(f"{timestamp} | {stage} | seconds={seconds:.6f} | {metadata_text}\n")


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as fp:
        return [json.loads(line) for line in fp if line.strip()]


def append_jsonl(path, record):
    with path.open("a") as fp:
        fp.write(json.dumps(record) + "\n")


def parse_active_space(active_space):
    match = re.fullmatch(r"(\d+)e(\d+)o", active_space.lower())
    if match is None:
        raise ValueError("active_space must use the format '<electrons>e<orbitals>o'.")
    return int(match.group(1)), int(match.group(2))


def load_molecule(xyz_path, num_active_e, num_active_o, log=None, **metadata):
    start = time.perf_counter()
    mol = Molecule(xyz_file=str(xyz_path), basis="sto-3g")
    mol.run_pyscf()

    active_mo_start = mol.nelec // 2 - num_active_e // 2
    active_mos = list(range(active_mo_start, active_mo_start + num_active_o))
    frozen_mos = [mo for mo in range(mol.nelec // 2) if mo not in active_mos]
    mol.hf_embedding(active=active_mos, frozen=frozen_mos)

    if log is not None:
        log("pyscf_and_embedding", time.perf_counter() - start, **metadata)
    return mol


def get_vqe_circuit(
    mol,
    molecule_name,
    active_space,
    ansatz_name,
    ansatz_function,
    protocol,
    vqe_params_file,
    log,
    ferm_qubit_map="jw",
    optimizer_method="L-BFGS-B",
    guess_amplitudes=None,
):
    metadata = {
        "molecule": molecule_name,
        "active_space": active_space,
        "ansatz": ansatz_name,
    }

    ansatz = ansatz_function(
        mol,
        ferm_qubit_map=ferm_qubit_map,
        guess_amplitudes=guess_amplitudes,
    )
    param_names = list(ansatz.param_names)

    cache_start = time.perf_counter()
    for record in reversed(read_jsonl(vqe_params_file)):
        if (
            record["molecule"] == molecule_name
            and record["active_space"] == active_space
            and record["ansatz"] == ansatz_name
            and record.get("param_names") == param_names
        ):
            ansatz._set_params(record["vqe_params"])
            log("vqe_cached", time.perf_counter() - cache_start, **metadata)
            next_guess_amplitudes = params2amplitudes(
                record["vqe_params"],
                ansatz.param_excitations,
            )
            return ansatz.circuit.copy(deep=True), next_guess_amplitudes

    start = time.perf_counter()
    vqe_energy, vqe_params, final_circuit = ansatz.run_vqe(
        protocol,
        method=optimizer_method,
        fast=True,
    )
    log("vqe_optimisation", time.perf_counter() - start, **metadata, optimizer=optimizer_method)

    append_jsonl(
        vqe_params_file,
        {
            "molecule": molecule_name,
            "active_space": active_space,
            "ansatz": ansatz_name,
            "vqe_energy": float(vqe_energy),
            "param_names": param_names,
            "vqe_params": {key: float(value) for key, value in vqe_params.items()},
        },
    )
    next_guess_amplitudes = params2amplitudes(vqe_params, ansatz.param_excitations)
    return final_circuit, next_guess_amplitudes


def save_sv_qse_record(path, molecule, active_space, ansatz, expansion, H, S):
    append_jsonl(
        path,
        {
            "molecule": molecule,
            "active_space": active_space,
            "ansatz": ansatz,
            "expansion": expansion,
            "h_matrix_real": np.real(H).tolist(),
            "h_matrix_imag": np.imag(H).tolist(),
            "s_matrix_real": np.real(S).tolist(),
            "s_matrix_imag": np.imag(S).tolist(),
        },
    )
