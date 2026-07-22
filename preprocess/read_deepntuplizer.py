"""
Read DeepNTuplizer AK8 ROOT files into per-jet pandas DataFrames.

Each row in the output DataFrame corresponds to one AK8 jet. Array-valued
columns hold merged charged (Cpfcan_*) and neutral (Npfcan_*) constituents in
the format expected by ``preprocess.feature_engineering.modify_df``.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import uproot

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

TREE_PATH = "deepntuplizerAK8/tree"

JET_BRANCHES = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_mass",
    "jet_qk_charge_05",
    "jet_qk_charge_10",
    "n_Cpfcand",
    "n_Npfcand",
]

CHARGED_BRANCHES = [
    "Cpfcan_pt",
    "Cpfcan_etarel",
    "Cpfcan_phirel",
    "Cpfcan_charge",
    "Cpfcan_puppiw",
    "Cpfcan_pdg",
    "Cpfcan_dxy",
    "Cpfcan_dxyerrinv",
    "Cpfcan_dz",
    "Cpfcan_px",
    "Cpfcan_py",
    "Cpfcan_pz",
    "Cpfcan_e",
]

NEUTRAL_BRANCHES = [
    "Npfcan_pt",
    "Npfcan_etarel",
    "Npfcan_phirel",
    "Npfcan_puppiw",
    "Npfcan_pdgID",
    "Npfcan_isGamma",
    "Npfcan_px",
    "Npfcan_py",
    "Npfcan_pz",
    "Npfcan_e",
]

READ_BRANCHES = JET_BRANCHES + CHARGED_BRANCHES + NEUTRAL_BRANCHES

# DeepNTuplizer AK8 jet-level metadata written with fj_ prefix.
RAW_FATJET_PROPERTIES = [
    "phi",
    "eta",
    "pt",
    "mass",
    "qk_charge_05",
    "qk_charge_10",
]

def _constituent_mass(px: np.ndarray, py: np.ndarray, pz: np.ndarray, energy: np.ndarray) -> np.ndarray:
    mass_sq = energy ** 2 - px ** 2 - py ** 2 - pz ** 2
    return np.sqrt(np.maximum(mass_sq, 0.0))


def _d0_err_from_inv(dxyerrinv: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        err = np.where(dxyerrinv > 0.0, 1.0 / dxyerrinv, 0.0)
    return np.nan_to_num(err, nan=0.0, posinf=0.0, neginf=0.0)


def _map_neutral_pdg(pdg_id: int, is_gamma: int) -> int:
    if int(pdg_id) == 2:
        return 22
    if int(pdg_id) == 3:
        return 130
    return int(pdg_id)


def _merge_constituents(
    jet_pt: float,
    charged: Dict[str, np.ndarray],
    neutral: Dict[str, np.ndarray],
) -> Optional[Dict[str, np.ndarray]]:
    """Build one jet's constituent arrays in training column names."""

    if jet_pt <= 0.0:
        return None

    c_pt = np.asarray(charged["Cpfcan_pt"], dtype=np.float64)
    n_pt = np.asarray(neutral["Npfcan_pt"], dtype=np.float64)
    n_charged = len(c_pt)
    n_neutral = len(n_pt)

    if n_charged == 0 and n_neutral == 0:
        return None

    rel_pt = np.concatenate([c_pt / jet_pt, n_pt / jet_pt])
    if not np.any(rel_pt > 0.0):
        return None

    eta = np.concatenate(
        [
            np.asarray(charged["Cpfcan_etarel"], dtype=np.float64),
            np.asarray(neutral["Npfcan_etarel"], dtype=np.float64),
        ]
    )
    phi = np.concatenate(
        [
            np.asarray(charged["Cpfcan_phirel"], dtype=np.float64),
            np.asarray(neutral["Npfcan_phirel"], dtype=np.float64),
        ]
    )
    charge = np.concatenate(
        [
            np.asarray(charged["Cpfcan_charge"], dtype=np.float64),
            np.zeros(n_neutral, dtype=np.float64),
        ]
    )
    puppi = np.concatenate(
        [
            np.asarray(charged["Cpfcan_puppiw"], dtype=np.float64),
            np.asarray(neutral["Npfcan_puppiw"], dtype=np.float64),
        ]
    )
    pdg = np.concatenate(
        [
            np.asarray(charged["Cpfcan_pdg"], dtype=np.int64),
            np.array(
                [
                    _map_neutral_pdg(pid, gamma)
                    for pid, gamma in zip(
                        neutral["Npfcan_pdgID"],
                        neutral["Npfcan_isGamma"],
                    )
                ],
                dtype=np.int64,
            ),
        ]
    )

    c_px = np.asarray(charged["Cpfcan_px"], dtype=np.float64)
    c_py = np.asarray(charged["Cpfcan_py"], dtype=np.float64)
    c_pz = np.asarray(charged["Cpfcan_pz"], dtype=np.float64)
    c_e = np.asarray(charged["Cpfcan_e"], dtype=np.float64)
    n_px = np.asarray(neutral["Npfcan_px"], dtype=np.float64)
    n_py = np.asarray(neutral["Npfcan_py"], dtype=np.float64)
    n_pz = np.asarray(neutral["Npfcan_pz"], dtype=np.float64)
    n_e = np.asarray(neutral["Npfcan_e"], dtype=np.float64)

    px = np.concatenate([c_px, n_px])
    py = np.concatenate([c_py, n_py])
    pz = np.concatenate([c_pz, n_pz])
    energy = np.concatenate([c_e, n_e])
    mass = _constituent_mass(px, py, pz, energy)

    d0 = np.concatenate(
        [
            np.asarray(charged["Cpfcan_dxy"], dtype=np.float64),
            np.zeros(n_neutral, dtype=np.float64),
        ]
    )
    d0_err = np.concatenate(
        [
            _d0_err_from_inv(np.asarray(charged["Cpfcan_dxyerrinv"], dtype=np.float64)),
            np.ones(n_neutral, dtype=np.float64),
        ]
    )
    dz = np.concatenate(
        [
            np.asarray(charged["Cpfcan_dz"], dtype=np.float64),
            np.zeros(n_neutral, dtype=np.float64),
        ]
    )
    dz_err = np.ones(n_charged + n_neutral, dtype=np.float64)

    return {
        "pt": rel_pt.astype(np.float32),
        "eta": eta.astype(np.float32),
        "phi": phi.astype(np.float32),
        "charge": charge.astype(np.float32),
        "puppiWeight": puppi.astype(np.float32),
        "pdgId": pdg.astype(np.int64),
        "d0": d0.astype(np.float32),
        "d0Err": d0_err.astype(np.float32),
        "dz": dz.astype(np.float32),
        "dzErr": dz_err.astype(np.float32),
        "mass": mass.astype(np.float32),
    }


def _jet_metadata(
    jet_pt: float,
    jet_eta: float,
    jet_phi: float,
    jet_mass: float,
    qk_charge_05: float,
    qk_charge_10: float,
) -> Dict[str, float]:
    metadata = {
        "fj_pt": float(jet_pt),
        "fj_eta": float(jet_eta),
        "fj_phi": float(jet_phi),
        "fj_mass": float(jet_mass),
        "fj_qk_charge_05": float(qk_charge_05),
        "fj_qk_charge_10": float(qk_charge_10),
    }
    for prop in RAW_FATJET_PROPERTIES:
        key = f"fj_{prop}"
        if key not in metadata and prop in {
            "pt",
            "eta",
            "phi",
            "mass",
            "qk_charge_05",
            "qk_charge_10",
        }:
            metadata[key] = metadata[f"fj_{prop}"]
    return metadata


def _passes_pt_window(jet_pt: float, lowerpt: Optional[float], upperpt: Optional[float]) -> bool:
    if lowerpt is not None and jet_pt < lowerpt:
        return False
    if upperpt is not None and jet_pt > upperpt:
        return False
    return True


def read_root_file(
    filepath: str,
    lowerpt: Optional[float] = None,
    upperpt: Optional[float] = None,
    max_jets: Optional[int] = None,
    entry_start: int = 0,
    entry_stop: Optional[int] = None,
) -> pd.DataFrame:
    """
    Read one DeepNTuplizer AK8 ROOT file into a per-jet DataFrame.
    """

    rows: List[Dict[str, object]] = []

    with uproot.open(filepath) as handle:
        if TREE_PATH not in handle:
            raise KeyError(f"{filepath} does not contain tree '{TREE_PATH}'")
        tree = handle[TREE_PATH]
        stop = entry_stop if entry_stop is not None else tree.num_entries
        if max_jets is not None:
            stop = min(stop, entry_start + max_jets)

        if stop <= entry_start:
            return pd.DataFrame()

        table = tree.arrays(
            READ_BRANCHES,
            entry_start=entry_start,
            entry_stop=stop,
            library="np",
        )

    n_entries = len(table["jet_pt"])
    logging.info("Read %s jets from %s", n_entries, filepath)

    for idx in range(n_entries):
        jet_pt = float(table["jet_pt"][idx])
        if not _passes_pt_window(jet_pt, lowerpt, upperpt):
            continue

        n_c = int(table["n_Cpfcand"][idx])
        n_n = int(table["n_Npfcand"][idx])

        charged = {branch: table[branch][idx] for branch in CHARGED_BRANCHES}
        neutral = {branch: table[branch][idx] for branch in NEUTRAL_BRANCHES}

        constituents = _merge_constituents(jet_pt, charged, neutral)
        if constituents is None:
            continue

        row = dict(constituents)
        row.update(
            _jet_metadata(
                jet_pt=jet_pt,
                jet_eta=float(table["jet_eta"][idx]),
                jet_phi=float(table["jet_phi"][idx]),
                jet_mass=float(table["jet_mass"][idx]),
                qk_charge_05=float(table["jet_qk_charge_05"][idx]),
                qk_charge_10=float(table["jet_qk_charge_10"][idx]),
            )
        )
        rows.append(row)

        if max_jets is not None and len(rows) >= max_jets:
            break

    logging.info("Kept %s jets after selection from %s", len(rows), filepath)
    return pd.DataFrame(rows)


def collect_root_files(path: str, recursive: bool = True) -> List[str]:
    if os.path.isfile(path):
        return [path]

    files: List[str] = []
    if recursive:
        for root, _, filenames in os.walk(path):
            for filename in sorted(filenames):
                if filename.endswith(".root"):
                    files.append(os.path.join(root, filename))
    else:
        files = sorted(
            os.path.join(path, filename)
            for filename in os.listdir(path)
            if filename.endswith(".root")
        )
    return files


def read_root_paths(
    paths: Sequence[str],
    lowerpt: Optional[float] = None,
    upperpt: Optional[float] = None,
    max_jets: Optional[int] = None,
    max_files: Optional[int] = None,
    recursive: bool = True,
) -> pd.DataFrame:
    """
    Read and concatenate multiple ROOT files or directories.
    """

    root_files: List[str] = []
    for path in paths:
        root_files.extend(collect_root_files(path, recursive=recursive))

    root_files = sorted(root_files)
    if max_files is not None:
        root_files = root_files[:max_files]

    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in {paths}")

    frames: List[pd.DataFrame] = []
    remaining = max_jets

    for filepath in root_files:
        frame = read_root_file(
            filepath,
            lowerpt=lowerpt,
            upperpt=upperpt,
            max_jets=remaining,
        )
        if len(frame) > 0:
            frames.append(frame)
        if max_jets is not None:
            remaining = max_jets - sum(len(frame) for frame in frames)
            if remaining <= 0:
                break

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)
