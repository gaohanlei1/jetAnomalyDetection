"""CMS DeepNTuplizer AK8 reader and DDP-safe streaming dataset.

The dataset root contains one directory per jet type. Every ROOT shard has
already been shuffled across production families and pT ranges, so train/val/test
splits are random label-level file splits recorded in a manifest. Streaming uses
one independent active shard pool per label, matching the JetClass loader.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import awkward as ak
import numpy as np
import torch
import uproot
from torch.utils.data import IterableDataset, get_worker_info

TREE_PATH = "deepntuplizerAK8/tree"

###########################################

# NOTE:
# PARTICLE_FEATURES and BATCH_NORMALIZED_FEATURES are available to change
# to customize the features seen by the model. 
# Do not change other MACROS unless you are changing the data pipeline.
# The first four entries are a strict four-momentum contract used by the 
# ParticleTransformer.
PARTICLE_FEATURES = [
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_pt",
    "log_pt_fraction",
    "part_deta",
    "part_dphi",
    "Cpfcan_dxysig",
    # "log_Cpfcan_dxysig",
    "Cpfcan_dzsig",
    # "log_Cpfcan_dzsig",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
]

BATCH_NORMALIZED_FEATURES = [
    "log_pt_fraction",
    "Cpfcan_dxysig",
    # "log_Cpfcan_dxysig",
    "Cpfcan_dzsig",
    # "log_Cpfcan_dzsig",
]

###########################################


AVAILABLE_PARTICLE_FEATURES = [
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_pt",
    "log_pt_fraction",
    "part_deta",
    "part_dphi",
    "Cpfcan_dxysig",
    "log_Cpfcan_dxysig",
    "Cpfcan_dzsig",
    "log_Cpfcan_dzsig",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
]

CMS_DEFAULT_JET_FEATURES = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_mass",
    "jet_qk_charge_05",
    "jet_qk_charge_10",
]

CMS_FEATURE_SOURCES = {
    "part_px": "concatenate(Cpfcan_px, Npfcan_px)",
    "part_py": "concatenate(Cpfcan_py, Npfcan_py)",
    "part_pz": "concatenate(Cpfcan_pz, Npfcan_pz)",
    "part_energy": "concatenate(Cpfcan_e, Npfcan_e)",
    "part_pt": "concatenate(Cpfcan_pt, Npfcan_pt) / jet_pt",
    "log_pt_fraction": "log(clip(part_pt, 1e-8)) on valid candidates",
    "part_deta": "concatenate(Cpfcan_etarel, Npfcan_etarel)",
    "part_dphi": "concatenate(Cpfcan_phirel, Npfcan_phirel)",
    "Cpfcan_dxysig": "ROOT Cpfcan_dxysig branch; neutral=0; no extra clipping",
    "log_Cpfcan_dxysig": "log(clip(Cpfcan_dxysig, 1e-8)); neutral=0",
    "Cpfcan_dzsig": "ROOT Cpfcan_dzsig branch; neutral=0; no extra clipping",
    "log_Cpfcan_dzsig": "log(clip(Cpfcan_dzsig, 1e-8)); neutral=0",
    "part_charge": "Cpfcan_charge; neutral=0",
    "part_isChargedHadron": "abs(merged PDG ID) == 211",
    "part_isNeutralHadron": "abs(merged PDG ID) == 130",
    "part_isPhoton": "abs(merged PDG ID) == 22",
    "part_isElectron": "abs(merged PDG ID) == 11",
    "part_isMuon": "abs(merged PDG ID) == 13",
}

# Only genuinely equivalent CMS names are mapped to JetClass-style names.
# Impact-parameter variables are intentionally absent because their semantics
# differ across the datasets.
CMS_TO_JETCLASS_FEATURE_MAP = {
    "pt": "part_pt",
    "log_pt": "log_pt_fraction",
    "eta": "part_deta",
    "phi": "part_dphi",
    "charge": "part_charge",
    "pdgId_-211": "part_isChargedHadron",
    "pdgId_211": "part_isChargedHadron",
    "pdgId_130": "part_isNeutralHadron",
    "pdgId_22": "part_isPhoton",
    "pdgId_-11": "part_isElectron",
    "pdgId_11": "part_isElectron",
    "pdgId_-13": "part_isMuon",
    "pdgId_13": "part_isMuon",
}

CMS_LABELS = [
    "label_QCD",
    "label_Hbb",
    "label_Wqq",
    "label_Zqq",
    "label_Tbqq",
]

CMS_LABEL_TO_DIRECTORY = {
    "label_QCD": "qcd",
    "label_Hbb": "hbb",
    "label_Wqq": "wjets",
    "label_Zqq": "zjets",
    "label_Tbqq": "ttbar",
}

CMS_LABEL_TO_FILENAME_PREFIX = {
    "label_QCD": "qcd_",
    "label_Hbb": "hbb_",
    "label_Wqq": "wjets_",
    "label_Zqq": "zjets_",
    "label_Tbqq": "ttbar_",
}


_BRANCH_ALIASES: Dict[str, Tuple[str, ...]] = {
    "jet_pt": ("jet_pt",),
    "jet_eta": ("jet_eta",),
    "jet_phi": ("jet_phi",),
    "jet_mass": ("jet_mass",),
    "jet_qk_charge_05": ("jet_qk_charge_05",),
    "jet_qk_charge_10": ("jet_qk_charge_10",),
    "Cpfcan_pt": ("Cpfcan_pt",),
    "Cpfcan_etarel": ("Cpfcan_etarel",),
    "Cpfcan_phirel": ("Cpfcan_phirel",),
    "Cpfcan_charge": ("Cpfcan_charge",),
    "Cpfcan_pdg": ("Cpfcan_pdg",),
    "Cpfcan_dxysig": ("Cpfcan_dxysig",),
    "Cpfcan_dzsig": ("Cpfcan_dzsig",),
    "Cpfcan_px": ("Cpfcan_px",),
    "Cpfcan_py": ("Cpfcan_py",),
    "Cpfcan_pz": ("Cpfcan_pz",),
    "Cpfcan_e": ("Cpfcan_e",),
    "Npfcan_pt": ("Npfcan_pt",),
    "Npfcan_etarel": ("Npfcan_etarel",),
    "Npfcan_phirel": ("Npfcan_phirel",),
    "Npfcan_pdgID": ("Npfcan_pdgID",),
    "Npfcan_isGamma": ("Npfcan_isGamma",),
    "Npfcan_px": ("Npfcan_px",),
    "Npfcan_py": ("Npfcan_py",),
    "Npfcan_pz": ("Npfcan_pz",),
    "Npfcan_e": ("Npfcan_e",),
}

_REQUIRED_PARTICLE_BRANCHES = (
    "jet_pt",
    "jet_eta",
    "Cpfcan_pt",
    "Cpfcan_etarel",
    "Cpfcan_phirel",
    "Cpfcan_charge",
    "Cpfcan_pdg",
    "Cpfcan_dxysig",
    "Cpfcan_dzsig",
    "Cpfcan_px",
    "Cpfcan_py",
    "Cpfcan_pz",
    "Cpfcan_e",
    "Npfcan_pt",
    "Npfcan_etarel",
    "Npfcan_phirel",
    "Npfcan_pdgID",
    "Npfcan_isGamma",
    "Npfcan_px",
    "Npfcan_py",
    "Npfcan_pz",
    "Npfcan_e",
)

_OPTIONAL_LOGICAL_BRANCHES: set[str] = set()


def _pad(a, maxlen: int, value=0, dtype: str = "float32") -> np.ndarray:
    """Zero-pad or truncate a jagged array to a fixed particle axis."""
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a.astype(dtype, copy=False)
    if isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.to_numpy(ak.values_astype(a, dtype))
    output = np.full((len(a), maxlen), value, dtype=dtype)
    for idx, sequence in enumerate(a):
        if len(sequence) == 0:
            continue
        truncated = np.asarray(sequence[:maxlen], dtype=dtype)
        output[idx, : len(truncated)] = truncated
    return output


def _resolve_tree(handle: uproot.ReadOnlyDirectory):
    if TREE_PATH in handle:
        return handle[TREE_PATH]
    if "tree" in handle:
        return handle["tree"]
    raise KeyError(f"ROOT file contains neither {TREE_PATH!r} nor 'tree'.")


def _resolve_branch_names(tree, logical_names: Sequence[str]) -> Dict[str, Optional[str]]:
    available = {str(name).split(";")[0] for name in tree.keys()}
    resolved: Dict[str, Optional[str]] = {}
    missing: List[str] = []
    for logical_name in logical_names:
        match = next(
            (candidate for candidate in _BRANCH_ALIASES[logical_name] if candidate in available),
            None,
        )
        resolved[logical_name] = match
        if match is None and logical_name not in _OPTIONAL_LOGICAL_BRANCHES:
            missing.append(logical_name)
    if missing:
        details = {name: _BRANCH_ALIASES[name] for name in missing}
        raise KeyError(f"Missing required CMS branches: {details}")
    return resolved


def _map_neutral_pdg(pdg_id: ak.Array, is_gamma: ak.Array) -> ak.Array:
    """Map DeepNTuplizer neutral codes to standard PDG IDs."""
    pid = ak.values_astype(pdg_id, "int64")
    gamma = ak.values_astype(is_gamma, "int64")
    mapped = ak.where(pid == 2, 22, pid)
    mapped = ak.where(pid == 3, 130, mapped)
    mapped = ak.where(gamma == 1, 22, mapped)
    return mapped


def _apply_particle_order(table: MutableMapping[str, ak.Array]) -> None:
    """Sort merged charged/neutral candidates by descending pT in every jet."""
    order = ak.argsort(table["part_pt"], axis=1, ascending=False)
    for key in list(table):
        value = table[key]
        if isinstance(value, ak.Array) and value.ndim > 1:
            table[key] = value[order]


def _build_particle_table(
    arrays: Mapping[str, ak.Array],
    *,
    eps: float = 1e-8,
) -> Dict[str, ak.Array]:
    """Merge charged and neutral candidates into the CMS-native schema."""
    jet_pt = arrays["jet_pt"]
    safe_jet_pt = ak.where(jet_pt > 0, jet_pt, 1.0)

    charged_pt_abs = arrays["Cpfcan_pt"]
    neutral_pt_abs = arrays["Npfcan_pt"]
    part_pt = ak.concatenate(
        [charged_pt_abs / safe_jet_pt, neutral_pt_abs / safe_jet_pt], axis=1
    )
    part_deta = ak.concatenate(
        [arrays["Cpfcan_etarel"], arrays["Npfcan_etarel"]], axis=1
    )
    part_dphi = ak.concatenate(
        [arrays["Cpfcan_phirel"], arrays["Npfcan_phirel"]], axis=1
    )
    part_charge = ak.concatenate(
        [arrays["Cpfcan_charge"], ak.zeros_like(neutral_pt_abs)], axis=1
    )

    charged_pdg = ak.values_astype(arrays["Cpfcan_pdg"], "int64")
    neutral_pdg = _map_neutral_pdg(
        arrays["Npfcan_pdgID"], arrays["Npfcan_isGamma"]
    )
    part_pdg = ak.concatenate([charged_pdg, neutral_pdg], axis=1)

    part_px = ak.concatenate([arrays["Cpfcan_px"], arrays["Npfcan_px"]], axis=1)
    part_py = ak.concatenate([arrays["Cpfcan_py"], arrays["Npfcan_py"]], axis=1)
    part_pz = ak.concatenate([arrays["Cpfcan_pz"], arrays["Npfcan_pz"]], axis=1)
    part_energy = ak.concatenate([arrays["Cpfcan_e"], arrays["Npfcan_e"]], axis=1)

    dxysig_charged = ak.values_astype(arrays["Cpfcan_dxysig"], "float32")
    dzsig_charged = ak.values_astype(arrays["Cpfcan_dzsig"], "float32")
    log_dxysig_charged = np.log(
        ak.where(dxysig_charged > eps, dxysig_charged, eps)
    )
    log_dzsig_charged = np.log(
        ak.where(dzsig_charged > eps, dzsig_charged, eps)
    )

    dxysig = ak.concatenate(
        [dxysig_charged, ak.zeros_like(neutral_pt_abs)], axis=1
    )
    log_dxysig = ak.concatenate(
        [log_dxysig_charged, ak.zeros_like(neutral_pt_abs)], axis=1
    )
    dzsig = ak.concatenate(
        [dzsig_charged, ak.zeros_like(neutral_pt_abs)], axis=1
    )
    log_dzsig = ak.concatenate(
        [log_dzsig_charged, ak.zeros_like(neutral_pt_abs)], axis=1
    )
    log_pt_fraction = np.log(ak.where(part_pt > eps, part_pt, eps))

    table: Dict[str, ak.Array] = {
        "part_px": part_px,
        "part_py": part_py,
        "part_pz": part_pz,
        "part_energy": part_energy,
        "part_pt": part_pt,
        "log_pt_fraction": log_pt_fraction,
        "part_deta": part_deta,
        "part_dphi": part_dphi,
        "Cpfcan_dxysig": dxysig,
        "log_Cpfcan_dxysig": log_dxysig,
        "Cpfcan_dzsig": dzsig,
        "log_Cpfcan_dzsig": log_dzsig,
        "part_charge": part_charge,
        "part_isChargedHadron": ak.values_astype(abs(part_pdg) == 211, "float32"),
        "part_isNeutralHadron": ak.values_astype(abs(part_pdg) == 130, "float32"),
        "part_isPhoton": ak.values_astype(abs(part_pdg) == 22, "float32"),
        "part_isElectron": ak.values_astype(abs(part_pdg) == 11, "float32"),
        "part_isMuon": ak.values_astype(abs(part_pdg) == 13, "float32"),
    }

    _apply_particle_order(table)
    return table


def read_file(
    filepath: str,
    max_num_particles: int = 128,
    particle_features: Optional[Sequence[str]] = None,
    jet_features: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    lowerpt: Optional[float] = None,
    upperpt: Optional[float] = None,
    label_name: Optional[str] = None,
    class_index: Optional[int] = None,
    num_classes: Optional[int] = None,
    jet_eta_max: Optional[float] = 2.5,
    particle_dr_max: Optional[float] = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one CMS ROOT file with a JetClass-style return signature.

    When ``particle_features`` is omitted, ``PARTICLE_FEATURES`` is
    returned. CMS impact-parameter features retain their native names and
    semantics rather than being renamed to JetClass significance features.

    ``lowerpt`` and ``upperpt`` are optional event-level cuts on the actual
    reconstructed ``jet_pt`` branch.  File names are never used as pT cuts.
    One-hot labels follow the exact order of the supplied ``labels`` sequence.
    """
    requested_particles = list(
        PARTICLE_FEATURES
        if particle_features is None
        else particle_features
    )
    requested_jets = list(
        CMS_DEFAULT_JET_FEATURES if jet_features is None else jet_features
    )
    label_axis = list(CMS_LABELS if labels is None else labels)
    if num_classes is not None and labels is None:
        if int(num_classes) != len(label_axis):
            label_axis = [f"class_{index}" for index in range(int(num_classes))]

    if label_name is not None:
        if label_name not in label_axis:
            raise ValueError(f"label_name={label_name!r} is not in labels={label_axis}")
        resolved_class_index = label_axis.index(label_name)
    elif class_index is not None:
        resolved_class_index = int(class_index)
    else:
        raise ValueError("Either label_name or class_index must be provided.")
    if not 0 <= resolved_class_index < len(label_axis):
        raise ValueError(
            f"class_index={resolved_class_index} outside one-hot axis of length "
            f"{len(label_axis)}"
        )

    with uproot.open(filepath) as handle:
        try:
            tree = _resolve_tree(handle)
        except KeyError as error:
            raise KeyError(f"Could not resolve event tree in {filepath!r}: {error}") from error
        
        logical_names = list(_REQUIRED_PARTICLE_BRANCHES)
        logical_names.extend(
            name for name in requested_jets if name in _BRANCH_ALIASES
        )
        logical_names = list(dict.fromkeys(logical_names))
        resolved = _resolve_branch_names(tree, logical_names)
        physical_names = sorted({name for name in resolved.values() if name is not None})
        raw_physical = tree.arrays(physical_names, library="ak")

    arrays: Dict[str, ak.Array] = {
        logical: raw_physical[physical]
        for logical, physical in resolved.items()
        if physical is not None
    }
    table = _build_particle_table(arrays)

    jet_table: Dict[str, ak.Array] = {
        logical: arrays[logical]
        for logical in requested_jets
        if logical in arrays
    }
    unsupported_jets = [name for name in requested_jets if name not in jet_table]
    if unsupported_jets:
        raise ValueError(
            f"Unsupported or unavailable CMS jet features: {unsupported_jets}. "
            f"Requested: {requested_jets}"
        )

    unsupported_particles = [name for name in requested_particles if name not in table]
    if unsupported_particles:
        raise ValueError(
            f"Unsupported CMS particle features: {unsupported_particles}. "
            f"Available CMS particle features are {AVAILABLE_PARTICLE_FEATURES}."
        )

    jet_pt = ak.to_numpy(arrays["jet_pt"])
    jet_eta = ak.to_numpy(arrays["jet_eta"])
    event_mask = np.isfinite(jet_pt) & np.isfinite(jet_eta) & (jet_pt > 0)
    if lowerpt is not None:
        event_mask &= jet_pt >= float(lowerpt)
    if upperpt is not None:
        event_mask &= jet_pt < float(upperpt)
    if jet_eta_max is not None:
        event_mask &= np.abs(jet_eta) <= float(jet_eta_max)

    if not np.any(event_mask):
        return (
            np.zeros(
                (0, len(requested_particles), int(max_num_particles)),
                dtype=np.float32,
            ),
            np.zeros((0, len(requested_jets)), dtype=np.float32),
            np.zeros((0, len(label_axis)), dtype=np.float32),
        )

    for key in list(table):
        table[key] = table[key][event_mask]
    for key in list(jet_table):
        jet_table[key] = jet_table[key][event_mask]

    # Match JetClass particle quality handling before padding/truncation.
    valid_particle = (
        (table["part_pt"] > 0)
        & (table["part_energy"] > 0)
        & np.isfinite(table["part_pt"])
        & np.isfinite(table["part_px"])
        & np.isfinite(table["part_py"])
        & np.isfinite(table["part_pz"])
        & np.isfinite(table["part_energy"])
        & np.isfinite(table["part_deta"])
        & np.isfinite(table["part_dphi"])
        & np.isfinite(table["Cpfcan_dxysig"])
        & np.isfinite(table["log_Cpfcan_dxysig"])
        & np.isfinite(table["Cpfcan_dzsig"])
        & np.isfinite(table["log_Cpfcan_dzsig"])
        & np.isfinite(table["part_charge"])
    )
    if particle_dr_max is not None:
        dr = np.sqrt(table["part_deta"] ** 2 + table["part_dphi"] ** 2)
        valid_particle = valid_particle & (dr < float(particle_dr_max))
    # Remove invalid constituents before fixed-length truncation so they cannot
    # displace valid lower-pT particles from the first max_num_particles slots.
    for key in list(table):
        table[key] = table[key][valid_particle]

    x_particles = np.stack(
        [_pad(table[name], int(max_num_particles)) for name in requested_particles],
        axis=1,
    ).astype(np.float32, copy=False)
    x_jets = np.stack(
        [ak.to_numpy(jet_table[name]).astype(np.float32, copy=False) for name in requested_jets],
        axis=1,
    )
    y = np.zeros((len(x_jets), len(label_axis)), dtype=np.float32)
    y[:, resolved_class_index] = 1.0
    return (
        np.ascontiguousarray(x_particles),
        np.ascontiguousarray(x_jets),
        np.ascontiguousarray(y),
    )


def validate_cms_labels(labels: Sequence[str]) -> None:
    unknown = sorted(set(labels) - set(CMS_LABELS))
    if unknown:
        raise ValueError(f"Unknown CMS labels: {unknown}. Available labels: {CMS_LABELS}")



FilesByLabel = Dict[str, List[str]]
SplitFilesByLabel = Dict[str, FilesByLabel]


def discover_cms_files(
    dataset_root: str,
    labels: Sequence[str],
) -> FilesByLabel:
    """Discover non-empty shuffled ROOT shards independently for every label."""
    validate_cms_labels(labels)
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"CMS dataset root does not exist: {dataset_root}")

    discovered: FilesByLabel = {}
    for label in labels:
        directory = root / CMS_LABEL_TO_DIRECTORY[label]
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Missing CMS class directory for {label}: {directory}"
            )
        prefix = CMS_LABEL_TO_FILENAME_PREFIX[label]
        files = sorted(
            str(path.resolve())
            for path in directory.glob(f"{prefix}*.root")
            if path.is_file() and path.stat().st_size > 0
        )
        if not files:
            raise FileNotFoundError(
                f"No CMS ROOT files found for {label} in {directory} "
                f"with prefix {prefix!r}."
            )
        discovered[label] = files
    return discovered


def _stable_label_seed(seed: int, label: str) -> int:
    digest = hashlib.blake2b(
        f"{int(seed)}|{label}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little", signed=False)


def _split_label_files(
    files: Sequence[str],
    *,
    val_fraction: float,
    test_fraction: float,
    rng: random.Random,
) -> Tuple[List[str], List[str], List[str]]:
    shuffled = list(files)
    rng.shuffle(shuffled)
    n_files = len(shuffled)
    if n_files < 3 and val_fraction > 0 and test_fraction > 0:
        raise ValueError(
            "At least three shuffled CMS ROOT shards are required per label "
            f"for disjoint train/val/test splits; got {n_files}."
        )

    n_val = int(round(n_files * val_fraction))
    n_test = int(round(n_files * test_fraction))
    if val_fraction > 0:
        n_val = max(1, n_val)
    if test_fraction > 0:
        n_test = max(1, n_test)

    while n_val + n_test > n_files - 1:
        if n_val >= n_test and n_val > int(val_fraction > 0):
            n_val -= 1
        elif n_test > int(test_fraction > 0):
            n_test -= 1
        else:
            raise ValueError(
                f"Cannot retain a train shard with n_files={n_files}, "
                f"val_fraction={val_fraction}, test_fraction={test_fraction}."
            )

    val = shuffled[:n_val]
    test = shuffled[n_val : n_val + n_test]
    train = shuffled[n_val + n_test :]
    return train, val, test


def split_cms_files(
    files_by_label: Mapping[str, Sequence[str]],
    *,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 42,
) -> SplitFilesByLabel:
    """Randomly split each already-shuffled jet type into train/val/test shards."""
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError(
            "CMS split fractions must be non-negative and satisfy "
            "val_fraction + test_fraction < 1."
        )

    splits: SplitFilesByLabel = {"train": {}, "val": {}, "test": {}}
    for label, files in files_by_label.items():
        rng = random.Random(_stable_label_seed(seed, label))
        train, val, test = _split_label_files(
            files,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            rng=rng,
        )
        splits["train"][label] = train
        splits["val"][label] = val
        splits["test"][label] = test
    return splits


def cms_split_manifest(
    splits: Mapping[str, Mapping[str, Sequence[str]]],
) -> Dict[str, object]:
    """Build a versioned manifest for label-level shuffled-shard splits."""
    split_payload: Dict[str, Dict[str, List[str]]] = {}
    flattened_for_hash: List[str] = []
    for split_name, labels in splits.items():
        split_payload[split_name] = {}
        for label, paths in labels.items():
            normalized_paths = [str(path) for path in paths]
            split_payload[split_name][label] = normalized_paths
            flattened_for_hash.extend(
                f"{split_name}|{label}|{path}" for path in normalized_paths
            )

    return {
        "version": 2,
        "layout": "label-directories-with-pre-shuffled-shards",
        "splits": split_payload,
        "sha256": hashlib.sha256(
            "\n".join(sorted(flattened_for_hash)).encode("utf-8")
        ).hexdigest(),
    }


def load_and_preprocess_cms_file(
    filepath: str,
    *,
    label_name: str,
    label_axis: Sequence[str],
    particle_features: Sequence[str],
    max_num_particles: int,
    lowerpt: Optional[float],
    upperpt: Optional[float],
    min_nodes: int,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one shard and return canonical ``(events, particles, features)`` arrays."""
    del eps
    x_particles, _, y = read_file(
        filepath,
        max_num_particles=max_num_particles,
        particle_features=particle_features,
        jet_features=["jet_pt"],
        labels=label_axis,
        lowerpt=lowerpt,
        upperpt=upperpt,
        label_name=label_name,
    )
    x_particles = np.transpose(x_particles, (0, 2, 1)).astype(
        np.float32, copy=False
    )
    y = np.asarray(y, dtype=np.float32)
    valid_counts = np.count_nonzero(np.any(x_particles != 0, axis=-1), axis=1)
    keep = valid_counts >= int(min_nodes)
    return (
        np.ascontiguousarray(x_particles[keep], dtype=np.float32),
        np.ascontiguousarray(y[keep], dtype=np.float32),
    )


class CMSIterableDataset(IterableDataset):
    """Stream pre-shuffled CMS ROOT shards with one active pool per jet type.

    Files for every label are sharded independently across all DDP ranks and
    DataLoader workers. The active-shard budget is at least the number of
    requested labels, so every label owns at least one active shard. Events are
    yielded through a balanced label cycle. In infinite mode, each label
    refills its own local shard queue independently, so a shorter label cannot
    disappear while other labels continue streaming.
    """

    def __init__(
        self,
        files_by_label: Mapping[str, Sequence[str]],
        labels_to_load: Sequence[str],
        label_axis: Sequence[str],
        particle_features: Sequence[str],
        max_num_particles: int,
        min_nodes: int = 4,
        lowerpt: Optional[float] = None,
        upperpt: Optional[float] = None,
        max_events: Optional[int] = None,
        shuffle_files: bool = True,
        shuffle_active_shards: int = 1,
        infinite: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()
        validate_cms_labels(labels_to_load)
        if not labels_to_load:
            raise ValueError("labels_to_load must contain at least one label.")

        self.labels_to_load = list(labels_to_load)
        self.label_axis = list(label_axis)
        missing_axis = sorted(set(self.labels_to_load) - set(self.label_axis))
        if missing_axis:
            raise ValueError(f"labels_to_load absent from label_axis: {missing_axis}")

        self.files_by_label: FilesByLabel = {}
        for label in self.labels_to_load:
            paths = [str(path) for path in files_by_label.get(label, [])]
            if not paths:
                raise ValueError(f"No files supplied for CMS label {label!r}.")
            self.files_by_label[label] = paths

        self.particle_features = list(particle_features)
        if not self.particle_features:
            raise ValueError("particle_features must contain at least one feature.")
        if len(set(self.particle_features)) != len(self.particle_features):
            raise ValueError(
                f"particle_features contains duplicates: {self.particle_features}"
            )
        unknown_features = sorted(
            set(self.particle_features) - set(AVAILABLE_PARTICLE_FEATURES)
        )
        if unknown_features:
            raise ValueError(
                f"Unsupported CMS particle features: {unknown_features}. "
                f"Available features are {AVAILABLE_PARTICLE_FEATURES}."
            )
        self.feature_names = list(self.particle_features)
        self.batch_normalized_feature_names = [
            name for name in BATCH_NORMALIZED_FEATURES
            if name in self.particle_features
        ]

        self.max_num_particles = int(max_num_particles)
        self.min_nodes = int(min_nodes)
        self.lowerpt = lowerpt
        self.upperpt = upperpt
        self.max_events = max_events
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_active_shards = max(
            len(self.labels_to_load), int(shuffle_active_shards)
        )
        self.effective_active_shards = self.shuffle_active_shards
        self.infinite = bool(infinite)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.filepaths = sorted(
            path for paths in self.files_by_label.values() for path in paths
        )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        worker_info = get_worker_info()
        local_worker_id = 0 if worker_info is None else worker_info.id
        local_num_workers = 1 if worker_info is None else worker_info.num_workers
        global_worker_id = self.rank * local_num_workers + local_worker_id
        global_num_workers = self.world_size * local_num_workers

        worker_files_by_label: FilesByLabel = {}
        for label in self.labels_to_load:
            label_files = sorted(self.files_by_label[label])
            worker_label_files = label_files[
                global_worker_id::global_num_workers
            ]
            if not worker_label_files:
                raise RuntimeError(
                    "This rank/DataLoader worker received no CMS ROOT shard for "
                    f"label={label!r}; global_worker_id={global_worker_id}, "
                    f"global_num_workers={global_num_workers}, "
                    f"num_label_files={len(label_files)}. Reduce world_size × "
                    "num_workers or produce more shards for this jet type."
                )
            worker_files_by_label[label] = worker_label_files

        worker_quota: Optional[int] = None
        if self.max_events is not None:
            base = int(self.max_events) // global_num_workers
            remainder = int(self.max_events) % global_num_workers
            worker_quota = base + int(global_worker_id < remainder)

        num_labels = len(self.labels_to_load)
        base_active_per_label = self.shuffle_active_shards // num_labels
        active_remainder = self.shuffle_active_shards % num_labels
        active_budget_by_label = {
            label: base_active_per_label + int(index < active_remainder)
            for index, label in enumerate(self.labels_to_load)
        }

        pass_index = 0
        while True:
            rng = random.Random(
                self.seed + 9176 * global_worker_id + 104729 * pass_index
            )
            files_by_label = {
                label: list(paths)
                for label, paths in worker_files_by_label.items()
            }
            if self.shuffle_files:
                for paths in files_by_label.values():
                    rng.shuffle(paths)

            yielded = 0
            if not self.shuffle_files:
                loaded_by_label: Dict[
                    str, List[Tuple[torch.Tensor, torch.Tensor]]
                ] = {}
                for label in self.labels_to_load:
                    label_events: List[Tuple[torch.Tensor, torch.Tensor]] = []
                    for filepath in files_by_label[label]:
                        x_particles, y = load_and_preprocess_cms_file(
                            filepath,
                            label_name=label,
                            label_axis=self.label_axis,
                            particle_features=self.particle_features,
                            max_num_particles=self.max_num_particles,
                            lowerpt=self.lowerpt,
                            upperpt=self.upperpt,
                            min_nodes=self.min_nodes,
                        )
                        label_events.extend(
                            (
                                torch.from_numpy(event_x.copy()),
                                torch.from_numpy(event_y.copy()),
                            )
                            for event_x, event_y in zip(x_particles, y)
                        )
                    loaded_by_label[label] = label_events

                if self.infinite:
                    empty_labels = [
                        label for label in self.labels_to_load
                        if not loaded_by_label[label]
                    ]
                    if empty_labels:
                        raise RuntimeError(
                            "Infinite CMS stream found labels with no valid local "
                            "events after cuts: "
                            f"{empty_labels}; global_worker_id={global_worker_id}."
                        )

                cursors = {label: 0 for label in self.labels_to_load}
                while True:
                    made_progress = False
                    for label in self.labels_to_load:
                        if worker_quota is not None and yielded >= worker_quota:
                            break

                        events = loaded_by_label[label]
                        if not events:
                            continue

                        cursor = cursors[label]
                        if cursor >= len(events):
                            if not self.infinite:
                                continue
                            cursor = 0
                            cursors[label] = 0

                        yield events[cursor]
                        cursors[label] = cursor + 1
                        yielded += 1
                        made_progress = True

                    if worker_quota is not None and yielded >= worker_quota:
                        break
                    if not made_progress:
                        break
            else:
                # Keep one independent file queue and active shard pool per
                # label. In infinite mode, exhausting one label's local file
                # queue refills only that label, so no class can disappear
                # from the balanced label cycle while other classes continue.
                file_queues_by_label = {
                    label: list(files_by_label[label])
                    for label in self.labels_to_load
                }
                total_local_files_by_label = {
                    label: len(files_by_label[label])
                    for label in self.labels_to_load
                }
                empty_shard_streak = {
                    label: 0 for label in self.labels_to_load
                }
                active_by_label: Dict[str, List[Dict[str, object]]] = {
                    label: [] for label in self.labels_to_load
                }

                def refill_label_queue(label: str) -> None:
                    """Rebuild only one label's queue, excluding active paths."""
                    active_paths = {
                        str(shard["filepath"])
                        for shard in active_by_label[label]
                    }
                    candidates = [
                        filepath
                        for filepath in worker_files_by_label[label]
                        if filepath not in active_paths
                    ]
                    if self.shuffle_files:
                        rng.shuffle(candidates)
                    file_queues_by_label[label] = candidates

                def load_next_shard(
                    label: str,
                    *,
                    allow_refill: bool,
                ) -> Optional[Dict[str, object]]:
                    while True:
                        if not file_queues_by_label[label]:
                            if not (self.infinite and allow_refill):
                                return None
                            refill_label_queue(label)
                            if not file_queues_by_label[label]:
                                return None

                        filepath = file_queues_by_label[label].pop()
                        x_particles, y = load_and_preprocess_cms_file(
                            filepath,
                            label_name=label,
                            label_axis=self.label_axis,
                            particle_features=self.particle_features,
                            max_num_particles=self.max_num_particles,
                            lowerpt=self.lowerpt,
                            upperpt=self.upperpt,
                            min_nodes=self.min_nodes,
                        )
                        if len(x_particles) == 0:
                            empty_shard_streak[label] += 1
                            if (
                                empty_shard_streak[label]
                                >= total_local_files_by_label[label]
                            ):
                                raise RuntimeError(
                                    "A complete local CMS class pass produced no "
                                    "valid events after cuts. "
                                    f"label={label!r}, lowerpt={self.lowerpt}, "
                                    f"upperpt={self.upperpt}, "
                                    f"min_nodes={self.min_nodes}, "
                                    f"global_worker_id={global_worker_id}."
                                )
                            continue

                        empty_shard_streak[label] = 0
                        order = np.arange(len(x_particles), dtype=np.int64)
                        np.random.default_rng(rng.randrange(2**32)).shuffle(order)
                        return {
                            "x": x_particles,
                            "y": y,
                            "order": order,
                            "cursor": 0,
                            "filepath": filepath,
                        }

                # Initial active pools are finite: each local path is loaded at
                # most once during initialization. Independent refill begins
                # only when an active shard from that same label is exhausted.
                for label in self.labels_to_load:
                    budget = active_budget_by_label[label]
                    for _ in range(min(budget, len(files_by_label[label]))):
                        shard = load_next_shard(
                            label,
                            allow_refill=False,
                        )
                        if shard is not None:
                            active_by_label[label].append(shard)

                available_labels = [
                    label for label in self.labels_to_load
                    if active_by_label[label]
                ]
                while available_labels:
                    if worker_quota is not None and yielded >= worker_quota:
                        break
                    label_cycle = list(available_labels)
                    rng.shuffle(label_cycle)
                    for label in label_cycle:
                        if worker_quota is not None and yielded >= worker_quota:
                            break
                        active = active_by_label[label]
                        if not active:
                            continue

                        shard_index = rng.randrange(len(active))
                        shard = active[shard_index]
                        cursor = int(shard["cursor"])
                        order = shard["order"]
                        event_index = int(order[cursor])
                        shard["cursor"] = cursor + 1
                        x_particles = shard["x"]
                        y = shard["y"]
                        yield (
                            torch.from_numpy(x_particles[event_index].copy()),
                            torch.from_numpy(y[event_index].copy()),
                        )
                        yielded += 1

                        if int(shard["cursor"]) >= len(order):
                            active.pop(shard_index)
                            replacement = load_next_shard(
                                label,
                                allow_refill=True,
                            )
                            if replacement is not None:
                                active.append(replacement)
                            elif self.infinite:
                                raise RuntimeError(
                                    "Infinite CMS stream failed to refill a label "
                                    "after its active shard was exhausted. "
                                    f"label={label!r}, "
                                    f"global_worker_id={global_worker_id}."
                                )

                    available_labels = [
                        label for label in self.labels_to_load
                        if active_by_label[label]
                    ]

            if not self.infinite:
                break
            if yielded == 0:
                raise RuntimeError(
                    "Infinite CMS stream completed a full pass without yielding "
                    "events. Check event cuts, particle cuts, and ROOT schema. "
                    f"global_worker_id={global_worker_id}."
                )
            pass_index += 1


def collate_cms_tensors(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Stack fixed-size events and derive the all-zero particle padding mask."""
    xs, ys = zip(*batch)
    x_particles = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    padding_mask = x_particles.eq(0).all(dim=-1)
    return {"x_particles": x_particles, "padding_mask": padding_mask, "y": y}


__all__ = [
    "AVAILABLE_PARTICLE_FEATURES",
    "PARTICLE_FEATURES",
    "BATCH_NORMALIZED_FEATURES",
    "CMS_FEATURE_SOURCES",
    "CMS_DEFAULT_JET_FEATURES",
    "CMSIterableDataset",
    "CMS_LABELS",
    "CMS_LABEL_TO_DIRECTORY",
    "CMS_LABEL_TO_FILENAME_PREFIX",
    "CMS_TO_JETCLASS_FEATURE_MAP",
    "cms_split_manifest",
    "collate_cms_tensors",
    "discover_cms_files",
    "load_and_preprocess_cms_file",
    "read_file",
    "split_cms_files",
    "validate_cms_labels",
]
