"""
DeepNTuplizer AK8 dataloader (JetClass-style API).

Follows the ``read_file`` convention from JetClass / Particle Transformer:
https://github.com/jet-universe/particle_transformer/blob/main/dataloader.py

DeepNTuplizer branches (``Cpfcan_*`` / ``Npfcan_*`` / ``jet_*``) are mapped to
particle- and jet-level feature arrays suitable for PART / LeJEPA training.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Sequence, Tuple, Union

import awkward as ak
import numpy as np
import uproot

TREE_PATH = "deepntuplizerAK8/tree"

# Default particle features match LeJEPA PART ``--node-features``.
DEFAULT_PARTICLE_FEATURES = [
    "pt",
    "eta",
    "phi",
    "d0/d0Err",
    "dz/dzErr",
    "charge",
    "mass",
    "log_pt",
    "pdgId_-211",
    "pdgId_-13",
    "pdgId_-11",
    "pdgId_11",
    "pdgId_13",
    "pdgId_22",
    "pdgId_130",
    "pdgId_211",
]

DEFAULT_JET_FEATURES = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_mass",
    "jet_qk_charge_05",
    "jet_qk_charge_10",
]

# One-hot PDG IDs used by DEFAULT_PARTICLE_FEATURES.
STANDARD_PDG_IDS = [-211, -13, -11, 11, 13, 22, 130, 211]

# Charged PDG IDs allowed for impact-parameter significance (matches old pipeline).
CHARGED_PDG_FOR_IP = np.array([-211, -13, -11, 11, 13, 211], dtype=np.int64)

_READ_BRANCHES = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_mass",
    "jet_qk_charge_05",
    "jet_qk_charge_10",
    "Cpfcan_pt",
    "Cpfcan_etarel",
    "Cpfcan_phirel",
    "Cpfcan_charge",
    "Cpfcan_pdg",
    "Cpfcan_dxy",
    "Cpfcan_dxyerrinv",
    "Cpfcan_dz",
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
]


def _pad(a, maxlen: int, value=0, dtype="float32"):
    """Zero-pad / truncate jagged arrays to fixed length (JetClass helper)."""
    if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
        return a
    if isinstance(a, ak.Array):
        if a.ndim == 1:
            a = ak.unflatten(a, 1)
        a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
        return ak.to_numpy(ak.values_astype(a, dtype))
    x = (np.ones((len(a), maxlen)) * value).astype(dtype)
    for idx, s in enumerate(a):
        if not len(s):
            continue
        trunc = np.asarray(s[:maxlen], dtype=dtype)
        x[idx, : len(trunc)] = trunc
    return x


def _map_neutral_pdg(pdg_id: ak.Array, is_gamma: ak.Array) -> ak.Array:
    """Map DeepNTuplizer neutral codes: 2→22 (γ), 3→130 (hadronic neutral)."""
    pid = ak.values_astype(pdg_id, "int64")
    gamma = ak.values_astype(is_gamma, "int64")
    out = pid
    out = ak.where(pid == 2, 22, out)
    out = ak.where(pid == 3, 130, out)
    # Prefer isGamma flag when set
    out = ak.where(gamma == 1, 22, out)
    return out


def _build_particle_table(tree_arrays) -> dict:
    """Merge charged + neutral constituents and derive PART training features."""
    jet_pt = tree_arrays["jet_pt"]

    c_pt = tree_arrays["Cpfcan_pt"]
    n_pt = tree_arrays["Npfcan_pt"]
    # Relative pT (cand / jet), matching the old pickle pipeline.
    part_pt = ak.concatenate([c_pt / jet_pt, n_pt / jet_pt], axis=1)

    part_eta = ak.concatenate(
        [tree_arrays["Cpfcan_etarel"], tree_arrays["Npfcan_etarel"]], axis=1
    )
    part_phi = ak.concatenate(
        [tree_arrays["Cpfcan_phirel"], tree_arrays["Npfcan_phirel"]], axis=1
    )
    part_charge = ak.concatenate(
        [tree_arrays["Cpfcan_charge"], ak.zeros_like(n_pt)],
        axis=1,
    )

    n_pdg = _map_neutral_pdg(tree_arrays["Npfcan_pdgID"], tree_arrays["Npfcan_isGamma"])
    c_pdg = ak.values_astype(tree_arrays["Cpfcan_pdg"], "int64")
    part_pdg = ak.concatenate([c_pdg, n_pdg], axis=1)

    px = ak.concatenate([tree_arrays["Cpfcan_px"], tree_arrays["Npfcan_px"]], axis=1)
    py = ak.concatenate([tree_arrays["Cpfcan_py"], tree_arrays["Npfcan_py"]], axis=1)
    pz = ak.concatenate([tree_arrays["Cpfcan_pz"], tree_arrays["Npfcan_pz"]], axis=1)
    energy = ak.concatenate([tree_arrays["Cpfcan_e"], tree_arrays["Npfcan_e"]], axis=1)
    mass_sq = energy ** 2 - px ** 2 - py ** 2 - pz ** 2
    part_mass = np.sqrt(ak.where(mass_sq > 0, mass_sq, 0.0))

    c_d0 = tree_arrays["Cpfcan_dxy"]
    c_d0err = ak.where(
        tree_arrays["Cpfcan_dxyerrinv"] > 0,
        1.0 / tree_arrays["Cpfcan_dxyerrinv"],
        0.0,
    )
    c_d0sig = ak.where(c_d0err > 0, c_d0 / c_d0err, 0.0)
    charged_mask = (
        (c_pdg == -211)
        | (c_pdg == -13)
        | (c_pdg == -11)
        | (c_pdg == 11)
        | (c_pdg == 13)
        | (c_pdg == 211)
    )
    c_d0sig = ak.where(charged_mask, c_d0sig, 0.0)
    c_d0sig = ak.where(c_d0sig > 5.0, 5.0, c_d0sig)
    c_d0sig = ak.where(c_d0sig < -5.0, -5.0, c_d0sig)
    part_d0sig = ak.concatenate([c_d0sig, ak.zeros_like(n_pt)], axis=1)

    c_dz = tree_arrays["Cpfcan_dz"]
    c_dzsig = ak.where(charged_mask, c_dz, 0.0)
    c_dzsig = ak.where(c_dzsig > 5.0, 5.0, c_dzsig)
    c_dzsig = ak.where(c_dzsig < -5.0, -5.0, c_dzsig)
    part_dzsig = ak.concatenate([c_dzsig, ak.zeros_like(n_pt)], axis=1)

    part_log_pt = np.log(ak.where(part_pt > 1e-8, part_pt, 1e-8))

    table = {
        "pt": part_pt,
        "eta": part_eta,
        "phi": part_phi,
        "charge": part_charge,
        "mass": part_mass,
        "log_pt": part_log_pt,
        "d0/d0Err": part_d0sig,
        "dz/dzErr": part_dzsig,
        "pdgId": part_pdg,
        "part_pt": part_pt,
        "part_eta": part_eta,
        "part_phi": part_phi,
        "part_charge": part_charge,
        "part_d0val": ak.concatenate([c_d0, ak.zeros_like(n_pt)], axis=1),
        "part_dzval": ak.concatenate([c_dz, ak.zeros_like(n_pt)], axis=1),
    }

    for pid in STANDARD_PDG_IDS:
        table[f"pdgId_{pid}"] = ak.values_astype(part_pdg == int(pid), "float32")

    table["jet_pt"] = tree_arrays["jet_pt"]
    table["jet_eta"] = tree_arrays["jet_eta"]
    table["jet_phi"] = tree_arrays["jet_phi"]
    table["jet_mass"] = tree_arrays["jet_mass"]
    table["jet_qk_charge_05"] = tree_arrays["jet_qk_charge_05"]
    table["jet_qk_charge_10"] = tree_arrays["jet_qk_charge_10"]
    table["jet_nparticles"] = ak.num(part_pt, axis=1)

    return table


def read_file(
    filepath: str,
    max_num_particles: int = 128,
    particle_features: Optional[Sequence[str]] = None,
    jet_features: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    lowerpt: Optional[float] = 150.0,
    upperpt: Optional[float] = None,
    class_index: int = 0,
    num_classes: int = 4,
):
    """Loads a single file from the DeepNTuplizer AK8 dataset.

    API mirrors JetClass ``read_file``:
    https://github.com/jet-universe/particle_transformer/blob/main/dataloader.py

    **Arguments**

    - **filepath** : _str_
        - Path to the DeepNTuplizer ROOT file (``deepntuplizerAK8/tree``).
    - **max_num_particles** : _int_
        - The maximum number of particles to load for each jet.
          Jets with fewer particles will be zero-padded,
          and jets with more particles will be truncated.
    - **particle_features** : _List[str]_
        - Particle-level features to load. Defaults match LeJEPA PART node
          features. Available names include:
            - pt (cand_pt / jet_pt), eta (etarel), phi (phirel)
            - d0/d0Err, dz/dzErr, charge, mass, log_pt
            - pdgId_-211, pdgId_-13, pdgId_-11, pdgId_11, pdgId_13,
              pdgId_22, pdgId_130, pdgId_211
            - JetClass-like aliases: part_pt, part_eta, part_phi, part_charge,
              part_d0val, part_dzval
    - **jet_features** : _List[str]_
        - Jet-level features. Available:
            - jet_pt, jet_eta, jet_phi, jet_mass
            - jet_qk_charge_05, jet_qk_charge_10, jet_nparticles
    - **labels** : _List[str]_ | None
        - DeepNTuplizer ntuples do not store JetClass ``label_*`` branches.
          If ``labels`` is None, a one-hot vector of length ``num_classes`` is
          built with a 1 at ``class_index`` (QCD=0, W=1, Z=2, ttbar=3 by
          convention). If a list is provided, it is used only as the class
          axis length / names for documentation; values still come from
          ``class_index``.
    - **lowerpt** / **upperpt** : _float_ | None
        - Optional AK8 jet pT window in GeV (default lowerpt=150).
    - **class_index** : _int_
        - Class id for this file's one-hot label.
    - **num_classes** : _int_
        - Length of the one-hot label axis (ignored if ``labels`` is set;
          then ``len(labels)`` is used).

    **Returns**

    - x_particles(_3-d numpy.ndarray_), x_jets(_2-d numpy.ndarray_), y(_2-d numpy.ndarray_)
        - ``x_particles``: zero-padded particle features with shape
          ``(num_jets, num_particle_features, max_num_particles)``.
        - ``x_jets``: jet features with shape ``(num_jets, num_jet_features)``.
        - ``y``: one-hot labels with shape ``(num_jets, num_classes)``.
    """
    if particle_features is None:
        particle_features = list(DEFAULT_PARTICLE_FEATURES)
    if jet_features is None:
        jet_features = list(DEFAULT_JET_FEATURES)

    n_classes = len(labels) if labels is not None else int(num_classes)
    if class_index < 0 or class_index >= n_classes:
        raise ValueError(f"class_index={class_index} out of range for num_classes={n_classes}")

    with uproot.open(filepath) as handle:
        if TREE_PATH not in handle:
            # Fall back to JetClass-style bare 'tree' if present
            if "tree" in handle:
                tree = handle["tree"]
            else:
                raise KeyError(f"{filepath} missing '{TREE_PATH}' (and 'tree')")
        else:
            tree = handle[TREE_PATH]
        raw = tree.arrays(_READ_BRANCHES, library="ak")

    table = _build_particle_table(raw)

    # Jet pT selection
    mask = np.ones(len(table["jet_pt"]), dtype=bool)
    jet_pt_np = ak.to_numpy(table["jet_pt"])
    if lowerpt is not None:
        mask &= jet_pt_np >= float(lowerpt)
    if upperpt is not None:
        mask &= jet_pt_np <= float(upperpt)
    if not np.any(mask):
        f = len(particle_features)
        j = len(jet_features)
        return (
            np.zeros((0, f, max_num_particles), dtype=np.float32),
            np.zeros((0, j), dtype=np.float32),
            np.zeros((0, n_classes), dtype=np.int32),
        )

    # Apply mask to jagged particle features via awkward
    for key in list(table.keys()):
        table[key] = table[key][mask]

    x_particles = np.stack(
        [
            np.asarray(_pad(table[name], maxlen=max_num_particles), dtype=np.float32)
            for name in particle_features
        ],
        axis=1,
    )
    x_jets = np.stack(
        [np.asarray(ak.to_numpy(table[name]), dtype=np.float32) for name in jet_features],
        axis=1,
    )
    y = np.zeros((x_jets.shape[0], n_classes), dtype=np.int32)
    y[:, class_index] = 1
    return x_particles, x_jets, y


def collect_root_files(path: str, recursive: bool = True) -> List[str]:
    """Return sorted ROOT file paths from a file or directory."""
    if os.path.isfile(path):
        return [path]
    pattern = os.path.join(path, "**", "*.root") if recursive else os.path.join(path, "*.root")
    files = sorted(f for f in glob.glob(pattern, recursive=recursive) if os.path.getsize(f) > 0)
    return files


def read_files(
    paths: Union[str, Sequence[str]],
    max_num_particles: int = 128,
    particle_features: Optional[Sequence[str]] = None,
    jet_features: Optional[Sequence[str]] = None,
    labels: Optional[Sequence[str]] = None,
    lowerpt: Optional[float] = 150.0,
    upperpt: Optional[float] = None,
    class_index: int = 0,
    num_classes: int = 4,
    max_files: Optional[int] = None,
    max_jets: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and concatenate multiple ROOT files / directories via ``read_file``."""
    if isinstance(paths, str):
        path_list = [paths]
    else:
        path_list = list(paths)

    root_files: List[str] = []
    for path in path_list:
        root_files.extend(collect_root_files(path))
    root_files = sorted(set(root_files))
    if max_files is not None:
        root_files = root_files[:max_files]
    if not root_files:
        raise FileNotFoundError(f"No ROOT files found in {paths}")

    xs, js, ys = [], [], []
    n_kept = 0
    for filepath in root_files:
        x_p, x_j, y = read_file(
            filepath,
            max_num_particles=max_num_particles,
            particle_features=particle_features,
            jet_features=jet_features,
            labels=labels,
            lowerpt=lowerpt,
            upperpt=upperpt,
            class_index=class_index,
            num_classes=num_classes,
        )
        if x_j.shape[0] == 0:
            continue
        if max_jets is not None:
            remaining = max_jets - n_kept
            if remaining <= 0:
                break
            x_p = x_p[:remaining]
            x_j = x_j[:remaining]
            y = y[:remaining]
        xs.append(x_p)
        js.append(x_j)
        ys.append(y)
        n_kept += x_j.shape[0]
        if max_jets is not None and n_kept >= max_jets:
            break

    if not xs:
        f = len(particle_features or DEFAULT_PARTICLE_FEATURES)
        j = len(jet_features or DEFAULT_JET_FEATURES)
        n_c = len(labels) if labels is not None else num_classes
        return (
            np.zeros((0, f, max_num_particles), dtype=np.float32),
            np.zeros((0, j), dtype=np.float32),
            np.zeros((0, n_c), dtype=np.int32),
        )

    return np.concatenate(xs, axis=0), np.concatenate(js, axis=0), np.concatenate(ys, axis=0)


def particles_to_node_tensors(
    x_particles: np.ndarray,
    min_nodes: int = 4,
    pt_feature_index: int = 0,
    label: int = 0,
) -> Tuple[List["object"], List[int]]:
    """
    Convert JetClass-style padded ``x_particles`` (N, F, P) into PART variable-length
    node tensors ``(N_i, F)``.

    Particles with non-positive ``pt`` (at ``pt_feature_index``) are treated as padding.
    """
    import torch

    if x_particles.ndim != 3:
        raise ValueError(f"Expected x_particles shape (N, F, P), got {x_particles.shape}")

    node_tensors = []
    labels: List[int] = []
    n_jets, n_feat, _ = x_particles.shape

    for i in range(n_jets):
        # (F, P) → (P, F)
        xi = np.transpose(x_particles[i], (1, 0)).astype(np.float32)
        pt = xi[:, pt_feature_index]
        valid = np.isfinite(xi).all(axis=1) & (pt > 0)
        xi = xi[valid]
        if xi.shape[0] < min_nodes:
            continue
        node_tensors.append(torch.tensor(xi, dtype=torch.float32))
        labels.append(label)

    return node_tensors, labels
