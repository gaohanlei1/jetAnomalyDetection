import numpy as np
import awkward as ak
import uproot
import vector
vector.register_awkward()


def read_file(
        filepath,
        max_num_particles=128,
        particle_features=['part_pt', 'part_eta', 'part_phi', 'part_energy'],
        jet_features=['jet_pt', 'jet_eta', 'jet_phi', 'jet_energy'],
        labels=['label_QCD', 'label_Hbb', 'label_Hcc', 'label_Hgg', 'label_H4q',
                'label_Hqql', 'label_Zqq', 'label_Wqq', 'label_Tbqq', 'label_Tbl']):
    """Loads a single file from the JetClass dataset.

    **Arguments**

    - **filepath** : _str_
        - Path to the ROOT data file.
    - **max_num_particles** : _int_
        - The maximum number of particles to load for each jet. 
        Jets with fewer particles will be zero-padded, 
        and jets with more particles will be truncated.
    - **particle_features** : _List[str]_
        - A list of particle-level features to be loaded. 
        The available particle-level features are:
            - part_px
            - part_py
            - part_pz
            - part_energy
            - part_pt
            - part_eta
            - part_phi
            - part_deta: np.where(jet_eta>0, part_eta-jet_p4, -(part_eta-jet_p4))
            - part_dphi: delta_phi(part_phi, jet_phi)
            - part_d0val
            - part_d0err
            - part_dzval
            - part_dzerr
            - part_charge
            - part_isChargedHadron
            - part_isNeutralHadron
            - part_isPhoton
            - part_isElectron
            - part_isMuon
    - **jet_features** : _List[str]_
        - A list of jet-level features to be loaded. 
        The available jet-level features are:
            - jet_pt
            - jet_eta
            - jet_phi
            - jet_energy
            - jet_nparticles
            - jet_sdmass
            - jet_tau1
            - jet_tau2
            - jet_tau3
            - jet_tau4
    - **labels** : _List[str]_
        - A list of truth labels to be loaded. 
        The available label names are:
            - label_QCD
            - label_Hbb
            - label_Hcc
            - label_Hgg
            - label_H4q
            - label_Hqql
            - label_Zqq
            - label_Wqq
            - label_Tbqq
            - label_Tbl

    **Returns**

    - x_particles(_3-d numpy.ndarray_), x_jets(_2-d numpy.ndarray_), y(_2-d numpy.ndarray_)
        - `x_particles`: a zero-padded numpy array of particle-level features 
                         in the shape `(num_jets, num_particle_features, max_num_particles)`.
        - `x_jets`: a numpy array of jet-level features
                    in the shape `(num_jets, num_jet_features)`.
        - `y`: a one-hot encoded numpy array of the truth lables
               in the shape `(num_jets, num_classes)`.
    """

    def _pad(a, maxlen, value=0, dtype='float32'):
        if isinstance(a, np.ndarray) and a.ndim >= 2 and a.shape[1] == maxlen:
            return a
        elif isinstance(a, ak.Array):
            if a.ndim == 1:
                a = ak.unflatten(a, 1)
            a = ak.fill_none(ak.pad_none(a, maxlen, clip=True), value)
            return ak.values_astype(a, dtype)
        else:
            x = (np.ones((len(a), maxlen)) * value).astype(dtype)
            for idx, s in enumerate(a):
                if not len(s):
                    continue
                trunc = s[:maxlen].astype(dtype)
                x[idx, :len(trunc)] = trunc
            return x

    table = uproot.open(filepath)['tree'].arrays()

    p4 = vector.zip({'px': table['part_px'],
                     'py': table['part_py'],
                     'pz': table['part_pz'],
                     'energy': table['part_energy']})
    table['part_pt'] = p4.pt
    table['part_eta'] = p4.eta
    table['part_phi'] = p4.phi

    x_particles = np.stack([ak.to_numpy(_pad(table[n], maxlen=max_num_particles)) for n in particle_features], axis=1)
    x_jets = np.stack([ak.to_numpy(table[n]).astype('float32') for n in jet_features], axis=1)
    y = np.stack([ak.to_numpy(table[n]).astype('int') for n in labels], axis=1)

    return x_particles, x_jets, y


# Streaming dataset implementation used by LeJEPA training.
import os
import random
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import IterableDataset, get_worker_info

JETCLASS_LABELS = [
    "label_QCD",
    "label_Hbb",
    "label_Hcc",
    "label_Hgg",
    "label_H4q",
    "label_Hqql",
    "label_Zqq",
    "label_Wqq",
    "label_Tbqq",
    "label_Tbl",
]

DEFAULT_PARTICLE_FEATURES = [
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_pt",
    "log_pt_fraction",
    "part_deta",
    "part_dphi",
    "d0_sig",
    "dz_sig",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
]

JETCLASS_BATCH_NORMALIZED_FEATURES = [
    "log_pt_fraction",
    "d0_sig",
    "dz_sig",
]

# Actual ROOT branches needed to construct the model features above.
RAW_PARTICLE_FEATURES = [
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    "part_pt",
    "part_deta",
    "part_dphi",
    "part_d0val",
    "part_d0err",
    "part_dzval",
    "part_dzerr",
    "part_charge",
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
]

DERIVED_PARTICLE_FEATURES = {
    "log_pt_fraction",
    "d0_sig",
    "dz_sig",
}


# Jet-level features are not model inputs. They are loaded only for event
# quality cuts and to convert absolute particle pt into a per-jet pt fraction.
REQUIRED_JET_FEATURES = [
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_energy",
]

LABEL_TO_FILE_PREFIX = {
    "label_QCD": "ZJetsToNuNu",
    "label_Hbb": "HToBB",
    "label_Hcc": "HToCC",
    "label_Hgg": "HToGG",
    "label_H4q": "HToWW4Q",
    "label_Hqql": "HToWW2Q1L",
    "label_Zqq": "ZToQQ",
    "label_Wqq": "WToQQ",
    "label_Tbqq": "TTBar",
    "label_Tbl": "TTBarLep",
}


def validate_requested_labels(labels: Sequence[str]) -> None:
    """Fail early on typos instead of silently loading the wrong physics sample."""

    unknown = sorted(set(labels) - set(JETCLASS_LABELS))
    if unknown:
        raise ValueError(
            f"Unknown JetClass labels: {unknown}. Available labels: {JETCLASS_LABELS}"
        )


def discover_jetclass_files(
    split_dir: str,
    labels: Sequence[str],
) -> List[str]:
    """Return all ROOT shards belonging to the requested JetClass labels."""

    files: List[str] = []
    for label in labels:
        prefix = LABEL_TO_FILE_PREFIX[label]
        matches = sorted(glob(os.path.join(split_dir, f"{prefix}_*.root")))
        if not matches:
            raise FileNotFoundError(
                f"No ROOT files found for {label} with prefix {prefix!r} in {split_dir}."
            )
        files.extend(matches)
    return files



class JetClassIterableDataset(IterableDataset):
    """
    Stream JetClass ROOT shards lazily with class-stratified shard mixing.

    Training behavior:
        - Files are grouped by label.
        - Each label is independently sharded across all DDP ranks and workers.
        - Each label maintains its own active shard pool.
        - Events are yielded according to a shuffled balanced label cycle.
        - When one shard is exhausted, it is replaced only by another shard
          from the same label.

    This avoids long class-composition blocks caused by using a single shared
    active shard pool containing class-pure ROOT files.

    Evaluation behavior:
        - If shuffle_files=False, files and events are read deterministically.
        - Labels are still interleaved to avoid exhausting one class before
          moving to the next.
    """

    def __init__(
        self,
        split_dir: str,
        labels_to_load: Sequence[str],
        particle_features: Sequence[str],
        max_num_particles: int,
        max_events: Optional[int] = None,
        shuffle_files: bool = False,
        shuffle_active_shards: int = 1,
        infinite: bool = False,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__()

        validate_requested_labels(labels_to_load)

        if len(labels_to_load) == 0:
            raise ValueError("labels_to_load must contain at least one label.")

        self.split_dir = split_dir
        self.labels_to_load = list(labels_to_load)
        self.feature_names = list(DEFAULT_PARTICLE_FEATURES)
        self.batch_normalized_feature_names = list(
            JETCLASS_BATCH_NORMALIZED_FEATURES
        )
        self.particle_features = list(particle_features)
        if self.particle_features != self.feature_names:
            raise ValueError(
                "JetClass LeJEPA training requires the dataset-native feature order. "
                f"Expected {self.feature_names}, got {self.particle_features}."
            )
        self.max_num_particles = int(max_num_particles)
        self.max_events = max_events
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_active_shards = max(
            len(self.labels_to_load),
            int(shuffle_active_shards),
        )
        self.infinite = bool(infinite)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)

        # Keep this for compatibility / diagnostics, although __iter__ rebuilds
        # the per-label file lists explicitly.
        self.filepaths = discover_jetclass_files(
            split_dir,
            self.labels_to_load,
        )

    def __iter__(self):
        worker_info = get_worker_info()

        local_worker_id = (
            0 if worker_info is None else worker_info.id
        )
        local_num_workers = (
            1 if worker_info is None else worker_info.num_workers
        )

        global_worker_id = (
            self.rank * local_num_workers
            + local_worker_id
        )
        global_num_workers = (
            self.world_size * local_num_workers
        )

        # -------------------------------------------------------------
        # Build and shard file lists independently for every label.
        # -------------------------------------------------------------

        worker_files_by_label: Dict[str, List[str]] = {}

        for label in self.labels_to_load:
            prefix = LABEL_TO_FILE_PREFIX[label]

            label_files = sorted(
                glob(
                    os.path.join(
                        self.split_dir,
                        f"{prefix}_*.root",
                    )
                )
            )

            if len(label_files) == 0:
                raise RuntimeError(
                    f"No ROOT shards found for label {label!r} "
                    f"in {self.split_dir!r}."
                )

            worker_label_files = label_files[
                global_worker_id::global_num_workers
            ]

            if len(worker_label_files) == 0:
                raise RuntimeError(
                    "This rank/worker received no ROOT shards for one label. "
                    f"label={label!r}, "
                    f"global_worker_id={global_worker_id}, "
                    f"global_num_workers={global_num_workers}, "
                    f"num_label_files={len(label_files)}. "
                    "Reduce the total number of workers or provide more shards."
                )

            worker_files_by_label[label] = worker_label_files

        # Split max_events approximately evenly across all workers.
        worker_quota: Optional[int] = None

        if self.max_events is not None:
            base = self.max_events // global_num_workers
            remainder = self.max_events % global_num_workers

            worker_quota = (
                base
                + int(global_worker_id < remainder)
            )

        # Divide active-shard budget across labels.
        #
        # Example:
        #   3 labels, shuffle_active_shards=3 -> 1 shard per label
        #   3 labels, shuffle_active_shards=6 -> 2 shards per label
        num_labels = len(self.labels_to_load)

        base_active_per_label = (
            self.shuffle_active_shards // num_labels
        )
        active_remainder = (
            self.shuffle_active_shards % num_labels
        )

        active_budget_by_label = {
            label: (
                base_active_per_label
                + int(label_idx < active_remainder)
            )
            for label_idx, label in enumerate(self.labels_to_load)
        }

        pass_index = 0

        while True:
            rng = random.Random(
                self.seed
                + 9176 * global_worker_id
                + 104729 * pass_index
            )

            files_by_label: Dict[str, List[str]] = {
                label: list(files)
                for label, files in worker_files_by_label.items()
            }

            if self.shuffle_files:
                for label in self.labels_to_load:
                    rng.shuffle(files_by_label[label])

            yielded = 0

            # =========================================================
            # Deterministic finite evaluation
            # =========================================================

            if not self.shuffle_files:
                loaded_by_label: Dict[str, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

                for label in self.labels_to_load:
                    label_events: List[Tuple[torch.Tensor, torch.Tensor]] = []

                    for filepath in files_by_label[label]:
                        x_particles, y = (
                            load_and_preprocess_jetclass_file(
                                filepath=filepath,
                                particle_features=self.particle_features,
                                max_num_particles=self.max_num_particles,
                            )
                        )

                        label_events.extend(
                            (
                                torch.from_numpy(event_x.copy()),
                                torch.from_numpy(event_y.copy()),
                            )
                            for event_x, event_y in zip(x_particles, y)
                        )

                    loaded_by_label[label] = label_events

                cursors = {
                    label: 0
                    for label in self.labels_to_load
                }

                # Deterministic round-robin over labels.
                while True:
                    made_progress = False

                    for label in self.labels_to_load:
                        cursor = cursors[label]
                        events = loaded_by_label[label]

                        if cursor >= len(events):
                            continue

                        if (
                            worker_quota is not None
                            and yielded >= worker_quota
                        ):
                            break

                        yield events[cursor]

                        cursors[label] += 1
                        yielded += 1
                        made_progress = True

                    if (
                        worker_quota is not None
                        and yielded >= worker_quota
                    ):
                        break

                    if not made_progress:
                        break

            # =========================================================
            # Class-balanced shuffled training stream
            # =========================================================

            else:
                file_iters_by_label = {
                    label: iter(files_by_label[label])
                    for label in self.labels_to_load
                }

                active_by_label: Dict[str, List[Dict[str, object]]] = {
                    label: []
                    for label in self.labels_to_load
                }

                def load_next_shard(label: str):
                    try:
                        filepath = next(
                            file_iters_by_label[label]
                        )
                    except StopIteration:
                        return None

                    x_particles, y = (
                        load_and_preprocess_jetclass_file(
                            filepath=filepath,
                            particle_features=self.particle_features,
                            max_num_particles=self.max_num_particles,
                            eps=1e-8,
                        )
                    )

                    order = np.arange(
                        len(x_particles),
                        dtype=np.int64,
                    )

                    np.random.default_rng(
                        rng.randrange(2**32)
                    ).shuffle(order)

                    return {
                        "x": x_particles,
                        "y": y,
                        "order": order,
                        "cursor": 0,
                        "filepath": filepath,
                    }

                # Initialize one independent active pool per label.
                for label in self.labels_to_load:
                    budget = active_budget_by_label[label]

                    for _ in range(
                        min(
                            budget,
                            len(files_by_label[label]),
                        )
                    ):
                        shard = load_next_shard(label)

                        if shard is not None:
                            active_by_label[label].append(shard)

                # Labels that currently still have available active shards.
                available_labels = [
                    label
                    for label in self.labels_to_load
                    if len(active_by_label[label]) > 0
                ]

                while available_labels:
                    if (
                        worker_quota is not None
                        and yielded >= worker_quota
                    ):
                        break

                    # Balanced cycle:
                    # every cycle contains each currently available label once,
                    # but the ordering of labels inside the cycle is random.
                    label_cycle = list(available_labels)
                    rng.shuffle(label_cycle)

                    for label in label_cycle:
                        if (
                            worker_quota is not None
                            and yielded >= worker_quota
                        ):
                            break

                        active_label_shards = active_by_label[label]

                        if len(active_label_shards) == 0:
                            continue

                        shard_idx = rng.randrange(
                            len(active_label_shards)
                        )
                        shard = active_label_shards[shard_idx]

                        cursor = int(shard["cursor"])
                        order = shard["order"]

                        event_idx = int(order[cursor])
                        shard["cursor"] = cursor + 1

                        x_particles = shard["x"]
                        y = shard["y"]

                        yield (
                            torch.from_numpy(
                                x_particles[event_idx].copy()
                            ),
                            torch.from_numpy(
                                y[event_idx].copy()
                            ),
                        )

                        yielded += 1

                        # Replace exhausted shard only with a new shard
                        # from the same label.
                        if shard["cursor"] >= len(order):
                            active_label_shards.pop(shard_idx)
                            del shard
                            del x_particles
                            del y
                            del order
                            
                            replacement = load_next_shard(label)

                            if replacement is not None:
                                active_label_shards.append(replacement)

                    available_labels = [
                        label
                        for label in self.labels_to_load
                        if len(active_by_label[label]) > 0
                    ]

            if not self.infinite:
                break

            if yielded == 0:
                raise RuntimeError(
                    "Infinite JetClass stream completed a pass "
                    "without yielding any events. "
                    f"global_worker_id={global_worker_id}, "
                    f"global_num_workers={global_num_workers}."
                )

            pass_index += 1
            
            
def load_and_preprocess_jetclass_file(
    filepath: str,
    particle_features: Sequence[str],
    max_num_particles: int,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load one ROOT shard and construct the requested model feature tensor.

    The returned tensor contains untouched px/py/pz/energy, pt fraction,
    log(pt fraction), relative angular coordinates, clipped impact-parameter
    significances, charge, and identity indicators. Absolute constituent eta/phi and the four raw impact-parameter value/error
    branches are never returned.
    """

    requested = list(particle_features)
    supported = set(DEFAULT_PARTICLE_FEATURES)
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(
            f"Unsupported model particle features: {unknown}. "
            f"Supported features are {DEFAULT_PARTICLE_FEATURES}."
        )

    loaded_particle_features = list(RAW_PARTICLE_FEATURES)
    feature_index = {
        name: i for i, name in enumerate(loaded_particle_features)
    }

    x_particles, x_jets, y = read_file(
        filepath,
        max_num_particles=max_num_particles,
        particle_features=loaded_particle_features,
        jet_features=REQUIRED_JET_FEATURES,
        labels=JETCLASS_LABELS,
    )

    x_particles = np.transpose(x_particles, (0, 2, 1)).astype(
        np.float32,
        copy=False,
    )
    x_jets = np.asarray(x_jets, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    jet_pt = x_jets[:, 0]
    jet_eta = x_jets[:, 1]
    jet_energy = x_jets[:, 3]
    valid_jets = (
        (jet_pt > 0)
        & (jet_energy > 0)
        & (jet_eta >= -2.5)
        & (jet_eta <= 2.5)
    )

    x_particles = x_particles[valid_jets]
    y = y[valid_jets]
    jet_pt = jet_pt[valid_jets]

    part_pt = x_particles[..., feature_index["part_pt"]]
    part_energy = x_particles[..., feature_index["part_energy"]]
    part_deta = x_particles[..., feature_index["part_deta"]]
    part_dphi = x_particles[..., feature_index["part_dphi"]]

    valid_particles = (
        (part_pt > 0)
        & (part_energy > 0)
        & (np.sqrt(np.square(part_deta) + np.square(part_dphi)) < 0.8)
    )

    # Raw invalid rows are zeroed before derived quantities are exported.
    x_particles[~valid_particles] = 0.0

    safe_jet_pt = np.where(jet_pt > 0, jet_pt, 1.0).astype(np.float32)
    pt_fraction = (
        x_particles[..., feature_index["part_pt"]]
        / safe_jet_pt[:, None]
    ).astype(np.float32, copy=False)
    pt_fraction[~valid_particles] = 0.0

    log_pt_fraction = np.zeros_like(pt_fraction, dtype=np.float32)
    log_pt_fraction[valid_particles] = np.log(
        np.clip(pt_fraction[valid_particles], eps, None)
    )

    d0val = x_particles[..., feature_index["part_d0val"]]
    d0err = x_particles[..., feature_index["part_d0err"]]
    dzval = x_particles[..., feature_index["part_dzval"]]
    dzerr = x_particles[..., feature_index["part_dzerr"]]

    d0_sig = np.zeros_like(pt_fraction, dtype=np.float32)
    dz_sig = np.zeros_like(pt_fraction, dtype=np.float32)
    d0_sig[valid_particles] = np.clip(
        d0val[valid_particles] / np.clip(d0err[valid_particles], eps, None),
        -20.0,
        20.0,
    )
    dz_sig[valid_particles] = np.clip(
        dzval[valid_particles] / np.clip(dzerr[valid_particles], eps, None),
        -20.0,
        20.0,
    )

    feature_arrays: Dict[str, np.ndarray] = {
        name: x_particles[..., index]
        for name, index in feature_index.items()
        if name not in {
            "part_pt",
            "part_d0val",
            "part_d0err",
            "part_dzval",
            "part_dzerr",
        }
    }
    feature_arrays.update(
        {
            "part_pt": pt_fraction,
            "log_pt_fraction": log_pt_fraction,
            "d0_sig": d0_sig,
            "dz_sig": dz_sig,
        }
    )

    output = np.stack(
        [feature_arrays[name] for name in requested],
        axis=-1,
    ).astype(np.float32, copy=False)
    output[~valid_particles] = 0.0

    return (
        np.ascontiguousarray(output, dtype=np.float32),
        np.ascontiguousarray(y, dtype=np.float32),
    )


def collate_jetclass_tensors(
    batch: Sequence[Tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Stack fixed-size jets and derive padding from all-zero particle rows."""

    xs, ys = zip(*batch)
    x_particles = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    padding_mask = x_particles.eq(0).all(dim=-1)
    return {
        "x_particles": x_particles,
        "padding_mask": padding_mask,
        "y": y,
    }



__all__ = [
    "DEFAULT_PARTICLE_FEATURES",
    "JETCLASS_BATCH_NORMALIZED_FEATURES",
    "JETCLASS_LABELS",
    "JetClassIterableDataset",
    "collate_jetclass_tensors",
    "discover_jetclass_files",
    "load_and_preprocess_jetclass_file",
    "read_file",
    "validate_requested_labels",
]
