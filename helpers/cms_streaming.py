"""
Streaming CMS DeepNTuplizer ROOT datasets (JetClass-style infinite shards).

Reads one ROOT file at a time via ``dataloader.read_file`` so training does not
materialize the full sample mix in RAM. An "epoch" is defined by the caller as
a fixed number of optimizer steps (``--steps-per-epoch``).
"""

from __future__ import annotations

import os
import random
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from dataloader import collect_root_files, particles_to_node_tensors, read_file


def discover_cms_files_by_class(
    class_dirs: Sequence[str],
) -> Dict[int, List[str]]:
    """Map class_id -> sorted ROOT paths under each class directory."""
    files_by_class: Dict[int, List[str]] = {}
    for class_id, path in enumerate(class_dirs):
        files = collect_root_files(path)
        if not files:
            raise FileNotFoundError(f"No ROOT files found for class {class_id} in {path}")
        files_by_class[class_id] = files
    return files_by_class


def split_files_train_val(
    files_by_class: Dict[int, List[str]],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """Deterministic per-class file split into train / val pools."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")

    train: Dict[int, List[str]] = {}
    val: Dict[int, List[str]] = {}
    rng = random.Random(seed)
    for class_id, files in files_by_class.items():
        shuffled = list(files)
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train[class_id] = shuffled
            val[class_id] = list(shuffled)
            continue
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        n_val = min(n_val, len(shuffled) - 1)
        val[class_id] = shuffled[:n_val]
        train[class_id] = shuffled[n_val:]
    return train, val


class CMSIterableDataset(IterableDataset):
    """
    Stream CMS ROOT shards with class-balanced mixing.

    Yields ``(node_tensor [N,F], class_id int)`` compatible with
    ``collate_node_tensors``.
    """

    def __init__(
        self,
        files_by_class: Dict[int, List[str]],
        particle_features: Sequence[str],
        max_num_particles: int = 128,
        min_nodes: int = 4,
        lowerpt: Optional[float] = 200.0,
        upperpt: Optional[float] = 400.0,
        num_classes: Optional[int] = None,
        max_events: Optional[int] = None,
        shuffle_files: bool = True,
        shuffle_active_shards: int = 4,
        infinite: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        if not files_by_class:
            raise ValueError("files_by_class must be non-empty.")

        self.files_by_class = {
            int(k): list(v) for k, v in files_by_class.items()
        }
        self.class_ids = sorted(self.files_by_class.keys())
        self.particle_features = list(particle_features)
        self.max_num_particles = int(max_num_particles)
        self.min_nodes = int(min_nodes)
        self.lowerpt = lowerpt
        self.upperpt = upperpt
        self.num_classes = int(
            num_classes if num_classes is not None else (max(self.class_ids) + 1)
        )
        self.max_events = max_events
        self.shuffle_files = bool(shuffle_files)
        self.shuffle_active_shards = max(
            len(self.class_ids),
            int(shuffle_active_shards),
        )
        self.infinite = bool(infinite)
        self.seed = int(seed)
        self.pt_index = (
            self.particle_features.index("pt")
            if "pt" in self.particle_features
            else 0
        )

    def _load_shard_events(
        self,
        filepath: str,
        class_id: int,
    ) -> List[Tuple[torch.Tensor, int]]:
        x_particles, _, _ = read_file(
            filepath,
            max_num_particles=self.max_num_particles,
            particle_features=self.particle_features,
            lowerpt=self.lowerpt,
            upperpt=self.upperpt,
            class_index=class_id,
            num_classes=self.num_classes,
        )
        if x_particles.shape[0] == 0:
            return []
        nodes, labels = particles_to_node_tensors(
            x_particles,
            min_nodes=self.min_nodes,
            pt_feature_index=self.pt_index,
            label=class_id,
        )
        return list(zip(nodes, labels))

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, int]]:
        worker_info = get_worker_info()
        local_worker_id = 0 if worker_info is None else worker_info.id
        local_num_workers = 1 if worker_info is None else worker_info.num_workers
        global_worker_id = local_worker_id
        global_num_workers = local_num_workers

        worker_files_by_class: Dict[int, List[str]] = {}
        for class_id in self.class_ids:
            label_files = self.files_by_class[class_id]
            worker_label_files = label_files[global_worker_id::global_num_workers]
            if not worker_label_files:
                # Fall back to full list so single-file classes still stream.
                worker_label_files = list(label_files)
            worker_files_by_class[class_id] = worker_label_files

        worker_quota: Optional[int] = None
        if self.max_events is not None:
            base = self.max_events // global_num_workers
            remainder = self.max_events % global_num_workers
            worker_quota = base + int(global_worker_id < remainder)

        num_classes = len(self.class_ids)
        base_active = self.shuffle_active_shards // num_classes
        active_remainder = self.shuffle_active_shards % num_classes
        active_budget_by_class = {
            class_id: base_active + int(idx < active_remainder)
            for idx, class_id in enumerate(self.class_ids)
        }

        pass_index = 0
        while True:
            rng = random.Random(
                self.seed + 9176 * global_worker_id + 104729 * pass_index
            )
            files_by_class = {
                class_id: list(files)
                for class_id, files in worker_files_by_class.items()
            }
            if self.shuffle_files:
                for class_id in self.class_ids:
                    rng.shuffle(files_by_class[class_id])

            yielded = 0

            if not self.shuffle_files:
                # Finite deterministic round-robin for val/eval.
                loaded: Dict[int, List[Tuple[torch.Tensor, int]]] = {}
                for class_id in self.class_ids:
                    events: List[Tuple[torch.Tensor, int]] = []
                    for filepath in files_by_class[class_id]:
                        events.extend(self._load_shard_events(filepath, class_id))
                        if worker_quota is not None and len(events) >= worker_quota:
                            break
                    loaded[class_id] = events
                cursors = {class_id: 0 for class_id in self.class_ids}
                while True:
                    made_progress = False
                    for class_id in self.class_ids:
                        cursor = cursors[class_id]
                        events = loaded[class_id]
                        if cursor >= len(events):
                            continue
                        if worker_quota is not None and yielded >= worker_quota:
                            break
                        yield events[cursor]
                        cursors[class_id] += 1
                        yielded += 1
                        made_progress = True
                    if worker_quota is not None and yielded >= worker_quota:
                        break
                    if not made_progress:
                        break
            else:
                file_iters = {
                    class_id: iter(files_by_class[class_id])
                    for class_id in self.class_ids
                }
                active: Dict[int, List[Dict[str, object]]] = {
                    class_id: [] for class_id in self.class_ids
                }

                def load_next_shard(class_id: int) -> Optional[Dict[str, object]]:
                    try:
                        filepath = next(file_iters[class_id])
                    except StopIteration:
                        return None
                    events = self._load_shard_events(filepath, class_id)
                    if not events:
                        return load_next_shard(class_id)
                    order = np.arange(len(events), dtype=np.int64)
                    np.random.default_rng(rng.randrange(2**32)).shuffle(order)
                    return {
                        "events": events,
                        "order": order,
                        "cursor": 0,
                        "filepath": filepath,
                    }

                for class_id in self.class_ids:
                    budget = active_budget_by_class[class_id]
                    for _ in range(min(budget, len(files_by_class[class_id]))):
                        shard = load_next_shard(class_id)
                        if shard is not None:
                            active[class_id].append(shard)

                available = [
                    class_id for class_id in self.class_ids if active[class_id]
                ]
                while available:
                    if worker_quota is not None and yielded >= worker_quota:
                        break
                    cycle = list(available)
                    rng.shuffle(cycle)
                    for class_id in cycle:
                        if worker_quota is not None and yielded >= worker_quota:
                            break
                        shards = active[class_id]
                        if not shards:
                            continue
                        shard_idx = rng.randrange(len(shards))
                        shard = shards[shard_idx]
                        cursor = int(shard["cursor"])
                        order = shard["order"]
                        event_idx = int(order[cursor])
                        shard["cursor"] = cursor + 1
                        yield shard["events"][event_idx]
                        yielded += 1
                        if shard["cursor"] >= len(order):
                            replacement = load_next_shard(class_id)
                            if replacement is None:
                                shards.pop(shard_idx)
                            else:
                                shards[shard_idx] = replacement
                    available = [
                        class_id for class_id in self.class_ids if active[class_id]
                    ]

            if not self.infinite:
                break
            if yielded == 0:
                raise RuntimeError(
                    "CMS stream completed a pass without yielding events. "
                    f"worker={global_worker_id}/{global_num_workers}, "
                    f"classes={self.class_ids}."
                )
            pass_index += 1
