#!/usr/bin/env python3
"""Score held-out real jets against a balanced, non-repeating train kNN pool.

The run manifest supplies training files for the saved background labels and
label_Real test files. Each reference class contributes the same number of
jets, stopping when the requested quota or the smallest class is exhausted.
The model is frozen. Exact Euclidean kNN scores select the top-fraction events.
Plot-only mode retains the saved event-array interface.
"""

from __future__ import annotations

import argparse
import heapq
import inspect
import json
import math
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from visualize.plot_random_jet_events import plot_random_jet_events

from datasets import cms_streaming
from models.part_jetclass import (
    CorruptedNegativeAugmentationConfig,
    LeJEPALossConfig,
    LeJEPASemiSupervisedTripletParticleTransformerRepresentation,
    MultiViewAugmentationConfig,
    ParticleTransformerConfig,
    SemiSupervisedLossConfig,
    TripletLossConfig,
)

REAL_LABEL = "label_Real"
FOUR_VECTOR_FEATURES = (
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
)
def _construct_with_supported_kwargs(factory, **kwargs):
    parameters = inspect.signature(factory).parameters
    return factory(**{key: value for key, value in kwargs.items() if key in parameters})


def _require_summary_list(summary: Mapping[str, object], key: str) -> List[str]:
    if key not in summary or summary[key] is None:
        raise KeyError(f"summary.json is missing required field {key!r}.")
    value = summary[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"summary.json field {key!r} must be a list.")
    result = [str(item) for item in value]
    if not result:
        raise ValueError(f"summary.json field {key!r} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"summary.json field {key!r} contains duplicates: {result}")
    return result


def _validate_four_vector_prefix(features: Sequence[str], source: str) -> None:
    if tuple(features[:4]) != FOUR_VECTOR_FEATURES:
        raise ValueError(
            f"{source} must begin with {list(FOUR_VECTOR_FEATURES)}, "
            f"found {list(features[:4])}."
        )


def _strip_common_prefix(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a balanced train kNN pool, score held-out real test jets, "
            "and save the top anomalies."
        )
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        help="Fine-tuning run containing summary.json and cms_split_manifest.json.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint override; defaults to <run_dir>/best_model.pth.",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--cms-split-manifest", type=Path, default=None)
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help=(
            "Optional batch cap for debugging. By default the complete finite "
            "label_Real test split is processed."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to the saved per-rank training batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Requested workers (defaults to training setting); evaluation overrides this to 1.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--reference-steps", type=int, default=100,
                        help="Requested pool size = steps * batch-size, rounded down to full class cycles.")
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--knn-reduction", choices=("mean", "kth"), default="mean")
    parser.add_argument("--knn-reference-batch-size", type=int, default=8192)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--bottom-fraction", type=float, default=None,
                        help="Lowest-score fraction to retain; defaults to --top-fraction.")
    parser.add_argument("--max-num-particles", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-visualize", type=int, default=12)
    parser.add_argument(
        "--histogram-sample-size",
        type=int,
        default=1_000_000,
        help="Random score sample used only to choose robust histogram bounds.",
    )
    parser.add_argument(
        "--heap-progress-every",
        type=int,
        default=100_000,
        help="Event interval for reporting top-event selection progress.",
    )

    # Plot-only mode.
    parser.add_argument(
        "--plot-events-npy",
        type=Path,
        default=None,
        help="Enable plot-only mode from an existing (E, N, F) .npy file.",
    )
    parser.add_argument(
        "--feature-list",
        type=Path,
        default=None,
        help="Feature-name JSON for --plot-events-npy.",
    )
    parser.add_argument(
        "--event-scores",
        type=Path,
        default=None,
        help="Optional score .npy aligned with --plot-events-npy.",
    )
    parser.add_argument("--plot-output", type=Path, default=None)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {value}")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"MPS requested but unavailable: {value}")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_to_dtype(precision: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[precision]


def autocast_context(device: torch.device, precision: str):
    return torch.autocast(
        device_type=device.type,
        dtype=precision_to_dtype(precision),
        enabled=device.type == "cuda" and precision in {"bf16", "fp16"},
    )


def _feature_index(
    features: Sequence[str], candidates: Sequence[str], fallback: str
) -> int:
    for name in candidates:
        if name in features:
            return features.index(name)
    return features.index(fallback)


class RealTestBackend:
    def __init__(
        self,
        *,
        summary: Mapping[str, object],
        run_dir: Path,
        dataset_root_override: Optional[Path],
        manifest_override: Optional[Path],
        max_num_particles_override: Optional[int],
    ) -> None:
        dataset_name = str(summary.get("dataset", ""))
        if dataset_name != "cms":
            raise ValueError(
                f"label_Real evaluation requires a CMS run; summary has {dataset_name!r}."
            )

        backgrounds = [str(x) for x in summary.get("background_labels", [])]
        signals = [str(x) for x in summary.get("signal_labels", [])]
        if not backgrounds or REAL_LABEL not in backgrounds:
            raise ValueError("The fine-tuning background labels must include label_Real.")
        self.reference_labels = list(dict.fromkeys(backgrounds))
        if signals:
            warnings.warn(
                f"Ignoring configured signal labels {signals}; this evaluator reads only label_Real/test."
            )

        if dataset_root_override is not None:
            self.dataset_root = dataset_root_override.expanduser().resolve()
        else:
            self.dataset_root = Path(str(summary["dataset_root"])).expanduser().resolve()
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"CMS dataset root does not exist: {self.dataset_root}")

        self.summary = summary
        self.run_dir = run_dir
        self.feature_names = _require_summary_list(summary, "particle_features")
        _validate_four_vector_prefix(self.feature_names, "Saved particle feature list")
        self.standardized_feature_names = _require_summary_list(
            summary, "batch_standardized_particle_features"
        )
        missing_standardized = sorted(
            set(self.standardized_feature_names) - set(self.feature_names)
        )
        if missing_standardized:
            raise ValueError(
                "Batch-standardized features are absent from particle_features: "
                f"{missing_standardized}"
            )
        unknown_features = sorted(
            set(self.feature_names) - set(cms_streaming.AVAILABLE_PARTICLE_FEATURES)
        )
        if unknown_features:
            raise ValueError(
                f"Current cms_streaming cannot reconstruct saved features: {unknown_features}"
            )

        self.label_axis = [
            str(x) for x in summary.get("dataset_label_axis", cms_streaming.CMS_LABELS)
        ]
        if REAL_LABEL not in self.label_axis:
            raise ValueError(
                f"Saved dataset label axis does not contain {REAL_LABEL}: {self.label_axis}"
            )
        self.max_num_particles = int(
            max_num_particles_override
            if max_num_particles_override is not None
            else summary.get("max_num_particles", 128)
        )
        self.min_nodes = int(summary.get("min_nodes", 4))
        self.shuffle_active_shards = int(summary.get("shuffle_active_shards", 3))
        self.pt_min = summary.get("cms_pt_min")
        self.pt_max = summary.get("cms_pt_max")

        manifest_path = (
            manifest_override.expanduser().resolve()
            if manifest_override is not None
            else run_dir / "cms_split_manifest.json"
        )
        if not manifest_path.is_file():
            raise FileNotFoundError(
                "Real-data evaluation requires the fresh fine-tuning manifest: "
                f"{manifest_path}"
            )
        with manifest_path.open() as handle:
            manifest = json.load(handle)
        if int(manifest.get("version", 0)) != 2:
            raise ValueError(f"Unsupported CMS manifest version: {manifest_path}")
        if "splits" not in manifest:
            raise ValueError(f"CMS manifest has no 'splits': {manifest_path}")
        self.manifest_path = manifest_path
        self.manifest_sha256 = str(manifest.get("sha256", "")) or None
        expected_hash = summary.get("cms_split_manifest_sha256")
        if expected_hash is not None and str(expected_hash) != self.manifest_sha256:
            raise RuntimeError(
                "Fine-tuning summary and cms_split_manifest.json disagree: "
                f"summary={expected_hash}, manifest={self.manifest_sha256}."
            )

        self.splits = self._resolve_manifest_paths(manifest["splits"])
        test_labels = self.splits.get("test", {})
        if REAL_LABEL not in test_labels:
            raise ValueError(
                "The test split must contain label_Real for this evaluator; "
                f"found {sorted(test_labels)}."
            )
        self.test_files = list(test_labels[REAL_LABEL])
        if not self.test_files:
            raise ValueError("The label_Real test split contains no ROOT shards.")

    def make_reference_loader(self, *, seed, batch_size, max_events, pin_memory):
        train = self.splits["train"]
        for label in self.reference_labels:
            if not train.get(label):
                raise ValueError(f"Missing train files for reference class {label}")
        reference_paths = [p for label in self.reference_labels for p in train[label]]
        if len(reference_paths) != len(set(reference_paths)):
            raise ValueError("Repeated file in reference pool manifest.")
        if set(reference_paths).intersection(self.test_files):
            raise ValueError("Reference train files overlap real test files.")
        dataset = cms_streaming.CMSIterableDataset(
            files_by_label=train, labels_to_load=self.reference_labels,
            label_axis=self.label_axis, particle_features=self.feature_names,
            max_num_particles=self.max_num_particles, min_nodes=self.min_nodes,
            lowerpt=self.pt_min, upperpt=self.pt_max, max_events=max_events,
            shuffle_files=True, infinite=False, seed=seed,
            stop_on_first_exhaustion=True,
        )
        return DataLoader(dataset, batch_size=batch_size, num_workers=1,
                          collate_fn=cms_streaming.collate_cms_tensors,
                          drop_last=False, pin_memory=pin_memory,
                          multiprocessing_context="spawn", prefetch_factor=1)

    def _resolve_manifest_paths(
        self, splits: Mapping[str, Mapping[str, Sequence[str]]]
    ) -> Dict[str, Dict[str, List[str]]]:
        old_root_value = self.summary.get("dataset_root")
        old_root = (
            Path(str(old_root_value)).expanduser()
            if old_root_value is not None
            else None
        )
        resolved: Dict[str, Dict[str, List[str]]] = {}
        missing: List[str] = []
        for split_name, label_map in splits.items():
            resolved[split_name] = {}
            for label, raw_paths in label_map.items():
                if isinstance(raw_paths, Mapping):
                    raise ValueError(
                        "Nested production-family manifests are unsupported by the shuffled-shard pipeline."
                    )
                paths: List[str] = []
                for raw_path in raw_paths:
                    original = Path(str(raw_path)).expanduser()
                    candidate: Optional[Path] = None
                    if original.is_file():
                        candidate = original.resolve()
                    elif old_root is not None:
                        try:
                            relative = original.relative_to(old_root)
                        except ValueError:
                            relative = None
                        if relative is not None:
                            relocated = (self.dataset_root / relative).resolve()
                            if relocated.is_file():
                                candidate = relocated
                    if candidate is None:
                        directory = cms_streaming.CMS_LABEL_TO_DIRECTORY[label]
                        relocated = (self.dataset_root / directory / original.name).resolve()
                        candidate = relocated
                    if not candidate.is_file():
                        missing.append(str(candidate))
                    paths.append(str(candidate))
                resolved[split_name][label] = paths
        if missing:
            preview = "\n  ".join(missing[:10])
            raise FileNotFoundError(
                "CMS manifest references missing ROOT shards. First missing paths:\n  "
                + preview
            )
        return resolved

    def make_loader(
        self,
        *,
        seed: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> DataLoader:
        dataset = _construct_with_supported_kwargs(
            cms_streaming.CMSIterableDataset,
            files_by_label=self.splits["test"],
            labels_to_load=[REAL_LABEL],
            label_axis=self.label_axis,
            particle_features=self.feature_names,
            max_num_particles=self.max_num_particles,
            min_nodes=self.min_nodes,
            lowerpt=self.pt_min,
            upperpt=self.pt_max,
            max_events=None,
            shuffle_files=True,
            shuffle_active_shards=self.shuffle_active_shards,
            infinite=False,
            seed=seed,
            rank=0,
            world_size=1,
            include_event_metadata=True,
        )
        if list(dataset.feature_names) != self.feature_names:
            raise RuntimeError(
                "Dataset feature order disagrees with summary.json: "
                f"dataset={dataset.feature_names}, summary={self.feature_names}"
            )
        kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": cms_streaming.collate_cms_tensors,
            "persistent_workers": False,
            "drop_last": False,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = 1
            kwargs["multiprocessing_context"] = "spawn"
        return DataLoader(**kwargs)


def build_model(
    summary: Mapping[str, object],
    backend: RealTestBackend,
    device: torch.device,
) -> torch.nn.Module:
    features = list(backend.feature_names)
    standardized = list(backend.standardized_feature_names)
    precision = str(summary.get("precision", "fp32"))

    model_name = str(summary.get("model", "semi-sup-triplet"))
    if model_name != "semi-sup-triplet":
        raise ValueError(f"Unsupported saved model variant: {model_name!r}")

    model_config = _construct_with_supported_kwargs(
        ParticleTransformerConfig,
        input_dim=len(features),
        input_feature_names=tuple(features),
        standardized_feature_names=tuple(standardized),
        embed_dim=int(summary["embed_dim"]),
        num_heads=int(summary["num_heads"]),
        num_layers=int(summary["num_layers"]),
        num_class_layers=int(summary.get("num_class_layers", 2)),
        ffn_mult=int(summary.get("ffn_mult", 4)),
        dropout=float(summary.get("dropout", 0.1)),
        class_dropout=float(summary.get("class_dropout", 0.0)),
        representation_dim=int(summary["representation_dim"]),
        use_pairwise_bias=bool(summary.get("use_pairwise_bias", True)),
        pairwise_hidden_dim=int(summary.get("pairwise_hidden_dim", 64)),
        pairwise_num_features=int(summary.get("pairwise_num_features", 4)),
        compute_dtype=precision_to_dtype(precision),
        use_internal_autocast=False,
        eps=float(summary.get("eps", 1e-8)),
        feature_norm_momentum=float(summary.get("feature_norm_momentum", 0.9)),
    )

    global_range = summary.get("global_drop_pt_frac_range", [0.0, 0.5])
    local_range = summary.get("local_drop_pt_frac_range", [0.5, 0.95])
    augmentation_config = MultiViewAugmentationConfig(
        num_global_views=int(summary.get("num_global_views", 2)),
        num_local_views=int(summary.get("num_local_views", 6)),
        global_drop_pt_frac_range=(float(global_range[0]), float(global_range[1])),
        local_drop_pt_frac_range=(float(local_range[0]), float(local_range[1])),
        min_nodes=int(summary.get("min_nodes", 4)),
        px_index=features.index("part_px"),
        py_index=features.index("part_py"),
        pz_index=features.index("part_pz"),
        energy_index=features.index("part_energy"),
        pt_index=features.index("part_pt"),
        log_pt_fraction_index=features.index("log_pt_fraction"),
        eps=float(summary.get("eps", 1e-8)),
        pt_drop_power=float(summary.get("pt_drop_power", 1.0)),
        zero_dropped_features=not bool(summary.get("keep_dropped_features", False)),
    )
    negative_config = _construct_with_supported_kwargs(
        CorruptedNegativeAugmentationConfig,
        num_negative_views=int(summary.get("num_negative_views", 4)),
        batch_mix_prob=float(summary.get("batch_mix_prob", 0.45)),
        pt_resample_prob=float(summary.get("pt_resample_prob", 0.25)),
        node_deta_dphi_rotation_prob=float(
            summary.get("node_deta_dphi_rotation_prob", 0.20)
        ),
        deta_dphi_shuffle_prob=float(summary.get("deta_dphi_shuffle_prob", 0.05)),
        identity_shuffle_prob=float(summary.get("identity_shuffle_prob", 0.05)),
        min_nodes=int(summary.get("min_nodes", 4)),
        eps=float(summary.get("eps", 1e-8)),
        deta_index=features.index("part_deta"),
        dphi_index=features.index("part_dphi"),
        pt_index=features.index("part_pt"),
        log_pt_fraction_index=features.index("log_pt_fraction"),
        d0_sig_index=_feature_index(
            features, ("d0_sig", "Cpfcan_dxysig"), "part_charge"
        ),
        dz_sig_index=_feature_index(
            features, ("dz_sig", "Cpfcan_dzsig"), "part_charge"
        ),
        charge_index=features.index("part_charge"),
        identity_start_index=features.index("part_isChargedHadron"),
        identity_end_index=features.index("part_isMuon") + 1,
        corrupt_node_frac=float(summary.get("corrupt_node_frac", 0.5)),
        batch_mix_anchor_frac_min=float(summary.get("batch_mix_anchor_frac_min", 0.3)),
        batch_mix_anchor_frac_max=float(summary.get("batch_mix_anchor_frac_max", 0.7)),
        renormalize_pt_sum=bool(summary.get("renormalize_negative_pt_sum", True)),
    )
    loss_config = LeJEPALossConfig(
        invariant_weight=float(summary.get("invariant_weight", 1.0)),
        sigreg_weight=float(summary.get("sigreg_weight", 0.02)),
        epps_pulley_num_points=int(summary.get("epps_pulley_num_points", 17)),
        num_slices=int(summary.get("num_slices", 1024)),
    )
    triplet_config = TripletLossConfig(
        triplet_weight=float(summary.get("triplet_weight", 0.1)),
        triplet_margin=float(summary.get("triplet_margin", 1.0)),
        normalize_representations_for_triplet=bool(
            summary.get("normalize_representations_for_triplet", False)
        ),
        use_global_views_as_positives=not bool(
            summary.get("use_all_views_as_triplet_positives", False)
        ),
    )
    model = LeJEPASemiSupervisedTripletParticleTransformerRepresentation(
        model_config=model_config,
        augmentation_config=augmentation_config,
        negative_augmentation_config=negative_config,
        loss_config=loss_config,
        triplet_loss_config=triplet_config,
        semi_supervised_config=SemiSupervisedLossConfig(
            classification_weight=0.0,
            num_classes=1,
        ),
    )
    return model.to(device)


def read_state_dict(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state_dict)}")
    cleaned = dict(state_dict)
    for prefix in ("module.", "model."):
        cleaned = _strip_common_prefix(cleaned, prefix)
    return cleaned


def load_model_weights(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    state_dict = read_state_dict(checkpoint_path, device)
    current = model.state_dict()
    filtered: Dict[str, torch.Tensor] = {}
    dropped_classification: List[str] = []
    shape_errors: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []
    for key, value in state_dict.items():
        if key in current and current[key].shape != value.shape:
            if key.startswith("classification_head."):
                dropped_classification.append(key)
                continue
            shape_errors.append((key, tuple(value.shape), tuple(current[key].shape)))
            continue
        filtered[key] = value
    if shape_errors:
        raise ValueError(
            "Checkpoint has non-classification shape mismatches: "
            f"{shape_errors[:10]}"
        )

    result = model.load_state_dict(filtered, strict=False)
    feature_stat_suffixes = (
        "_feature_running_mean",
        "_feature_running_var",
        "_feature_num_batches_tracked",
    )
    allowed_missing = set(dropped_classification)
    allowed_missing.update(
        key for key in result.missing_keys if key.endswith(feature_stat_suffixes)
    )
    unexpected_missing = sorted(set(result.missing_keys) - allowed_missing)
    if unexpected_missing:
        raise RuntimeError(f"Checkpoint is missing unexpected parameters: {unexpected_missing}")
    if result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint contains unexpected parameters: {result.unexpected_keys}"
        )
    if dropped_classification:
        warnings.warn(
            "Dropped an incompatible classification head. Evaluation uses only "
            "the backbone and representation head.",
            RuntimeWarning,
        )

    checkpoint_stat_suffixes = {
        suffix
        for suffix in feature_stat_suffixes
        if any(key.endswith(suffix) for key in state_dict)
    }
    if len(checkpoint_stat_suffixes) == len(feature_stat_suffixes):
        model._use_frozen_feature_stats_in_eval = True
    else:
        model._use_frozen_feature_stats_in_eval = False
        warnings.warn(
            "Legacy checkpoint has incomplete feature running statistics; "
            "evaluation will use the model's legacy per-batch behavior.",
            RuntimeWarning,
        )
    model.eval()


@torch.inference_mode()
def encode_single_view(
    model: torch.nn.Module,
    x: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    cls = model(x, padding_mask=padding_mask)
    if not hasattr(model, "representation_head"):
        raise AttributeError("Loaded model has no representation_head.")
    return model.representation_head(cls)


def iter_limited(
    loader: DataLoader,
    *,
    max_steps: Optional[int],
    description: str,
) -> Iterator[Mapping[str, torch.Tensor]]:
    if max_steps is not None and max_steps <= 0:
        raise ValueError("--eval-steps must be positive when provided.")
    progress = tqdm(loader, total=max_steps, desc=description)
    for step, batch in enumerate(progress):
        if max_steps is not None and step >= max_steps:
            break
        yield batch


def count_test_events(
    loader: DataLoader,
    max_steps: Optional[int],
) -> int:
    total = 0
    for batch in iter_limited(loader, max_steps=max_steps, description="Count real test"):
        total += int(batch["x_particles"].shape[0])
    return total


@torch.no_grad()
def collect_reference_pool(model, loader, *, device, precision, label_axis,
                           labels, requested_per_class):
    chunks = []
    counts = {label: 0 for label in labels}
    for batch in tqdm(loader, desc="Encode balanced train reference"):
        with autocast_context(device, precision):
            z = encode_single_view(model, batch["x_particles"].to(device),
                                   batch["padding_mask"].to(device))
        chunks.append(z.detach().float().cpu())
        for index in batch["y"].argmax(dim=1).tolist():
            counts[label_axis[index]] += 1
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"Unbalanced reference pool: {counts}")
    actual = next(iter(counts.values()))
    if actual < requested_per_class:
        warnings.warn(f"Reference request {requested_per_class} jets/class exceeds smallest "
                      f"available class m={actual}; stopped without refill. Counts: {counts}")
    if not chunks:
        raise RuntimeError("No complete class cycle available for reference pool.")
    reference = torch.cat(chunks)
    if not torch.isfinite(reference).all():
        raise ValueError("Non-finite reference representations.")
    print(f"Reference pool counts: {counts}; total={len(reference)}")
    return reference, counts


@torch.no_grad()
def knn_scores(query, reference, *, k, reduction, reference_batch_size):
    """Exact kNN; keep the reference on CPU and transfer bounded blocks."""
    if not 1 <= k <= len(reference):
        raise ValueError(f"k={k} requires at least {k} reference jets, got {len(reference)}")
    if reduction not in ("mean", "kth") or reference_batch_size < 1:
        raise ValueError("Invalid kNN reduction or reference batch size.")
    query = query.float()
    if not torch.isfinite(query).all():
        raise ValueError("Non-finite query representations.")
    best = query.new_full((len(query), k), float("inf"))
    for start in range(0, len(reference), reference_batch_size):
        block = reference[start:start + reference_batch_size].to(query.device)
        distances = torch.cdist(query, block)
        candidates = distances.topk(min(k, len(block)), largest=False).values
        best = torch.cat((best, candidates), dim=1).topk(k, largest=False).values
    scores = best.mean(dim=1) if reduction == "mean" else best.max(dim=1).values
    return scores.cpu().numpy()


def validate_world_size():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = max(world_size, torch.distributed.get_world_size())
    if world_size != 1:
        raise RuntimeError(f"Real-data kNN evaluation requires world size == 1; got {world_size}.")


def _fraction_tag(fraction: float) -> str:
    percent = 100.0 * fraction
    if math.isclose(percent, round(percent), rel_tol=0.0, abs_tol=1e-10):
        return f"{int(round(percent))}pct"
    return f"{percent:.3f}".rstrip("0").rstrip(".").replace(".", "p") + "pct"


def quality_filter(events: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
    """Reject any event whose leading particle carries strictly more than 80% of jet pT."""
    pt_index = list(feature_names).index("part_pt")
    valid = np.any(events != 0, axis=-1)
    pt = np.where(valid, events[..., pt_index], -np.inf)
    return np.max(pt, axis=1) > 0.8


def save_quality_partition(events, ordered, feature_names, events_path, scores_path, metadata=None):
    """Partition score-ordered candidates, saving both groups in that same order."""
    groups = {"retained": [], "vetoed": []}
    for start in range(0, len(ordered), 256):
        items = ordered[start:start + 256]
        slots = np.asarray([slot for _, slot in items], dtype=np.int64)
        veto = quality_filter(events[slots], feature_names)
        for item, reject in zip(items, veto):
            groups["vetoed" if reject else "retained"].append(item)
    prefix = events_path.stem.removesuffix("_events")
    veto_events_path = events_path.with_name(prefix + "_quality_filtered_events.npy")
    veto_scores_path = events_path.with_name(prefix + "_quality_filtered_event_scores.npy")
    for group, event_file, score_file in (
        ("retained", events_path, scores_path),
        ("vetoed", veto_events_path, veto_scores_path),
    ):
        items = groups[group]
        output = np.lib.format.open_memmap(event_file, mode="w+", dtype=np.float32,
                                          shape=(len(items), *events.shape[1:]))
        scores = np.lib.format.open_memmap(score_file, mode="w+", dtype=np.float32,
                                          shape=(len(items),))
        metadata_output = None
        if metadata is not None:
            metadata_output = np.lib.format.open_memmap(
                event_file.with_name(event_file.stem + "_metadata.npy"), mode="w+",
                dtype=np.float32, shape=(len(items), metadata.shape[1]))
        for start in range(0, len(items), 256):
            chunk = items[start:start + 256]
            slots = np.asarray([slot for _, slot in chunk], dtype=np.int64)
            output[start:start + len(chunk)] = events[slots]
            if metadata_output is not None:
                metadata_output[start:start + len(chunk)] = metadata[slots]
            scores[start:start + len(chunk)] = [score for score, _ in chunk]
        if metadata_output is not None:
            metadata_output.flush()
            del metadata_output
        output.flush()
        scores.flush()
        del output, scores
    return dict(retained_count=len(groups["retained"]),
                quality_filter_count=len(groups["vetoed"]),
                vetoed_events_path=veto_events_path, vetoed_event_scores_path=veto_scores_path)


def dump_visualized_events(events_path, indices, output_path, feature_names):
    """Export the plotted tokens; metadata shares every selection/permutation with events."""
    events_path = Path(events_path)
    metadata_path = events_path.with_name(events_path.stem + "_metadata.npy")
    if not metadata_path.exists():
        warnings.warn(f"Missing {metadata_path}; cannot export jet metadata from legacy arrays.")
        return None
    events = np.load(events_path, mmap_mode="r")
    metadata = np.load(metadata_path, mmap_mode="r")
    if metadata.shape != (len(events), 3 + events.shape[1] * 7):
        raise ValueError("Event metadata shape disagrees with saved events")
    pt_index = list(feature_names).index("part_pt")
    records = []
    for index in indices:
        row = metadata[index]
        valid = np.any(events[index] != 0, axis=-1)
        particles = row[3:].reshape(-1, 7)[valid]
        pt_frac = events[index, :, pt_index][valid]
        record = {key: float(value) for key, value in zip(
            ("jet_pt", "jet_eta", "jet_phi"), row[:3])}
        for column, key in enumerate(("pt", "deta", "dphi", "charge", "pdgId", "dzsig", "dxysig")):
            values = particles[:, column]
            record[key] = values.astype(np.int64).tolist() if key == "pdgId" else values.tolist()
        record["event_index"] = int(index)
        record["pt_frac"] = pt_frac.tolist()
        records.append(record)
    output_path.write_text(json.dumps(records, indent=2, allow_nan=False))
    return output_path


def plot_quality_groups(selection, *, tail, output_dir, feature_names,
                       feature_payload, num_events, seed):
    """Use the same random sampling and score-order plotting for both groups."""
    plots = {}
    for group, count, events_path, scores_path in (
        ("retained", selection[f"{tail}_count"], selection["events_path"], selection["event_scores_path"]),
        ("vetoed", selection["quality_filter_count"],
         selection["vetoed_events_path"], selection["vetoed_event_scores_path"]),
    ):
        events_path = Path(events_path)
        features_path = events_path.with_name(events_path.stem + "_features.json")
        features_path.write_text(json.dumps(feature_payload, indent=2))
        suffix = "" if group == "retained" else "_quality_filtered"
        path = output_dir / f"real_test_{tail}{suffix}_events_random_visualization.png"
        indices = []
        if count:
            direction = "highest" if tail == "top" else "lowest"
            status = "quality-filtered" if group == "vetoed" else "retained"
            indices = plot_random_jet_events(
                events_path=events_path, feature_names=feature_names, output_path=path,
                num_events=num_events, seed=seed, scores_path=Path(scores_path),
                title=f"Random {status} sample from {direction}-scoring real-data jets",
            )
        else:
            warnings.warn(f"No {tail} {group} events; skipping corresponding jet visualization.")
            path = None
        json_path = output_dir / f"real_test_{tail}{suffix}_events_random_visualization.json"
        json_path = dump_visualized_events(events_path, indices, json_path, feature_names)
        plots[group] = dict(path=path, indices=indices, features_path=features_path, json_path=json_path)
    return plots


class BottomEventSelection:
    """Bounded max-heap for lowest scores; apply veto only after score selection."""

    def __init__(self, output_dir, fraction, total_events, max_num_particles, num_features):
        self.count = max(1, int(math.ceil(total_events * fraction)))
        self.total_events = total_events
        self.shape = (max_num_particles, num_features)
        tag = _fraction_tag(fraction)
        self.temp_path = output_dir / f".real_test_bottom_{tag}_events_unsorted.npy"
        self.events_path = output_dir / f"real_test_bottom_{tag}_events.npy"
        self.scores_path = output_dir / f"real_test_bottom_{tag}_event_scores.npy"
        self.events = np.lib.format.open_memmap(self.temp_path, mode="w+", dtype=np.float32,
                                               shape=(self.count, *self.shape))
        self.metadata = None
        self.heap = []

    def update(self, events, scores, metadata=None):
        if metadata is not None and self.metadata is None:
            self.metadata = np.lib.format.open_memmap(
                str(self.temp_path) + ".metadata.npy", mode="w+", dtype=np.float32,
                shape=(self.count, metadata.shape[1]))
        for i, score in enumerate(scores):
            priority = -float(score)
            if len(self.heap) < self.count:
                slot = len(self.heap)
                self.events[slot] = events[i]
                if metadata is not None:
                    self.metadata[slot] = metadata[i]
                heapq.heappush(self.heap, (priority, slot))
            elif priority > self.heap[0][0]:
                slot = self.heap[0][1]
                heapq.heapreplace(self.heap, (priority, slot))
                self.events[slot] = events[i]
                if metadata is not None:
                    self.metadata[slot] = metadata[i]

    def finish(self, feature_names):
        if len(self.heap) != self.count:
            raise RuntimeError("Bottom-event heap did not reach the expected count.")
        # Descending negative score is ascending actual score.
        ordered = sorted(self.heap, key=lambda item: item[0], reverse=True)
        cutoff = -ordered[-1][0]
        partition = save_quality_partition(
            self.events, [(-priority, slot) for priority, slot in ordered],
            feature_names, self.events_path, self.scores_path, self.metadata,
        )
        count = partition.pop("retained_count")
        del self.events, self.metadata
        Path(str(self.temp_path) + ".metadata.npy").unlink(missing_ok=True)
        self.temp_path.unlink(missing_ok=True)
        print(f"Bottom quality filter: removed {self.count - count}/{self.count}; retained {count}.")
        return dict(events_path=self.events_path, event_scores_path=self.scores_path,
                    bottom_count_before_quality_filter=self.count,
                    bottom_count=count, **partition,
                    bottom_fraction_realized=count / self.total_events, cutoff=float(cutoff))


def score_and_select_top_events(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    total_events: int,
    reference: torch.Tensor,
    k: int,
    reduction: str,
    reference_batch_size: int,
    top_fraction: float,
    device: torch.device,
    precision: str,
    max_steps: Optional[int],
    output_dir: Path,
    max_num_particles: int,
    num_features: int,
    feature_names: Sequence[str],
    heap_progress_every: int,
    bottom_fraction: Optional[float] = None,
) -> Dict[str, object]:
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("--top-fraction must lie strictly between 0 and 1.")
    bottom_fraction = top_fraction if bottom_fraction is None else bottom_fraction
    if not 0.0 < bottom_fraction < 1.0:
        raise ValueError("--bottom-fraction must lie strictly between 0 and 1.")
    for name in ("part_pt",):
        if name not in feature_names:
            raise ValueError(f"Quality filter requires feature {name}")
    top_count = max(1, int(math.ceil(total_events * top_fraction)))
    tag = _fraction_tag(top_fraction)

    all_scores_path = output_dir / "real_test_knn_scores.npy"
    temp_events_path = output_dir / f".real_test_top_{tag}_events_unsorted.npy"
    final_events_path = output_dir / f"real_test_top_{tag}_events.npy"
    final_scores_path = output_dir / f"real_test_top_{tag}_event_scores.npy"

    all_scores = np.lib.format.open_memmap(
        all_scores_path, mode="w+", dtype=np.float32, shape=(total_events,)
    )
    temp_events = np.lib.format.open_memmap(
        temp_events_path,
        mode="w+",
        dtype=np.float32,
        shape=(top_count, max_num_particles, num_features),
    )

    temp_metadata = None
    metadata_path = Path(str(temp_events_path) + ".metadata.npy")
    bottom = BottomEventSelection(output_dir, bottom_fraction, total_events,
                                  max_num_particles, num_features)
    heap: List[Tuple[float, int]] = []
    offset = 0
    score_sum = 0.0
    score_sumsq = 0.0
    score_min = float("inf")
    score_max = float("-inf")
    next_report = max(1, int(heap_progress_every))

    for batch in iter_limited(loader, max_steps=max_steps, description="Score complete real test"):
        x_cpu = batch["x_particles"]
        batch_metadata = batch.get("event_metadata")
        if batch_metadata is not None:
            batch_metadata = batch_metadata.numpy()
            if temp_metadata is None:
                temp_metadata = np.lib.format.open_memmap(
                    metadata_path, mode="w+", dtype=np.float32,
                    shape=(top_count, batch_metadata.shape[1]))
        batch_size = int(x_cpu.shape[0])
        if offset + batch_size > total_events:
            raise RuntimeError("Score pass produced more events than the count pass.")
        x = x_cpu.to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            z = encode_single_view(model, x, mask)
        scores = knn_scores(z, reference, k=k, reduction=reduction,
                            reference_batch_size=reference_batch_size)
        scores32 = scores.astype(np.float32, copy=False)
        all_scores[offset : offset + batch_size] = scores32
        bottom.update(x_cpu.numpy(), scores, batch_metadata)

        score_sum += float(scores.sum(dtype=np.float64))
        score_sumsq += float(np.square(scores, dtype=np.float64).sum(dtype=np.float64))
        score_min = min(score_min, float(scores.min()))
        score_max = max(score_max, float(scores.max()))

        local_start = 0
        fill_count = min(top_count - len(heap), batch_size)
        for local_index in range(fill_count):
            slot = len(heap)
            score = float(scores[local_index])
            temp_events[slot] = x_cpu[local_index].numpy()
            if batch_metadata is not None:
                temp_metadata[slot] = batch_metadata[local_index]
            heapq.heappush(heap, (score, slot))
        local_start += fill_count

        if local_start < batch_size:
            threshold = heap[0][0]
            candidates = np.flatnonzero(scores[local_start:] > threshold) + local_start
            for local_index in candidates.tolist():
                score = float(scores[local_index])
                if score <= heap[0][0]:
                    continue
                slot = heap[0][1]
                heapq.heapreplace(heap, (score, slot))
                temp_events[slot] = x_cpu[local_index].numpy()
                if batch_metadata is not None:
                    temp_metadata[slot] = batch_metadata[local_index]

        offset += batch_size
        if offset >= next_report:
            current_cut = heap[0][0] if len(heap) == top_count else float("nan")
            tqdm.write(
                f"Processed {offset:,}/{total_events:,} events; "
                f"current top cut={current_cut:.6g}"
            )
            next_report += max(1, int(heap_progress_every))

    if offset != total_events:
        raise RuntimeError(
            f"Score pass event count mismatch: expected {total_events}, got {offset}."
        )
    if len(heap) != top_count:
        raise RuntimeError(f"Top-event heap has {len(heap)} entries; expected {top_count}.")

    all_scores.flush()
    temp_events.flush()

    heap.sort(key=lambda item: item[0], reverse=True)
    # Keep the original score cut. Veto only selected events; never backfill
    # with lower-scoring events to restore the requested top fraction.
    cutoff = float(heap[-1][0])
    pre_veto_count = top_count
    partition = save_quality_partition(temp_events, heap, feature_names,
                                      final_events_path, final_scores_path, temp_metadata)
    top_count = partition.pop("retained_count")
    print(f"Top quality filter: removed {pre_veto_count - top_count}/{pre_veto_count}; "
          f"retained {top_count} events.")
    del temp_events, all_scores, temp_metadata
    metadata_path.unlink(missing_ok=True)
    temp_events_path.unlink(missing_ok=True)

    bottom_selection = bottom.finish(feature_names)
    mean_score = score_sum / total_events
    variance = max(0.0, score_sumsq / total_events - mean_score * mean_score)
    return {
        "bottom": bottom_selection,
        "all_scores_path": all_scores_path,
        "events_path": final_events_path,
        "event_scores_path": final_scores_path,
        "top_count_before_quality_filter": pre_veto_count,
        **partition,
        "top_count": int(top_count),
        "top_fraction_realized": float(top_count / total_events),
        "cutoff": cutoff,
        "score_stats": {
            "count": int(total_events),
            "mean": float(mean_score),
            "std": float(math.sqrt(variance)),
            "min": float(score_min),
            "max": float(score_max),
        },
    }


def plot_score_distribution(
    *,
    scores_path: Path,
    cutoff: float,
    output_path: Path,
    seed: int,
    histogram_sample_size: int,
    tail: str = "top",
) -> Dict[str, float]:
    scores = np.load(scores_path, mmap_mode="r")
    total = len(scores)
    sample_size = min(total, max(1, int(histogram_sample_size)))
    rng = np.random.default_rng(seed)
    sample_indices = (
        np.arange(total, dtype=np.int64)
        if sample_size == total
        else rng.choice(total, size=sample_size, replace=False)
    )
    sample = np.asarray(scores[sample_indices], dtype=np.float64)
    low, high = np.quantile(sample, [0.001, 0.995])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(sample))
        high = float(np.max(sample))
    high = max(high, cutoff)
    if tail == "bottom":
        low = min(low, cutoff)
    if high <= low:
        high = low + 1.0
    bins = np.linspace(low, high, 121)
    counts = np.zeros(len(bins) - 1, dtype=np.int64)
    chunk = 1_000_000
    for start in range(0, total, chunk):
        values = np.asarray(scores[start : start + chunk], dtype=np.float64)
        values = np.clip(values, low, high)
        counts += np.histogram(values, bins=bins)[0]
    widths = np.diff(bins)
    density = counts / (counts.sum() * widths)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.stairs(density, bins, color="#3F5368", linewidth=1.8)
    ax.fill_between(
        bins[:-1], density, step="post", color="#8195A8", alpha=0.18
    )
    ax.axvline(
        cutoff,
        color="#B7657B",
        linewidth=1.8,
        linestyle="--",
        label=f"{tail}-event cut = {cutoff:.3g}",
    )
    ax.axvspan(*( (low, cutoff) if tail == "bottom" else (cutoff, high) ),
               color="#B7657B", alpha=0.08)
    ax.set_xlabel("Euclidean kNN score")
    ax.set_ylabel("Density")
    ax.set_title("Held-out CMS real-data anomaly-score distribution")
    ax.grid(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return {
        "histogram_low": float(low),
        "histogram_high": float(high),
        "sample_q05": float(np.quantile(sample, 0.05)),
        "sample_median": float(np.median(sample)),
        "sample_q95": float(np.quantile(sample, 0.95)),
    }


def load_feature_names(path: Path) -> List[str]:
    with path.open() as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        features = [str(x) for x in payload]
    elif isinstance(payload, Mapping) and "particle_features" in payload:
        features = [str(x) for x in payload["particle_features"]]
    else:
        raise ValueError(
            f"Feature-list JSON must be a list or contain particle_features: {path}"
        )
    if not features or len(set(features)) != len(features):
        raise ValueError(f"Invalid feature list in {path}: {features}")
    return features


def infer_feature_list_path(events_path: Path) -> Path:
    candidate = events_path.with_name(events_path.stem + "_features.json")
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(
        "Could not infer the separate feature list. Pass --feature-list; expected "
        f"candidate was {candidate}."
    )


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    return value


def run_plot_only(args: argparse.Namespace) -> None:
    events_path = args.plot_events_npy.expanduser().resolve()
    if not events_path.is_file():
        raise FileNotFoundError(events_path)
    feature_path = (
        args.feature_list.expanduser().resolve()
        if args.feature_list is not None
        else infer_feature_list_path(events_path)
    )
    score_path = (
        args.event_scores.expanduser().resolve()
        if args.event_scores is not None
        else None
    )
    output_path = (
        args.plot_output.expanduser().resolve()
        if args.plot_output is not None
        else events_path.with_name(events_path.stem + "_random_visualization.png")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed if args.seed is not None else 404)
    seed_everything(seed)
    feature_names = load_feature_names(feature_path)
    selected = plot_random_jet_events(
        events_path=events_path,
        feature_names=feature_names,
        output_path=output_path,
        num_events=args.num_visualize,
        seed=seed,
        scores_path=score_path,
    )
    dump_visualized_events(events_path, selected, output_path.with_suffix(".json"), feature_names)
    print(f"Saved visualization to {output_path}")
    print(f"Selected event-array indices: {selected}")


def main() -> None:
    args = parse_args()
    validate_world_size()
    if args.plot_events_npy is not None:
        run_plot_only(args)
        return
    if args.run_dir is None:
        raise ValueError("run_dir is required unless --plot-events-npy is used.")

    run_dir = args.run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    with summary_path.open() as handle:
        summary = json.load(handle)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "real_data_evaluation"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else run_dir / "best_model.pth"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    device = resolve_device(args.device)
    seed = int(
        args.seed
        if args.seed is not None
        else summary.get("base_seed", summary.get("seed", 42))
    )
    seed_everything(seed)
    backend = RealTestBackend(
        summary=summary,
        run_dir=run_dir,
        dataset_root_override=args.dataset_root,
        manifest_override=args.cms_split_manifest,
        max_num_particles_override=args.max_num_particles,
    )

    batch_size = int(
        args.batch_size
        if args.batch_size is not None
        else summary.get("per_rank_batch_size", summary.get("batch_size", 128))
    )
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    requested_workers = int(
        args.num_workers
        if args.num_workers is not None
        else summary.get("num_workers", 0)
    )
    if requested_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    num_workers = 1
    warnings.warn(f"Overriding evaluation num_workers={requested_workers} to 1; "
                  "balanced finite reference sampling requires one worker.")
    for name in ("reference_steps", "knn_k", "knn_reference_batch_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.eval_steps is not None and args.eval_steps < 1:
        raise ValueError("--eval-steps must be positive.")
    top_fraction = float(args.top_fraction)
    if not 0 < top_fraction < 1:
        raise ValueError("--top-fraction must be between 0 and 1.")
    bottom_fraction = top_fraction if args.bottom_fraction is None else float(args.bottom_fraction)
    if not 0 < bottom_fraction < 1:
        raise ValueError("--bottom-fraction must be between 0 and 1.")
    precision = str(summary.get("precision", "fp32"))
    loader_seed = seed + 404

    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset root: {backend.dataset_root}")
    print(f"Manifest: {backend.manifest_path}")
    print(f"label_Real test shards: {len(backend.test_files)}")
    print(f"Device: {device}; batch size: {batch_size}; workers: {num_workers}")
    print(f"kNN k={args.knn_k}, reduction={args.knn_reduction}; top fraction={top_fraction}")
    if args.eval_steps is None:
        print("Evaluation scope: complete finite label_Real test split")
    else:
        print(f"Evaluation scope: first {args.eval_steps} finite batches (debug cap)")

    def make_loader() -> DataLoader:
        return backend.make_loader(
            seed=loader_seed,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

    total_events = count_test_events(make_loader(), args.eval_steps)
    if total_events < 1:
        raise RuntimeError(f"Only {total_events} held-out test events were found.")
    print(f"Finite held-out test events: {total_events:,}")

    model = build_model(summary, backend, device)
    load_model_weights(model, checkpoint_path, device)

    model.eval().requires_grad_(False)
    requested_per_class = args.reference_steps * batch_size // len(backend.reference_labels)
    if requested_per_class < 1:
        raise ValueError("Requested reference pool must contain at least one jet per class.")
    reference, reference_counts = collect_reference_pool(
        model, backend.make_reference_loader(
            seed=seed + 505, batch_size=batch_size,
            max_events=requested_per_class * len(backend.reference_labels),
            pin_memory=device.type == "cuda"),
        device=device, precision=precision, label_axis=backend.label_axis,
        labels=backend.reference_labels, requested_per_class=requested_per_class,
    )
    if len(reference) < args.knn_k:
        raise ValueError(f"Reference pool has {len(reference)} jets, fewer than k={args.knn_k}.")
    reference_path = output_dir / "train_reference_latents.npy"
    np.save(reference_path, reference.numpy())

    selection = score_and_select_top_events(
        model=model,
        loader=make_loader(),
        total_events=total_events,
        reference=reference, k=args.knn_k, reduction=args.knn_reduction,
        reference_batch_size=args.knn_reference_batch_size,
        top_fraction=top_fraction,
        device=device,
        precision=precision,
        max_steps=args.eval_steps,
        output_dir=output_dir,
        max_num_particles=backend.max_num_particles,
        num_features=len(backend.feature_names),
        feature_names=backend.feature_names,
        heap_progress_every=args.heap_progress_every,
        bottom_fraction=bottom_fraction,
    )

    events_path = Path(selection["events_path"])
    feature_list_path = events_path.with_name(events_path.stem + "_features.json")
    with feature_list_path.open("w") as handle:
        json.dump(
            {
                "particle_features": backend.feature_names,
                "shape_semantics": ["events", "particles", "features"],
                "source_run": str(run_dir),
                "source_summary": str(summary_path),
            },
            handle,
            indent=2,
        )

    distribution_path = output_dir / "real_test_score_distribution.png"
    histogram_stats = plot_score_distribution(
        scores_path=Path(selection["all_scores_path"]),
        cutoff=float(selection["cutoff"]),
        output_path=distribution_path,
        seed=seed + 606,
        histogram_sample_size=args.histogram_sample_size,
    )
    feature_payload = json.loads(feature_list_path.read_text())
    top_plots = plot_quality_groups(
        selection, tail="top", output_dir=output_dir, feature_names=backend.feature_names,
        feature_payload=feature_payload, num_events=args.num_visualize, seed=seed + 707,
    )
    visualization_path = top_plots["retained"]["path"]
    selected_visual_indices = top_plots["retained"]["indices"]

    results = {
        "run_dir": run_dir,
        "checkpoint": checkpoint_path,
        "dataset": "cms",
        "label": REAL_LABEL,
        "dataset_root": backend.dataset_root,
        "cms_split_manifest": backend.manifest_path,
        "cms_split_manifest_sha256": backend.manifest_sha256,
        "test_root_shards": len(backend.test_files),
        "particle_features": backend.feature_names,
        "batch_standardized_particle_features": backend.standardized_feature_names,
        "max_num_particles": backend.max_num_particles,
        "device": str(device),
        "precision": precision,
        "seed": seed,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "eval_steps": args.eval_steps,
        "total_test_events": total_events,
        "reference_steps": args.reference_steps,
        "reference_requested_per_class": requested_per_class,
        "reference_counts": reference_counts,
        "reference_labels": backend.reference_labels,
        "knn_k": args.knn_k,
        "knn_reduction": args.knn_reduction,
        "top_fraction_requested": top_fraction,
        "score_stats": selection["score_stats"],
        "histogram_stats": histogram_stats,
        "top_count_before_quality_filter": selection["top_count_before_quality_filter"],
        "quality_filter_count": selection["quality_filter_count"],
        "quality_filter_pt_fraction": 0.8,
        "top_count": selection["top_count"],
        "top_fraction_realized": selection["top_fraction_realized"],
        "top_cutoff": selection["cutoff"],
        "random_visualized_event_array_indices": selected_visual_indices,
        "vetoed_random_visualized_event_array_indices": top_plots["vetoed"]["indices"],
        "files": {
            "reference_latents": reference_path,
            "all_scores": selection["all_scores_path"],
            "top_events": selection["events_path"],
            "top_event_scores": selection["event_scores_path"],
            "top_event_features": feature_list_path,
            "score_distribution": distribution_path,
            "random_visualization": visualization_path,
            "visualized_jets_json": top_plots["retained"]["json_path"],
            "quality_filtered_visualized_jets_json": top_plots["vetoed"]["json_path"],
            "quality_filtered_events": selection["vetoed_events_path"],
            "quality_filtered_event_scores": selection["vetoed_event_scores_path"],
            "quality_filtered_event_features": top_plots["vetoed"]["features_path"],
            "quality_filtered_random_visualization": top_plots["vetoed"]["path"],
        },
    }
    results_path = output_dir / "real_test_evaluation.json"
    with results_path.open("w") as handle:
        json.dump(safe_json(results), handle, indent=2)

    bottom_selection = selection["bottom"]
    bottom_events_path = Path(bottom_selection["events_path"])
    bottom_features_path = bottom_events_path.with_name(bottom_events_path.stem + "_features.json")
    with feature_list_path.open() as handle:
        feature_payload = json.load(handle)
    bottom_features_path.write_text(json.dumps(feature_payload, indent=2))
    bottom_distribution_path = output_dir / "real_test_bottom_score_distribution.png"
    bottom_histogram = plot_score_distribution(
        scores_path=Path(selection["all_scores_path"]), cutoff=bottom_selection["cutoff"],
        output_path=bottom_distribution_path, seed=seed + 606,
        histogram_sample_size=args.histogram_sample_size, tail="bottom",
    )
    bottom_plots = plot_quality_groups(
        bottom_selection, tail="bottom", output_dir=output_dir, feature_names=backend.feature_names,
        feature_payload=feature_payload, num_events=args.num_visualize, seed=seed + 707,
    )
    bottom_visualization_path = bottom_plots["retained"]["path"]
    bottom_visual_indices = bottom_plots["retained"]["indices"]
    # Mirror the top manifest, with bottom-specific selection metadata/files.
    bottom_results = {key: value for key, value in results.items()
                      if not key.startswith("top_") and key not in (
                          "files", "histogram_stats", "quality_filter_count",
                          "random_visualized_event_array_indices")}
    bottom_results.update(
        bottom_fraction_requested=bottom_fraction,
        bottom_count_before_quality_filter=bottom_selection["bottom_count_before_quality_filter"],
        quality_filter_count=bottom_selection["quality_filter_count"],
        bottom_count=bottom_selection["bottom_count"],
        bottom_fraction_realized=bottom_selection["bottom_fraction_realized"],
        bottom_cutoff=bottom_selection["cutoff"], histogram_stats=bottom_histogram,
        random_visualized_event_array_indices=bottom_visual_indices,
        vetoed_random_visualized_event_array_indices=bottom_plots["vetoed"]["indices"],
        files=dict(reference_latents=reference_path, all_scores=selection["all_scores_path"],
                   bottom_events=bottom_events_path,
                   bottom_event_scores=bottom_selection["event_scores_path"],
                   bottom_event_features=bottom_features_path,
                   score_distribution=bottom_distribution_path,
                   random_visualization=bottom_visualization_path,
                   visualized_jets_json=bottom_plots["retained"]["json_path"],
                   quality_filtered_visualized_jets_json=bottom_plots["vetoed"]["json_path"],
                   quality_filtered_events=bottom_selection["vetoed_events_path"],
                   quality_filtered_event_scores=bottom_selection["vetoed_event_scores_path"],
                   quality_filtered_event_features=bottom_plots["vetoed"]["features_path"],
                   quality_filtered_random_visualization=bottom_plots["vetoed"]["path"]),
    )
    bottom_manifest_path = output_dir / "real_test_bottom_evaluation.json"
    bottom_manifest_path.write_text(json.dumps(safe_json(bottom_results), indent=2))
    print(f"Bottom-event cut: {bottom_selection['cutoff']:.6g}")
    print(f"Bottom events: {bottom_selection['bottom_count']:,} -> {bottom_events_path}")

    print(f"Saved evaluation outputs to {output_dir}")
    print(f"Top-event cut: {float(selection['cutoff']):.6g}")
    print(f"Top events: {selection['top_count']:,} -> {selection['events_path']}")


if __name__ == "__main__":
    main()
