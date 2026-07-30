#!/usr/bin/env python3
"""Evaluate a real-CMS fine-tuned LeJEPA model on the held-out label_Real test split.

The script reads only ``label_Real`` from the CMS test split recorded in the
fine-tuning run's ``cms_split_manifest.json``. It never reads train or val.
Evaluation is finite: every selected test shard is traversed once, without the
infinite refill used during training.

Evaluation workflow:
    1. Count the finite held-out test events (or an optional debug batch cap).
    2. Select an exact random ``fit_fraction`` of those events and fit one global
       Gaussian in representation space.
    3. Score the complete held-out test sample with global squared Mahalanobis
       distance.
    4. Save the highest-scoring ``top_fraction`` events, their scores, the full
       score array, the Gaussian parameters, a score distribution, and a random
       visualization of selected anomalous jets.

A plot-only mode can load a saved ``(events, particles, features)`` NumPy array
and its separately saved feature-name JSON without loading a model.

Examples
--------
Full real-data evaluation (defaults to best_model.pth):
    python -u scripts/evaluate_label_real_anomalies.py \
        plots/real-data-finetune-run

Use last_model.pth and draw 16 selected jets:
    python -u scripts/evaluate_label_real_anomalies.py \
        plots/real-data-finetune-run \
        --checkpoint plots/real-data-finetune-run/last_model.pth \
        --num-visualize 16

Plot-only mode:
    python -u scripts/evaluate_label_real_anomalies.py \
        --plot-events-npy real_test_top_5pct_events.npy \
        --feature-list real_test_top_5pct_events_features.json \
        --event-scores real_test_top_5pct_event_scores.npy \
        --num-visualize 12
"""

from __future__ import annotations

import argparse
import heapq
import inspect
import json
import math
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

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
IDENTITY_FEATURES = (
    "part_isChargedHadron",
    "part_isNeutralHadron",
    "part_isPhoton",
    "part_isElectron",
    "part_isMuon",
)
IDENTITY_MARKERS = {
    "part_isChargedHadron": "o",
    "part_isNeutralHadron": "h",
    "part_isPhoton": "^",
    "part_isElectron": "s",
    "part_isMuon": "D",
    "unidentified": "X",
}
IDENTITY_LABELS = {
    "part_isChargedHadron": "charged hadron",
    "part_isNeutralHadron": "neutral hadron",
    "part_isPhoton": "photon",
    "part_isElectron": "electron",
    "part_isMuon": "muon",
    "unidentified": "other",
}
CHARGE_COLORS = {
    -1: "#4F78C8",
    0: "#8B8F97",
    1: "#D65A5A",
}


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
            "Fit a global Gaussian on half of held-out label_Real test events, "
            "score the entire held-out test split, and save the top anomalies."
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
        help="Defaults to summary.json num_workers and is capped by test shards.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--mahalanobis-cov-eps", type=float, default=None)
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
        if backgrounds != [REAL_LABEL]:
            raise ValueError(
                "The fine-tuning run must have background_labels=['label_Real']; "
                f"found {backgrounds}."
            )
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
        if set(test_labels) != {REAL_LABEL}:
            raise ValueError(
                "The test split must contain only label_Real for this evaluator; "
                f"found {sorted(test_labels)}."
            )
        self.test_files = list(test_labels[REAL_LABEL])
        if not self.test_files:
            raise ValueError("The label_Real test split contains no ROOT shards.")

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
            shuffle_files=False,
            shuffle_active_shards=self.shuffle_active_shards,
            infinite=False,
            seed=seed,
            rank=0,
            world_size=1,
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
            kwargs["prefetch_factor"] = int(self.summary.get("prefetch_factor", 2))
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


def fit_global_gaussian(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    total_events: int,
    fit_fraction: float,
    seed: int,
    device: torch.device,
    precision: str,
    cov_eps: float,
    max_steps: Optional[int],
    representation_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("--fit-fraction must lie strictly between 0 and 1.")
    fit_count = int(round(total_events * fit_fraction))
    fit_count = min(max(fit_count, 2), total_events - 1)
    rng = np.random.default_rng(seed)

    remaining_total = total_events
    remaining_fit = fit_count
    sum_z = np.zeros(representation_dim, dtype=np.float64)
    sum_zz = np.zeros((representation_dim, representation_dim), dtype=np.float64)
    selected = 0

    for batch in iter_limited(loader, max_steps=max_steps, description="Fit global Gaussian"):
        batch_size = int(batch["x_particles"].shape[0])
        if batch_size > remaining_total:
            raise RuntimeError("Fit pass produced more events than the count pass.")
        choose_count = int(
            rng.hypergeometric(
                ngood=remaining_fit,
                nbad=remaining_total - remaining_fit,
                nsample=batch_size,
            )
        )
        remaining_total -= batch_size
        remaining_fit -= choose_count
        if choose_count == 0:
            continue

        local_indices = np.sort(
            rng.choice(batch_size, size=choose_count, replace=False)
        )
        index_tensor = torch.as_tensor(local_indices, dtype=torch.long)
        x = batch["x_particles"].index_select(0, index_tensor).to(
            device, non_blocking=True
        )
        mask = batch["padding_mask"].index_select(0, index_tensor).to(
            device, non_blocking=True
        )
        with autocast_context(device, precision):
            z = encode_single_view(model, x, mask)
        z64 = z.detach().float().cpu().numpy().astype(np.float64, copy=False)
        if z64.ndim != 2 or z64.shape[1] != representation_dim:
            raise RuntimeError(
                f"Unexpected representation shape {z64.shape}; expected (*, {representation_dim})."
            )
        sum_z += z64.sum(axis=0)
        sum_zz += z64.T @ z64
        selected += len(z64)

    if remaining_total != 0 or remaining_fit != 0 or selected != fit_count:
        raise RuntimeError(
            "Exact fit-subset selection failed: "
            f"remaining_total={remaining_total}, remaining_fit={remaining_fit}, "
            f"selected={selected}, target={fit_count}."
        )

    mean = sum_z / selected
    covariance = (sum_zz - selected * np.outer(mean, mean)) / (selected - 1)
    covariance = 0.5 * (covariance + covariance.T)
    regularized = covariance + cov_eps * np.eye(representation_dim, dtype=np.float64)
    precision_matrix = np.linalg.pinv(regularized)
    raw_eig = np.linalg.eigvalsh(covariance)
    reg_eig = np.linalg.eigvalsh(regularized)
    positive = raw_eig[raw_eig > 0]
    diagnostics: Dict[str, object] = {
        "num_fit_events": int(selected),
        "fit_fraction_realized": float(selected / total_events),
        "latent_dim": int(representation_dim),
        "cov_eps": float(cov_eps),
        "raw_min_eigenvalue": float(raw_eig.min()),
        "raw_max_eigenvalue": float(raw_eig.max()),
        "raw_condition_number_positive_spectrum": (
            float(raw_eig.max() / positive.min()) if len(positive) else float("inf")
        ),
        "regularized_min_eigenvalue": float(reg_eig.min()),
        "regularized_max_eigenvalue": float(reg_eig.max()),
        "regularized_condition_number": float(reg_eig.max() / reg_eig.min()),
    }
    return mean, covariance, precision_matrix, diagnostics


def mahalanobis_scores(
    latents: np.ndarray,
    mean: np.ndarray,
    precision_matrix: np.ndarray,
) -> np.ndarray:
    centered = np.asarray(latents, dtype=np.float64) - mean
    return np.einsum("ni,ij,nj->n", centered, precision_matrix, centered)


def _fraction_tag(fraction: float) -> str:
    percent = 100.0 * fraction
    if math.isclose(percent, round(percent), rel_tol=0.0, abs_tol=1e-10):
        return f"{int(round(percent))}pct"
    return f"{percent:.3f}".rstrip("0").rstrip(".").replace(".", "p") + "pct"


def score_and_select_top_events(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    total_events: int,
    mean: np.ndarray,
    precision_matrix: np.ndarray,
    top_fraction: float,
    device: torch.device,
    precision: str,
    max_steps: Optional[int],
    output_dir: Path,
    max_num_particles: int,
    num_features: int,
    heap_progress_every: int,
) -> Dict[str, object]:
    if not 0.0 < top_fraction < 1.0:
        raise ValueError("--top-fraction must lie strictly between 0 and 1.")
    top_count = max(1, int(math.ceil(total_events * top_fraction)))
    tag = _fraction_tag(top_fraction)

    all_scores_path = output_dir / "real_test_global_mahalanobis_scores.npy"
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

    heap: List[Tuple[float, int]] = []
    offset = 0
    score_sum = 0.0
    score_sumsq = 0.0
    score_min = float("inf")
    score_max = float("-inf")
    next_report = max(1, int(heap_progress_every))

    for batch in iter_limited(loader, max_steps=max_steps, description="Score complete real test"):
        x_cpu = batch["x_particles"]
        batch_size = int(x_cpu.shape[0])
        if offset + batch_size > total_events:
            raise RuntimeError("Score pass produced more events than the count pass.")
        x = x_cpu.to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            z = encode_single_view(model, x, mask)
        z_np = z.detach().float().cpu().numpy()
        scores = mahalanobis_scores(z_np, mean, precision_matrix)
        scores32 = scores.astype(np.float32, copy=False)
        all_scores[offset : offset + batch_size] = scores32

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
    final_events = np.lib.format.open_memmap(
        final_events_path,
        mode="w+",
        dtype=np.float32,
        shape=(top_count, max_num_particles, num_features),
    )
    final_scores = np.lib.format.open_memmap(
        final_scores_path, mode="w+", dtype=np.float32, shape=(top_count,)
    )
    copy_chunk = 256
    for start in tqdm(range(0, top_count, copy_chunk), desc="Order selected events"):
        stop = min(start + copy_chunk, top_count)
        slots = np.asarray([heap[i][1] for i in range(start, stop)], dtype=np.int64)
        final_events[start:stop] = temp_events[slots]
        final_scores[start:stop] = np.asarray(
            [heap[i][0] for i in range(start, stop)], dtype=np.float32
        )
    final_events.flush()
    final_scores.flush()
    cutoff = float(heap[-1][0])

    del final_events, final_scores, temp_events, all_scores
    temp_events_path.unlink(missing_ok=True)

    mean_score = score_sum / total_events
    variance = max(0.0, score_sumsq / total_events - mean_score * mean_score)
    return {
        "all_scores_path": all_scores_path,
        "events_path": final_events_path,
        "event_scores_path": final_scores_path,
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
        label=f"top-event cut = {cutoff:.3g}",
    )
    ax.axvspan(cutoff, high, color="#B7657B", alpha=0.08)
    ax.set_xlabel("Global squared Mahalanobis score")
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


def linear_nonnegative_unit(
    values: np.ndarray,
    *,
    upper_quantile: float = 0.95,
    feature_name: str,
) -> Tuple[np.ndarray, float, float]:
    """Linearly map a nonnegative significance feature to [0, 1].

    The visualization uses the native significance values directly:

        low  = 0
        high = the requested upper quantile of the finite raw values

    Values above ``high`` are saturated at 1. No logarithm, asinh, signed
    transformation, centering, or other nonlinear mapping is applied.
    """
    raw = np.asarray(values, dtype=np.float64)
    finite_mask = np.isfinite(raw)
    finite = raw[finite_mask]
    if len(finite) == 0:
        raise ValueError(
            f"Cannot color by {feature_name}: no finite significance values."
        )

    negative = finite[finite < 0.0]
    if len(negative) > 0:
        raise ValueError(
            f"{feature_name} is expected to be nonnegative, but found "
            f"{len(negative)} negative values; minimum={float(negative.min()):.6g}."
        )

    low = 0.0
    high = float(np.quantile(finite, upper_quantile))
    if not np.isfinite(high):
        raise ValueError(
            f"Cannot color by {feature_name}: non-finite {upper_quantile:.3f} quantile."
        )

    if high <= low:
        unit = np.zeros_like(raw, dtype=np.float64)
        return unit, low, high

    unit = (raw - low) / (high - low)
    unit = np.clip(unit, 0.0, 1.0)
    unit[~finite_mask] = 0.0
    return unit, low, high


def bivariate_edge_colors(dxy_unit: np.ndarray, dz_unit: np.ndarray) -> np.ndarray:
    # Muted four-corner palette. Horizontal motion is controlled only by dxy;
    # vertical motion is controlled only by dz, with bilinear interpolation.
    c00 = np.asarray(to_rgb("#0D0D0D"))  # low dxy, low dz
    c10 = np.asarray(to_rgb("#3BF707"))  # high dxy, low dz
    c01 = np.asarray(to_rgb("#082DFD"))  # low dxy, high dz
    c11 = np.asarray(to_rgb("#F22704"))  # high dxy, high dz
    x = np.asarray(dxy_unit, dtype=np.float64)[..., None]
    y = np.asarray(dz_unit, dtype=np.float64)[..., None]
    return (
        (1.0 - x) * (1.0 - y) * c00
        + x * (1.0 - y) * c10
        + (1.0 - x) * y * c01
        + x * y * c11
    )


def plot_random_jet_events(
    *,
    events_path: Path,
    feature_names: Sequence[str],
    output_path: Path,
    num_events: int,
    seed: int,
    scores_path: Optional[Path] = None,
) -> List[int]:
    events = np.load(events_path, mmap_mode="r")
    if events.ndim != 3:
        raise ValueError(f"Expected events array shape (E, N, F), got {events.shape}.")
    if events.shape[2] != len(feature_names):
        raise ValueError(
            f"Events feature dimension {events.shape[2]} does not match feature list "
            f"length {len(feature_names)}."
        )
    required = {
        "part_deta",
        "part_dphi",
        "part_pt",
        "part_charge",
        "Cpfcan_dxysig",
        "Cpfcan_dzsig",
        *IDENTITY_FEATURES,
    }
    missing = sorted(required - set(feature_names))
    if missing:
        raise ValueError(f"Cannot visualize jets; missing features: {missing}")
    if len(events) == 0:
        raise ValueError(f"Events file is empty: {events_path}")
    n = min(max(1, int(num_events)), len(events))
    rng = np.random.default_rng(seed)
    selected_indices = np.sort(rng.choice(len(events), size=n, replace=False))
    selected = np.asarray(events[selected_indices], dtype=np.float32)
    scores = None
    if scores_path is not None:
        score_array = np.load(scores_path, mmap_mode="r")
        if len(score_array) != len(events):
            raise ValueError(
                f"Score array length {len(score_array)} does not match events {len(events)}."
            )
        scores = np.asarray(score_array[selected_indices], dtype=np.float64)

    index = {name: feature_names.index(name) for name in feature_names}
    valid_masks = ~np.all(selected == 0, axis=-1)
    all_valid = selected[valid_masks]
    if len(all_valid) == 0:
        raise ValueError("Selected jets contain no non-padding particle candidates.")

    all_dxy_unit, dxy_low, dxy_high = linear_nonnegative_unit(
        all_valid[:, index["Cpfcan_dxysig"]],
        upper_quantile=0.95,
        feature_name="Cpfcan_dxysig",
    )
    all_dz_unit, dz_low, dz_high = linear_nonnegative_unit(
        all_valid[:, index["Cpfcan_dzsig"]],
        upper_quantile=0.95,
        feature_name="Cpfcan_dzsig",
    )

    coord_abs = np.concatenate(
        [
            np.abs(all_valid[:, index["part_deta"]]),
            np.abs(all_valid[:, index["part_dphi"]]),
        ]
    )
    axis_limit = max(0.15, float(np.max(coord_abs)) * 1.08)

    ncols = min(4, max(1, int(math.ceil(math.sqrt(n)))))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.35 * ncols + 1.8, 3.35 * nrows + 1.1),
        squeeze=False,
    )

    global_offset = 0
    for panel_index, event_index in enumerate(selected_indices):
        ax = axes.flat[panel_index]
        event = selected[panel_index]
        valid = valid_masks[panel_index]
        particles = event[valid]
        count = len(particles)
        dxy_unit = all_dxy_unit[global_offset : global_offset + count]
        dz_unit = all_dz_unit[global_offset : global_offset + count]
        global_offset += count
        edge_colors = bivariate_edge_colors(dxy_unit, dz_unit)

        identities = particles[:, [index[name] for name in IDENTITY_FEATURES]]
        identity_argmax = identities.argmax(axis=1)
        identity_strength = identities.max(axis=1)
        identity_names = np.asarray(
            [IDENTITY_FEATURES[i] for i in identity_argmax], dtype=object
        )
        identity_names[identity_strength <= 0.5] = "unidentified"

        charge_values = particles[:, index["part_charge"]]
        charge_class = np.where(charge_values > 0.5, 1, np.where(charge_values < -0.5, -1, 0))
        face_colors = np.asarray([CHARGE_COLORS[int(c)] for c in charge_class])
        pt_frac = np.clip(particles[:, index["part_pt"]], 0.0, 1.0)
        sizes = 8 + 240.0 * pt_frac

        for identity_name, marker in IDENTITY_MARKERS.items():
            mask = identity_names == identity_name
            if not np.any(mask):
                continue
            ax.scatter(
                particles[mask, index["part_deta"]],
                particles[mask, index["part_dphi"]],
                s=sizes[mask],
                marker=marker,
                c=face_colors[mask],
                edgecolors=edge_colors[mask],
                linewidths=1.25,
                alpha=0.5,
            )

        title = f"event {int(event_index)}"
        if scores is not None:
            title += f"\nscore {scores[panel_index]:.1f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_xticks([-axis_limit, 0, axis_limit], labels=[f"{-axis_limit:.2f}", "0.00", f"{axis_limit:.2f}"], fontsize=8)
        ax.set_yticks([-axis_limit, 0, axis_limit], labels=[f"{-axis_limit:.2f}", "0.00", f"{axis_limit:.2f}"], fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.set_box_aspect(1)
        ax.axhline(0, color="#D7DADF", linewidth=0.6, zorder=0)
        ax.axvline(0, color="#D7DADF", linewidth=0.6, zorder=0)
        ax.set_xlabel(r"$\Delta\eta$")
        ax.set_ylabel(r"$\Delta\phi$")
        ax.grid(False)

    for ax in axes.flat[n:]:
        ax.set_visible(False)

    identity_handles = [
        Line2D(
            [0],
            [0],
            marker=IDENTITY_MARKERS[name],
            color="none",
            markerfacecolor="#8B8F97",
            markeredgecolor="#59636C",
            markersize=7,
            label=IDENTITY_LABELS[name],
        )
        for name in IDENTITY_FEATURES
    ]
    charge_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=CHARGE_COLORS[value],
            markeredgecolor="none",
            markersize=7,
            label=label,
        )
        for value, label in ((1, "charge +1"), (0, "charge 0"), (-1, "charge -1"))
    ]
    fig.legend(
        handles=identity_handles + charge_handles,
        loc="lower center",
        bbox_to_anchor=(0.43, 0.01),
        ncol=min(8, len(identity_handles) + len(charge_handles)),
        frameon=False,
        fontsize=8,
    )

    # A compact bivariate key for the outline color.
    key_ax = fig.add_axes([0.865, 0.06, 0.105, 0.105])
    grid = np.linspace(0.0, 1.0, 64)
    xx, yy = np.meshgrid(grid, grid)
    key_ax.imshow(
        bivariate_edge_colors(xx, yy),
        origin="lower",
        extent=(0, 1, 0, 1),
        aspect="equal",
    )
    key_ax.set_xlabel("dxy sig (raw)", fontsize=7, labelpad=1)
    key_ax.set_ylabel("dz sig (raw)", fontsize=7, labelpad=1)
    key_ax.set_xticks(
        [0, 1],
        [f"{dxy_low:.3g}", f"{dxy_high:.3g}"],
        fontsize=6,
    )
    key_ax.set_yticks(
        [0, 1],
        [f"{dz_low:.3g}", f"{dz_high:.3g}"],
        fontsize=6,
    )
    key_ax.set_title(
        "outline color\nlinear; high = raw q95",
        fontsize=7,
        pad=2,
    )

    fig.suptitle(
        "Random sample from highest-scoring held-out real-data jets",
        fontsize=13,
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.07,
        right=0.84,
        bottom=0.15,
        top=0.91,
        wspace=0.34,
        hspace=0.45,
    )
    fig.savefig(output_path, dpi=230, bbox_inches="tight")
    plt.close(fig)
    return [int(x) for x in selected_indices]


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
    selected = plot_random_jet_events(
        events_path=events_path,
        feature_names=load_feature_names(feature_path),
        output_path=output_path,
        num_events=args.num_visualize,
        seed=seed,
        scores_path=score_path,
    )
    print(f"Saved visualization to {output_path}")
    print(f"Selected event-array indices: {selected}")


def main() -> None:
    args = parse_args()
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
    num_workers = min(requested_workers, len(backend.test_files))
    if num_workers != requested_workers:
        warnings.warn(
            f"Reduced num_workers from {requested_workers} to {num_workers} so "
            "every worker receives at least one label_Real test shard."
        )
    fit_fraction = float(args.fit_fraction)
    top_fraction = float(args.top_fraction)
    cov_eps = float(
        args.mahalanobis_cov_eps
        if args.mahalanobis_cov_eps is not None
        else summary.get("mahalanobis_cov_eps", 1e-4)
    )
    if cov_eps <= 0:
        raise ValueError("--mahalanobis-cov-eps must be positive.")
    precision = str(summary.get("precision", "fp32"))
    representation_dim = int(summary["representation_dim"])
    loader_seed = seed + 404

    print(f"Run directory: {run_dir}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset root: {backend.dataset_root}")
    print(f"Manifest: {backend.manifest_path}")
    print(f"label_Real test shards: {len(backend.test_files)}")
    print(f"Device: {device}; batch size: {batch_size}; workers: {num_workers}")
    print(f"Fit fraction: {fit_fraction}; top fraction: {top_fraction}")
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
    if total_events < 4:
        raise RuntimeError(f"Only {total_events} held-out test events were found.")
    print(f"Finite held-out test events: {total_events:,}")

    model = build_model(summary, backend, device)
    load_model_weights(model, checkpoint_path, device)

    mean, covariance, precision_matrix, gaussian_diagnostics = fit_global_gaussian(
        model=model,
        loader=make_loader(),
        total_events=total_events,
        fit_fraction=fit_fraction,
        seed=seed + 505,
        device=device,
        precision=precision,
        cov_eps=cov_eps,
        max_steps=args.eval_steps,
        representation_dim=representation_dim,
    )
    gaussian_path = output_dir / "real_test_global_gaussian.npz"
    np.savez(
        gaussian_path,
        mean=mean,
        covariance=covariance,
        precision=precision_matrix,
        fit_count=np.asarray(gaussian_diagnostics["num_fit_events"], dtype=np.int64),
        fit_fraction=np.asarray(fit_fraction, dtype=np.float64),
        cov_eps=np.asarray(cov_eps, dtype=np.float64),
    )

    selection = score_and_select_top_events(
        model=model,
        loader=make_loader(),
        total_events=total_events,
        mean=mean,
        precision_matrix=precision_matrix,
        top_fraction=top_fraction,
        device=device,
        precision=precision,
        max_steps=args.eval_steps,
        output_dir=output_dir,
        max_num_particles=backend.max_num_particles,
        num_features=len(backend.feature_names),
        heap_progress_every=args.heap_progress_every,
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
    visualization_path = output_dir / "real_test_top_events_random_visualization.png"
    selected_visual_indices = plot_random_jet_events(
        events_path=events_path,
        feature_names=backend.feature_names,
        output_path=visualization_path,
        num_events=args.num_visualize,
        seed=seed + 707,
        scores_path=Path(selection["event_scores_path"]),
    )

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
        "fit_fraction_requested": fit_fraction,
        "top_fraction_requested": top_fraction,
        "gaussian": gaussian_diagnostics,
        "score_stats": selection["score_stats"],
        "histogram_stats": histogram_stats,
        "top_count": selection["top_count"],
        "top_fraction_realized": selection["top_fraction_realized"],
        "top_cutoff": selection["cutoff"],
        "random_visualized_event_array_indices": selected_visual_indices,
        "files": {
            "gaussian": gaussian_path,
            "all_scores": selection["all_scores_path"],
            "top_events": selection["events_path"],
            "top_event_scores": selection["event_scores_path"],
            "top_event_features": feature_list_path,
            "score_distribution": distribution_path,
            "random_visualization": visualization_path,
        },
    }
    results_path = output_dir / "real_test_evaluation.json"
    with results_path.open("w") as handle:
        json.dump(safe_json(results), handle, indent=2)

    print(f"Saved evaluation outputs to {output_dir}")
    print(f"Top-event cut: {float(selection['cutoff']):.6g}")
    print(f"Top events: {selection['top_count']:,} -> {selection['events_path']}")


if __name__ == "__main__":
    main()
