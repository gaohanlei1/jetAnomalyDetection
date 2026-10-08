#!/usr/bin/env python3
"""Post-training latent diagnostics for JetClass or CMS LeJEPA runs.

The dataset backend, particle feature order, batch-standardized feature list,
label axis, and CMS per-type shuffled-shard split are reconstructed from the
training ``summary.json``.  A command-line dataset override is available for
explicit recovery workflows, but normal use requires only the run directory.

All representations are collected once from validation and retained in RAM;
representations are never serialized. Background validation is both the fitting
and scoring sample (kNN excludes each background event itself). Each signal has
its own ROC/distribution pair per score. Gaussian mixtures use negative log
probability density so that higher values consistently mean more anomalous.
The labeled mixture fixes one Gaussian per background class, with empirical
class-frequency weights; component count, EM iterations and restarts apply
only to the unlabeled mixture. Covariance type and regularization are shared.
Joint-class Mahalanobis calibrates each class distance against that class's
own background distances, then scores 1 - max_k(2 * min(q_k, 1 - q_k)).

Example:
    python -u scripts/diagnose_lejepa_latents.py \
        plots/run-lejepa-semi-sup-triplet
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from datasets import cms_streaming
from datasets import jetclass_streaming
from models.part_jetclass import (
    CorruptedNegativeAugmentationConfig,
    LeJEPALossConfig,
    LeJEPASemiSupervisedTripletParticleTransformerRepresentation,
    MultiViewAugmentationConfig,
    ParticleTransformerConfig,
    SemiSupervisedLossConfig,
    TripletLossConfig,
)

FOUR_VECTOR_FEATURES = (
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
)



def _construct_with_supported_kwargs(factory, **kwargs):
    """Call a class/function while tolerating removed compatibility kwargs."""

    parameters = inspect.signature(factory).parameters
    supported = {key: value for key, value in kwargs.items() if key in parameters}
    return factory(**supported)


def _first_summary_list(
    summary: Mapping[str, object],
    keys: Sequence[str],
) -> Optional[List[str]]:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return [str(item) for item in value]
    return None


def _require_summary_list(
    summary: Mapping[str, object],
    key: str,
) -> List[str]:
    """Read a required non-empty ordered string list from summary.json."""
    if key not in summary or summary[key] is None:
        raise KeyError(
            f"summary.json is missing required field {key!r}. "
            "Diagnostics must use the exact feature schema saved by training."
        )
    value = summary[key]
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(
            f"summary.json field {key!r} must be a list of feature names, "
            f"got {type(value).__name__}."
        )
    result = [str(item) for item in value]
    if not result:
        raise ValueError(f"summary.json field {key!r} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(
            f"summary.json field {key!r} contains duplicate feature names: "
            f"{result}."
        )
    return result


def _dataset_metadata_list(
    dataset: object,
    names: Sequence[str],
) -> Optional[List[str]]:
    for name in names:
        value = getattr(dataset, name, None)
        if value is not None:
            return [str(item) for item in value]
    return None


def _validate_four_vector_prefix(features: Sequence[str], source: str) -> None:
    actual = tuple(features[:4])
    if actual != FOUR_VECTOR_FEATURES:
        raise ValueError(
            f"{source} must begin with the ordered four-vector features "
            f"{list(FOUR_VECTOR_FEATURES)}, found {list(actual)}."
        )


def _strip_common_prefix(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


@dataclass
class DiagnosticDatasetBackend:
    dataset_name: str
    dataset_root: Path
    run_dir: Path
    summary: Mapping[str, object]
    feature_names: List[str]
    batch_standardized_feature_names: List[str]
    label_axis: List[str]
    max_num_particles: int
    min_nodes: int
    shuffle_active_shards: int
    cms_splits: Optional[Dict[str, Dict[str, List[str]]]] = None
    cms_manifest_sha256: Optional[str] = None
    cms_manifest_source: Optional[str] = None

    @classmethod
    def from_summary(
        cls,
        *,
        summary: Mapping[str, object],
        run_dir: Path,
        dataset_override: Optional[str],
        dataset_root_override: Optional[Path],
        cms_manifest_override: Optional[Path],
        max_num_particles_override: Optional[int],
    ) -> "DiagnosticDatasetBackend":
        dataset_name = str(dataset_override or summary.get("dataset", "jetclass"))
        if dataset_name not in {"jetclass", "cms"}:
            raise ValueError(
                f"Unsupported dataset {dataset_name!r}; expected 'jetclass' or 'cms'."
            )

        if dataset_root_override is not None:
            dataset_root = dataset_root_override.expanduser().resolve()
        else:
            if "dataset_root" not in summary:
                raise KeyError(
                    "summary.json has no dataset_root; pass --dataset-root explicitly."
                )
            dataset_root = Path(str(summary["dataset_root"])).expanduser().resolve()

        features = _require_summary_list(summary, "particle_features")
        _validate_four_vector_prefix(features, "Saved particle feature list")

        standardized = _require_summary_list(
            summary,
            "batch_standardized_particle_features",
        )
        missing_standardized = sorted(set(standardized) - set(features))
        if missing_standardized:
            raise ValueError(
                "summary.json requests batch standardization for features absent "
                f"from the input schema: {missing_standardized}."
            )

        if dataset_name == "jetclass":
            default_axis = list(jetclass_streaming.JETCLASS_LABELS)
        else:
            default_axis = list(cms_streaming.CMS_LABELS)
        label_axis = _first_summary_list(summary, ("dataset_label_axis",)) or default_axis

        max_num_particles = int(
            max_num_particles_override
            if max_num_particles_override is not None
            else summary.get("max_num_particles", 128)
        )
        min_nodes = int(summary.get("min_nodes", 4))
        shuffle_active_shards = int(summary.get("shuffle_active_shards", 3))

        backend = cls(
            dataset_name=dataset_name,
            dataset_root=dataset_root,
            run_dir=run_dir,
            summary=summary,
            feature_names=features,
            batch_standardized_feature_names=standardized,
            label_axis=label_axis,
            max_num_particles=max_num_particles,
            min_nodes=min_nodes,
            shuffle_active_shards=shuffle_active_shards,
        )
        backend._initialize_dataset(cms_manifest_override)
        return backend

    def _initialize_dataset(self, cms_manifest_override: Optional[Path]) -> None:
        if self.dataset_name == "jetclass":
            for split_name, directory_name in (("val", "val_5M"),):
                directory = self.dataset_root / directory_name
                if not directory.is_dir():
                    raise FileNotFoundError(
                        f"Missing JetClass {split_name} split directory: {directory}"
                    )
            return

        manifest_path = (
            cms_manifest_override.expanduser().resolve()
            if cms_manifest_override is not None
            else self.run_dir / "cms_split_manifest.json"
        )
        expected_hash = self.summary.get("cms_split_manifest_sha256")

        if manifest_path.is_file():
            with manifest_path.open() as handle:
                manifest = json.load(handle)
            if "splits" not in manifest:
                raise ValueError(
                    f"CMS split manifest has no 'splits' mapping: {manifest_path}"
                )
            if int(manifest.get("version", 0)) != 2:
                raise ValueError(
                    "The CMS split manifest uses the obsolete production-family "
                    f"layout: {manifest_path}. Recreate it with the shuffled-shard "
                    "CMS pipeline."
                )
            self.cms_manifest_sha256 = str(manifest.get("sha256", "")) or None
            self.cms_manifest_source = str(manifest_path)
            if (
                expected_hash is not None
                and self.cms_manifest_sha256 is not None
                and str(expected_hash) != self.cms_manifest_sha256
            ):
                raise RuntimeError(
                    "The saved CMS split manifest hash does not match summary.json: "
                    f"summary={expected_hash}, manifest={self.cms_manifest_sha256}."
                )
            self.cms_splits = self._resolve_manifest_paths(manifest["splits"])
        else:
            requested_labels = list(dict.fromkeys(
                list(self.summary["background_labels"])
                + list(self.summary.get("signal_labels", []))
            ))
            discovered = cms_streaming.discover_cms_files(
                str(self.dataset_root), requested_labels
            )
            self.cms_splits = cms_streaming.split_cms_files(
                discovered,
                val_fraction=float(self.summary.get("cms_val_fraction", 0.1)),
                test_fraction=float(self.summary.get("cms_test_fraction", 0.1)),
                seed=int(self.summary.get("cms_split_seed", 42)),
            )
            rebuilt_manifest = cms_streaming.cms_split_manifest(self.cms_splits)
            self.cms_manifest_sha256 = str(rebuilt_manifest["sha256"])
            self.cms_manifest_source = "reconstructed from summary.json"
            if expected_hash is not None and str(expected_hash) != self.cms_manifest_sha256:
                raise RuntimeError(
                    "Reconstructed CMS split does not match the split used by "
                    "training. "
                    f"summary={expected_hash}, reconstructed="
                    f"{self.cms_manifest_sha256}. Restore cms_split_manifest.json "
                    "from the run directory."
                )

        assert self.cms_splits is not None
        for split_name in ("train", "val", "test"):
            if split_name not in self.cms_splits:
                raise ValueError(f"CMS split mapping is missing {split_name!r}.")

    def _resolve_manifest_paths(
        self,
        splits: Mapping[str, Mapping[str, Sequence[str]]],
    ) -> Dict[str, Dict[str, List[str]]]:
        old_root_value = self.summary.get("dataset_root")
        old_root = (
            Path(str(old_root_value)).expanduser()
            if old_root_value is not None
            else None
        )
        resolved: Dict[str, Dict[str, List[str]]] = {}
        missing: List[str] = []
        for split_name, labels in splits.items():
            resolved[split_name] = {}
            for label, paths in labels.items():
                if isinstance(paths, Mapping):
                    raise ValueError(
                        "CMS manifest contains nested production-family mappings; "
                        "the shuffled-shard pipeline expects label -> path list."
                    )
                resolved_paths: List[str] = []
                for raw_path in paths:
                    path = Path(str(raw_path)).expanduser()
                    if path.is_file():
                        candidate = path.resolve()
                    else:
                        candidate = None
                        if old_root is not None:
                            try:
                                relative = path.relative_to(old_root)
                            except ValueError:
                                relative = None
                            if relative is not None:
                                relocated = (self.dataset_root / relative).resolve()
                                if relocated.is_file():
                                    candidate = relocated
                        if candidate is None:
                            directory_name = cms_streaming.CMS_LABEL_TO_DIRECTORY[label]
                            candidate = (
                                self.dataset_root / directory_name / path.name
                            ).resolve()
                    if split_name == "val" and not candidate.is_file():
                        missing.append(str(candidate))
                    resolved_paths.append(str(candidate))
                resolved[split_name][label] = resolved_paths
        if missing:
            preview = "\n  ".join(missing[:10])
            raise FileNotFoundError(
                "CMS split manifest references ROOT shards that do not exist "
                f"under the resolved dataset root. First missing paths:\n  {preview}"
            )
        return resolved

    def validate_requested_labels(self, labels: Sequence[str]) -> None:
        unknown = sorted(set(labels) - set(self.label_axis))
        if unknown:
            raise ValueError(
                f"Labels {unknown} are absent from the saved dataset label axis "
                f"{self.label_axis}."
            )
        if self.dataset_name == "jetclass":
            jetclass_streaming.validate_requested_labels(labels)
        else:
            cms_streaming.validate_cms_labels(labels)

    def _validate_dataset_metadata(self, dataset: object) -> None:
        dataset_features = _dataset_metadata_list(
            dataset,
            ("feature_names", "particle_features"),
        )
        if dataset_features is not None and dataset_features != self.feature_names:
            raise RuntimeError(
                "Dataset object feature metadata disagrees with summary.json: "
                f"dataset={dataset_features}, summary={self.feature_names}."
            )
        dataset_standardized = _dataset_metadata_list(
            dataset,
            (
                "batch_normalized_feature_names",
                "batch_standardized_feature_names",
                "batch_normalized_particle_features",
            ),
        )
        if (
            dataset_standardized is not None
            and dataset_standardized != self.batch_standardized_feature_names
        ):
            raise RuntimeError(
                "Dataset object batch-normalization metadata disagrees with "
                f"summary.json: dataset={dataset_standardized}, "
                f"summary={self.batch_standardized_feature_names}."
            )

    def make_dataset(
        self,
        split_name: str,
        labels: Sequence[str],
        seed: int,
    ):
        self.validate_requested_labels(labels)
        if self.dataset_name == "jetclass":
            split_directory = {
                "train": self.dataset_root / "train_100M",
                "val": self.dataset_root / "val_5M",
                "test": self.dataset_root / "test_20M",
            }[split_name]
            dataset = jetclass_streaming.JetClassIterableDataset(
                split_dir=str(split_directory),
                labels_to_load=labels,
                particle_features=self.feature_names,
                max_num_particles=self.max_num_particles,
                max_events=None,
                shuffle_files=True,
                shuffle_active_shards=self.shuffle_active_shards,
                infinite=True,
                seed=seed,
                rank=0,
                world_size=1,
            )
        else:
            assert self.cms_splits is not None
            dataset = _construct_with_supported_kwargs(
                cms_streaming.CMSIterableDataset,
                files_by_label=self.cms_splits[split_name],
                labels_to_load=labels,
                label_axis=self.label_axis,
                particle_features=self.feature_names,
                max_num_particles=self.max_num_particles,
                min_nodes=self.min_nodes,
                lowerpt=self.summary.get("cms_pt_min"),
                upperpt=self.summary.get("cms_pt_max"),
                max_events=None,
                shuffle_files=True,
                shuffle_active_shards=self.shuffle_active_shards,
                infinite=True,
                seed=seed,
                rank=0,
                world_size=1,
            )
        self._validate_dataset_metadata(dataset)
        return dataset

    def make_loader(
        self,
        *,
        split_name: str,
        labels: Sequence[str],
        seed: int,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
    ) -> DataLoader:
        dataset = self.make_dataset(split_name, labels, seed)
        collate_fn = (
            jetclass_streaming.collate_jetclass_tensors
            if self.dataset_name == "jetclass"
            else cms_streaming.collate_cms_tensors
        )
        kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "collate_fn": collate_fn,
            "persistent_workers": False,
            "drop_last": True,
        }
        if num_workers > 0:
            kwargs["prefetch_factor"] = int(self.summary.get("prefetch_factor", 1))
        return DataLoader(**kwargs)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose latent geometry and anomaly scores of a JetClass or CMS "
            "LeJEPA semi-supervised triplet run."
        )
    )
    parser.add_argument(
        "run_dir", type=Path, help="Directory containing summary.json and a checkpoint."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        choices=["jetclass", "cms"],
        default=None,
        help="Override summary.json. Normally the saved dataset field is used.",
    )
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument(
        "--cms-split-manifest",
        type=Path,
        default=None,
        help=(
            "Optional CMS split manifest override. By default the script uses "
            "<run_dir>/cms_split_manifest.json and only reconstructs the split "
            "when that file is absent."
        ),
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help="Batches per dataset; defaults to summary.json eval_steps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Defaults to the saved per-rank batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "Diagnostic DataLoader workers. Defaults to summary.json num_workers."
        ),
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--knn-k", type=int, default=30)
    parser.add_argument("--knn-reduction", choices=("mean", "kth"), default="mean",
                        help="Mean of k Euclidean distances or kth-neighbor distance; background excludes itself.")
    parser.add_argument("--score-batch-size", type=int, default=1024)
    parser.add_argument("--knn-reference-batch-size", type=int, default=8192)
    parser.add_argument("--gmm-n-components", type=int, default=4,
                        help="Unlabeled EM components; labeled mixture always has one component per background class.")
    parser.add_argument("--gmm-covariance-type", choices=("full", "diag"), default="full")
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-4)
    parser.add_argument("--gmm-max-iter", type=int, default=100,
                        help="Unlabeled EM iteration limit (fixed-label fit is closed-form).")
    parser.add_argument("--gmm-tol", type=float, default=1e-3)
    parser.add_argument("--gmm-n-init", type=int, default=1,
                        help="Unlabeled EM restarts (fixed-label fit is deterministic).")
    parser.add_argument("--score-hist-bins", type=int, default=80)
    parser.add_argument("--mahalanobis-cov-eps", type=float, default=None)
    parser.add_argument(
        "--max-num-particles",
        type=int,
        default=None,
        help="Override summary.json; otherwise uses the training value.",
    )
    return parser.parse_args()

def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {value}")
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
    features: Sequence[str],
    candidates: Sequence[str],
    fallback: str,
) -> int:
    for name in candidates:
        if name in features:
            return features.index(name)
    return features.index(fallback)


def build_model(
    summary: Mapping[str, object],
    backend: DiagnosticDatasetBackend,
    device: torch.device,
) -> torch.nn.Module:
    features = list(backend.feature_names)
    standardized = list(backend.batch_standardized_feature_names)
    _validate_four_vector_prefix(features, "Model input feature list")
    precision = str(summary.get("precision", "fp32"))

    model_name = str(summary.get("model", "semi-sup-triplet"))
    if model_name != "semi-sup-triplet":
        raise ValueError(
            "The revised training script supports only "
            "LeJEPASemiSupervisedTripletParticleTransformerRepresentation; "
            f"summary.json records model={model_name!r}."
        )

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
    loss_config = LeJEPALossConfig(
        invariant_weight=float(summary.get("invariant_weight", 1.0)),
        sigreg_weight=float(summary.get("sigreg_weight", 0.02)),
        epps_pulley_num_points=int(summary.get("epps_pulley_num_points", 17)),
        num_slices=int(summary.get("num_slices", 1024)),
    )

    negative_augmentation_config = _construct_with_supported_kwargs(
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
            features,
            ("d0_sig", "Cpfcan_dxysig"),
            "part_charge",
        ),
        dz_sig_index=_feature_index(
            features,
            ("dz_sig", "Cpfcan_dzsig"),
            "part_charge",
        ),
        charge_index=features.index("part_charge"),
        identity_start_index=features.index("part_isChargedHadron"),
        identity_end_index=features.index("part_isMuon") + 1,
        corrupt_node_frac=float(summary.get("corrupt_node_frac", 0.5)),
        batch_mix_anchor_frac_min=float(summary.get("batch_mix_anchor_frac_min", 0.3)),
        batch_mix_anchor_frac_max=float(summary.get("batch_mix_anchor_frac_max", 0.7)),
        renormalize_pt_sum=bool(summary.get("renormalize_negative_pt_sum", True)),
    )
    triplet_loss_config = TripletLossConfig(
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
        negative_augmentation_config=negative_augmentation_config,
        loss_config=loss_config,
        triplet_loss_config=triplet_loss_config,
        semi_supervised_config=SemiSupervisedLossConfig(
            classification_weight=float(summary.get("classification_weight", 0.1)),
            num_classes=int(
                summary.get(
                    "num_classification_classes",
                    len(summary["background_labels"]),
                )
            ),
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
    state_dict = dict(state_dict)
    for prefix in ("module.", "model."):
        state_dict = _strip_common_prefix(state_dict, prefix)
    return state_dict


@torch.no_grad()
def encode_single_view(
    model: torch.nn.Module,
    x: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Encode one unaugmented view in the trained representation space."""
    cls = model(x, padding_mask=padding_mask)
    if not hasattr(model, "representation_head"):
        raise AttributeError(
            "The loaded model has no representation_head; representation-space "
            "diagnostics cannot continue."
        )
    return model.representation_head(cls)


@torch.no_grad()
def collect_full_latents(
    model: torch.nn.Module,
    loader: DataLoader,
    steps: int,
    device: torch.device,
    precision: str,
    description: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect unaugmented representation-space latents from one loader."""
    model.eval()
    zs, ys = [], []
    iterator = iter(loader)
    for _ in tqdm(range(steps), desc=description):
        batch = next(iterator)
        x = batch["x_particles"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        with autocast_context(device, precision):
            z = encode_single_view(
                model,
                x,
                padding_mask=mask,
            )
        zs.append(z.detach().float().cpu())
        ys.append(batch["y"].float())
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


def label_ids(y: np.ndarray, label_axis: Sequence[str]) -> np.ndarray:
    if y.ndim != 2 or y.shape[1] != len(label_axis):
        raise ValueError(
            f"Unexpected one-hot label shape {y.shape}; saved label axis has "
            f"{len(label_axis)} entries: {list(label_axis)}."
        )
    return np.argmax(y, axis=1).astype(np.int64)


def class_mask(
    y: np.ndarray,
    label: str,
    label_axis: Sequence[str],
) -> np.ndarray:
    return label_ids(y, label_axis) == list(label_axis).index(label)


def labels_mask(
    y: np.ndarray,
    labels: Sequence[str],
    label_axis: Sequence[str],
) -> np.ndarray:
    ids = label_ids(y, label_axis)
    requested_ids = np.asarray(
        [list(label_axis).index(label) for label in labels],
        dtype=np.int64,
    )
    return np.isin(ids, requested_ids)


@torch.no_grad()
def fit_mahalanobis(latents: torch.Tensor, cov_eps: float):
    if len(latents) < 2:
        raise ValueError("Mahalanobis fitting requires at least two background events.")
    mean = latents.mean(0)
    centered = latents - mean
    cov = centered.T @ centered / (len(latents) - 1)
    reg_cov = cov + cov_eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
    precision = torch.linalg.pinv(reg_cov, hermitian=True)
    eig = torch.linalg.eigvalsh(reg_cov)
    return mean, precision, {
        "num_fit_events": len(latents), "latent_dim": latents.shape[1],
        "cov_eps": cov_eps, "regularized_min_eigenvalue": eig.min().item(),
        "regularized_max_eigenvalue": eig.max().item(),
    }


@torch.no_grad()
def mahalanobis_scores(latents, mean, precision, batch_size=1024):
    scores = []
    for start in range(0, len(latents), batch_size):
        delta = latents[start:start + batch_size] - mean
        scores.append(((delta @ precision) * delta).sum(1).clamp_min(0).cpu().numpy())
    return np.concatenate(scores)


def mahalanobis_percentiles(class_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Empirical distance percentiles, retaining the lower/upper tail direction.

    q = (number strictly below + half the number equal) / class sample count.
    Midpoint ties treat both tails symmetrically; scores outside the observed
    range have percentile zero or one. These are not posterior probabilities.
    """
    reference = np.sort(np.asarray(class_scores, dtype=np.float64))
    scores = np.asarray(scores, dtype=np.float64)
    if reference.ndim != 1 or reference.size == 0:
        raise ValueError("Class calibration scores must be a non-empty 1D array.")
    if not np.isfinite(reference).all() or not np.isfinite(scores).all():
        raise ValueError("Mahalanobis calibration and query scores must be finite.")
    left = np.searchsorted(reference, scores, side="left")
    right = np.searchsorted(reference, scores, side="right")
    return (left + right) / (2.0 * reference.size)


def mahalanobis_compatibility(class_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """Two-sided compatibility using the class's empirical distance percentile."""
    percentile = mahalanobis_percentiles(class_scores, scores)
    return 2.0 * np.minimum(percentile, 1.0 - percentile)


@torch.no_grad()
def knn_scores(
    query: torch.Tensor,
    reference: torch.Tensor,
    k: int,
    batch_size: int,
    reference_batch_size: int,
    reduction: str = "mean",
    exclude_self: bool = False,
) -> np.ndarray:
    """Exact Euclidean kNN with bounded query/reference blocks.

    exclude_self requires query to be the reference in the same row order.
    Only the matching row is excluded; duplicate representations remain valid.
    """
    available = len(reference) - int(exclude_self)
    if not 1 <= k <= available:
        raise ValueError(f"kNN k={k} requires at least {k + int(exclude_self)} background events.")
    if exclude_self and query is not reference:
        raise ValueError("Self exclusion requires the same reference tensor as query.")
    result = []
    for start in range(0, len(query), batch_size):
        q = query[start:start + batch_size]
        best = q.new_full((len(q), k), float("inf"))
        for offset in range(0, len(reference), reference_batch_size):
            ref = reference[offset:offset + reference_batch_size]
            distances = torch.cdist(q, ref)
            if exclude_self:
                rows = torch.arange(start, start + len(q), device=q.device)
                inside = (rows >= offset) & (rows < offset + len(ref))
                distances[torch.arange(len(q), device=q.device)[inside], rows[inside] - offset] = float("inf")
            candidates = distances.topk(min(k, len(ref)), largest=False).values
            best = torch.cat((best, candidates), dim=1).topk(k, largest=False).values
        values = best.mean(dim=1) if reduction == "mean" else best.max(dim=1).values
        result.append(values.cpu().numpy())
    return np.concatenate(result)


class TorchGaussianMixture:
    """Chunked, device-native Gaussian fitting and negative log density.

    With labels, responsibilities are fixed one-hot assignments (one Gaussian
    per class). Otherwise EM learns responsibilities without any class labels.
    Both paths use MLE covariance plus reg_covar * I and empirical weights.
    """

    def __init__(self, n_components, covariance_type, reg_covar, max_iter, tol,
                 n_init, batch_size, seed):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.batch_size = batch_size
        self.seed = seed

    def _set_parameters(self, means, covariances, weights):
        self.means = means
        self.covariances = covariances
        self.weights = weights
        if self.covariance_type == "full":
            self.cholesky = torch.linalg.cholesky(covariances)
            self.logdet = 2 * self.cholesky.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        else:
            self.logdet = covariances.log().sum(-1)

    def _log_components(self, x):
        # Loop over components to avoid allocating a B x K x D x D tensor.
        columns = []
        constant = x.shape[1] * np.log(2 * np.pi)
        for j in range(len(self.means)):
            delta = x - self.means[j]
            if self.covariance_type == "full":
                whitened = torch.linalg.solve_triangular(self.cholesky[j], delta.T, upper=False)
                distance = whitened.square().sum(0)
            else:
                distance = (delta.square() / self.covariances[j]).sum(1)
            columns.append(self.weights[j].log() - 0.5 * (constant + self.logdet[j] + distance))
        return torch.stack(columns, dim=1)

    def _covariance(self, delta, weights, count):
        if self.covariance_type == "full":
            cov = delta.T @ (delta * weights[:, None]) / count
            return cov + self.reg_covar * torch.eye(delta.shape[1], device=delta.device, dtype=delta.dtype)
        return (delta.square() * weights[:, None]).sum(0) / count + self.reg_covar

    @torch.no_grad()
    def fit(self, x, labels=None):
        if len(x) < 2 or not torch.isfinite(x).all():
            raise ValueError("Gaussian mixture requires at least two finite background representations.")
        # Center once to keep sufficient-statistic subtraction well conditioned.
        self.origin = x.mean(0)
        x = x - self.origin
        if labels is not None:
            classes = torch.unique(labels, sorted=True)
            means, covariances, counts = [], [], []
            for label in classes:
                points = x[labels == label]
                if len(points) < 2:
                    raise ValueError(f"Gaussian class {label.item()} needs at least two events.")
                mean = points.mean(0)
                means.append(mean)
                covariances.append(self._covariance(points - mean, points.new_ones(len(points)), len(points)))
                counts.append(len(points))
            self._set_parameters(torch.stack(means), torch.stack(covariances), x.new_tensor(counts) / len(x))
            self.n_iter_, self.converged_ = 0, True
            self.lower_bound_ = self._mean_log_density(x)
            return self
        if not 1 <= self.n_components <= len(x):
            raise ValueError("--gmm-n-components must be between 1 and the background event count.")
        generator = torch.Generator(device=x.device).manual_seed(self.seed)
        global_cov = self._covariance(x, x.new_ones(len(x)), len(x))
        best = None
        for _ in range(self.n_init):
            # k-means++ seeding uses only representations, never class labels.
            indices = [int(torch.randint(len(x), (1,), generator=generator, device=x.device).item())]
            nearest = (x - x[indices[0]]).square().sum(1)
            for _ in range(1, self.n_components):
                if nearest.sum() > 0:
                    index = int(torch.multinomial(nearest, 1, generator=generator).item())
                else:
                    remaining = torch.ones(len(x), device=x.device, dtype=torch.bool)
                    remaining[indices] = False
                    index = int(torch.nonzero(remaining)[0].item())
                indices.append(index)
                nearest = torch.minimum(nearest, (x - x[index]).square().sum(1))
            self._set_parameters(x[indices].clone(), global_cov.unsqueeze(0).repeat(self.n_components, *([1] * global_cov.ndim)), x.new_full((self.n_components,), 1 / self.n_components))
            previous = self._mean_log_density(x)
            converged = False
            for iteration in range(1, self.max_iter + 1):
                counts = x.new_zeros(self.n_components)
                sums = torch.zeros_like(self.means)
                moments = torch.zeros_like(self.covariances)
                # Accumulate around each OLD mean rather than raw x*x moments.
                for start in range(0, len(x), self.batch_size):
                    points = x[start:start + self.batch_size]
                    responsibilities = self._log_components(points).softmax(1)
                    counts += responsibilities.sum(0)
                    for j in range(self.n_components):
                        delta = points - self.means[j]
                        weighted = delta * responsibilities[:, j:j + 1]
                        sums[j] += weighted.sum(0)
                        if self.covariance_type == "full":
                            moments[j] += delta.T @ weighted
                        else:
                            moments[j] += (delta * weighted).sum(0)
                counts = counts.clamp_min(torch.finfo(x.dtype).eps)
                shifts = sums / counts[:, None]
                means = self.means + shifts
                if self.covariance_type == "full":
                    cov = moments / counts[:, None, None] - shifts[:, :, None] * shifts[:, None, :]
                    # Roundoff can introduce tiny negative eigenvalues.
                    eig, vectors = torch.linalg.eigh((cov + cov.transpose(-1, -2)) / 2)
                    cov = (vectors * eig.clamp_min(0).unsqueeze(-2)) @ vectors.transpose(-1, -2)
                    cov += self.reg_covar * torch.eye(x.shape[1], device=x.device, dtype=x.dtype)
                else:
                    cov = (moments / counts[:, None] - shifts.square()).clamp_min(0) + self.reg_covar
                self._set_parameters(means, cov, counts / counts.sum())
                current = self._mean_log_density(x)
                if abs(current - previous) < self.tol:
                    converged = True
                    break
                previous = current
            if best is None or current > best[0]:
                best = (current, self.means.clone(), self.covariances.clone(), self.weights.clone(), iteration, converged)
        self.lower_bound_, means, covariances, weights, self.n_iter_, self.converged_ = best
        self._set_parameters(means, covariances, weights)
        if not self.converged_:
            warnings.warn("Unlabeled GMM did not converge; consider increasing --gmm-max-iter.")
        return self

    def _mean_log_density(self, x):
        total = x.new_zeros(())
        for start in range(0, len(x), self.batch_size):
            total += self._log_components(x[start:start + self.batch_size]).logsumexp(1).sum()
        return float((total / len(x)).item())

    @torch.no_grad()
    def scores(self, x):
        return np.concatenate([
            (-self._log_components(x[start:start + self.batch_size] - self.origin).logsumexp(1)).cpu().numpy()
            for start in range(0, len(x), self.batch_size)
        ])

    def diagnostics(self):
        return {"n_components": len(self.means), "covariance_type": self.covariance_type,
                "reg_covar": self.reg_covar, "weights": self.weights.cpu().tolist(),
                "n_iter": self.n_iter_, "converged": self.converged_,
                "mean_fit_log_density": self.lower_bound_}


def auc(background: np.ndarray, signal: np.ndarray) -> float:
    y = np.concatenate(
        [np.zeros(len(background), dtype=np.int64), np.ones(len(signal), dtype=np.int64)]
    )
    return float(roc_auc_score(y, np.concatenate([background, signal])))


def score_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "median": float(np.median(x)),
        "q05": float(np.quantile(x, 0.05)),
        "q95": float(np.quantile(x, 0.95)),
    }

def plot_pair_latent_space(
    background: np.ndarray,
    signal: np.ndarray,
    background_label: str,
    signal_label: str,
    path: Path,
    seed: int,
    max_points: int | None = None,
    annotate_roles: bool = True,
) -> None:
    """Plot one pair of classes in one PCA basis."""

    background = np.asarray(background, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    if background.ndim != 2 or signal.ndim != 2:
        raise ValueError(
            "Expected background and signal latents with shape (N, D), got "
            f"{background.shape} and {signal.shape}."
        )
    if background.shape[1] != signal.shape[1]:
        raise ValueError(
            "Background and signal latent dimensions differ: "
            f"{background.shape[1]} vs {signal.shape[1]}."
        )
    if len(background) == 0 or len(signal) == 0:
        raise ValueError(
            "Cannot plot an empty pairwise latent sample: "
            f"background={len(background)}, signal={len(signal)}."
        )

    rng = np.random.default_rng(seed)
    background_plot = background
    signal_plot = signal

    if max_points is not None:
        max_points = int(max_points)
        if max_points < 1:
            raise ValueError("max_points must be positive when provided.")

        if len(background_plot) > max_points:
            indices = rng.choice(len(background_plot), max_points, replace=False)
            background_plot = background_plot[indices]

        if len(signal_plot) > max_points:
            indices = rng.choice(len(signal_plot), max_points, replace=False)
            signal_plot = signal_plot[indices]

    combined = np.concatenate([background_plot, signal_plot], axis=0)
    combined = combined - combined.mean(axis=0, keepdims=True)

    _, singular_values, vh = np.linalg.svd(combined, full_matrices=False)
    components = vh[:2].T
    reduced = combined @ components

    num_background = len(background_plot)
    background_2d = reduced[:num_background]
    signal_2d = reduced[num_background:]

    total_variance = np.square(singular_values).sum()
    explained_variance_ratio = (
        np.square(singular_values[:2]) / total_variance
        if total_variance > 0.0
        else np.zeros(2, dtype=np.float64)
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        background_2d[:, 0],
        background_2d[:, 1],
        s=10,
        alpha=0.45,
        marker="o",
        label=(
            f"{background_label} (Background)"
            if annotate_roles
            else background_label
        ),
    )
    ax.scatter(
        signal_2d[:, 0],
        signal_2d[:, 1],
        s=18,
        alpha=0.65,
        marker="x",
        label=(
            f"{signal_label} (Signal)"
            if annotate_roles
            else signal_label
        ),
    )

    ax.set_xlabel(f"PC1 ({100.0 * explained_variance_ratio[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100.0 * explained_variance_ratio[1]:.1f}% variance)")
    ax.set_title(f"Pairwise validation latent space: {background_label} vs {signal_label}")
    ax.grid(alpha=0.2)
    ax.legend(
        loc="best",
        fontsize=8,
        frameon=True,
        markerscale=1.3,
    )

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pair_lda_residual_pca_space(
    background: np.ndarray,
    signal: np.ndarray,
    background_label: str,
    signal_label: str,
    path: Path,
    seed: int,
    cov_eps: float,
    max_points: int | None = None,
    annotate_roles: bool = True,
) -> None:
    """Plot Fisher LDA1 against PCA1 of the residual orthogonal subspace.

    The Fisher direction is fitted on the displayed background-signal pair.
    After Euclidean projection onto the unit LDA direction, that component is
    removed from every centered latent. PCA is then fitted to the residuals,
    and its leading component supplies the second visualization axis.
    """

    background = np.asarray(background, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)

    if background.ndim != 2 or signal.ndim != 2:
        raise ValueError(
            "Expected background and signal latents with shape (N, D), got "
            f"{background.shape} and {signal.shape}."
        )
    if background.shape[1] != signal.shape[1]:
        raise ValueError(
            "Background and signal latent dimensions differ: "
            f"{background.shape[1]} vs {signal.shape[1]}."
        )
    if background.shape[1] < 2:
        raise ValueError(
            "LDA1 + residual PCA1 plotting requires latent dimension >= 2."
        )
    if len(background) < 2 or len(signal) < 2:
        raise ValueError(
            "At least two events per class are required for pairwise LDA: "
            f"background={len(background)}, signal={len(signal)}."
        )

    rng = np.random.default_rng(seed)
    background_plot = background
    signal_plot = signal

    if max_points is not None:
        max_points = int(max_points)
        if max_points < 1:
            raise ValueError("max_points must be positive when provided.")
        if len(background_plot) > max_points:
            indices = rng.choice(len(background_plot), max_points, replace=False)
            background_plot = background_plot[indices]
        if len(signal_plot) > max_points:
            indices = rng.choice(len(signal_plot), max_points, replace=False)
            signal_plot = signal_plot[indices]

    mean_background = background_plot.mean(axis=0)
    mean_signal = signal_plot.mean(axis=0)
    centered_background = background_plot - mean_background
    centered_signal = signal_plot - mean_signal

    cov_background = np.atleast_2d(
        np.cov(centered_background, rowvar=False)
    ).astype(np.float64, copy=False)
    cov_signal = np.atleast_2d(
        np.cov(centered_signal, rowvar=False)
    ).astype(np.float64, copy=False)
    pooled_cov = (
        (len(background_plot) - 1) * cov_background
        + (len(signal_plot) - 1) * cov_signal
    ) / (len(background_plot) + len(signal_plot) - 2)
    regularized_cov = pooled_cov + float(cov_eps) * np.eye(
        pooled_cov.shape[0], dtype=np.float64
    )

    mean_difference = mean_signal - mean_background
    lda_direction = np.linalg.pinv(regularized_cov) @ mean_difference
    lda_norm = np.linalg.norm(lda_direction)
    if not np.isfinite(lda_norm) or lda_norm <= np.finfo(np.float64).eps:
        raise RuntimeError(
            f"Degenerate Fisher direction for {background_label} vs {signal_label}."
        )
    lda_direction /= lda_norm

    combined = np.concatenate([background_plot, signal_plot], axis=0)
    center = combined.mean(axis=0, keepdims=True)
    centered = combined - center
    lda_scores = centered @ lda_direction
    residuals = centered - lda_scores[:, None] * lda_direction[None, :]

    _, residual_singular_values, residual_vh = np.linalg.svd(
        residuals, full_matrices=False
    )
    residual_pc1 = residual_vh[0]
    residual_pc1 -= np.dot(residual_pc1, lda_direction) * lda_direction
    residual_pc1_norm = np.linalg.norm(residual_pc1)
    if residual_pc1_norm <= np.finfo(np.float64).eps:
        raise RuntimeError(
            f"Degenerate residual PCA direction for {background_label} vs {signal_label}."
        )
    residual_pc1 /= residual_pc1_norm
    residual_pc1_scores = residuals @ residual_pc1

    residual_total_variance = np.square(residual_singular_values).sum()
    residual_explained_ratio = (
        float(np.square(residual_singular_values[0]) / residual_total_variance)
        if residual_total_variance > 0.0
        else 0.0
    )

    num_background = len(background_plot)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(
        lda_scores[:num_background],
        residual_pc1_scores[:num_background],
        s=10,
        alpha=0.45,
        marker="o",
        label=(
            f"{background_label} (Background)"
            if annotate_roles
            else background_label
        ),
    )
    ax.scatter(
        lda_scores[num_background:],
        residual_pc1_scores[num_background:],
        s=18,
        alpha=0.65,
        marker="x",
        label=(
            f"{signal_label} (Signal)"
            if annotate_roles
            else signal_label
        ),
    )
    ax.set_xlabel("Fisher LDA1")
    ax.set_ylabel(
        f"Residual PC1 ({100.0 * residual_explained_ratio:.1f}% residual variance)"
    )
    ax.set_title(
        f"Pairwise validation LDA latent space: {background_label} vs {signal_label}"
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, frameon=True, markerscale=1.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_grouped_latent_space(
    groups: Sequence[Tuple[str, np.ndarray, str]],
    path: Path,
    title: str,
    seed: int,
    max_points: int | None = None,
) -> None:
    """Plot multiple named latent groups in one jointly fitted PCA basis."""

    rng = np.random.default_rng(seed)
    prepared: List[Tuple[str, np.ndarray, str]] = []
    for name, values, marker in groups:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"Expected latent group {name!r} with shape (N, D).")
        if len(values) == 0:
            continue
        if max_points is not None and len(values) > int(max_points):
            indices = rng.choice(len(values), int(max_points), replace=False)
            values = values[indices]
        prepared.append((name, values, marker))

    if len(prepared) < 2:
        raise RuntimeError(
            f"At least two non-empty groups are required for {path.name}."
        )

    latent_dims = {values.shape[1] for _, values, _ in prepared}
    if len(latent_dims) != 1:
        raise ValueError(f"Latent dimensions disagree across groups: {latent_dims}.")

    combined = np.concatenate([values for _, values, _ in prepared], axis=0)
    centered = combined - combined.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    reduced = centered @ vh[:2].T
    total_variance = np.square(singular_values).sum()
    explained = (
        np.square(singular_values[:2]) / total_variance
        if total_variance > 0.0
        else np.zeros(2, dtype=np.float64)
    )

    fig, ax = plt.subplots(figsize=(9, 7.5))
    offset = 0
    for name, values, marker in prepared:
        stop = offset + len(values)
        ax.scatter(
            reduced[offset:stop, 0],
            reduced[offset:stop, 1],
            s=12 if marker == "o" else 18,
            alpha=0.5 if marker == "o" else 0.65,
            marker=marker,
            label=name,
        )
        offset = stop

    ax.set_xlabel(f"PC1 ({100.0 * explained[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({100.0 * explained[1]:.1f}% variance)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, frameon=True, markerscale=1.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_score_distribution(
    curves: Sequence[Tuple[str, np.ndarray, np.ndarray]],
    title: str, xlabel: str, path: Path, signal_label: str, bins: int = 80,
) -> None:
    """One panel per background, common bins within each BG/signal pair."""
    columns = min(3, len(curves))
    rows = (len(curves) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.8 * rows), squeeze=False)
    for ax, (name, background, signal) in zip(axes.flat, curves):
        combined = np.concatenate([background, signal])
        edges = np.histogram_bin_edges(combined, bins=bins)
        ax.hist(background, bins=edges, density=True, color="tab:blue", alpha=0.55,
                label=f"{name} (background)")
        ax.hist(signal, bins=edges, density=True, color="tab:orange", alpha=0.55,
                label=f"{signal_label} (signal)")
        ax.set(title=name, xlabel=xlabel, ylabel="Density")
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(curves):]:
        ax.set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_comparison(
    curves: Sequence[Tuple[str, np.ndarray, np.ndarray]], title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for name, background, signal in curves:
        y = np.concatenate(
            [
                np.zeros(len(background), dtype=np.int64),
                np.ones(len(signal), dtype=np.int64),
            ]
        )
        scores = np.concatenate([background, signal])
        fpr, tpr, _ = roc_curve(y, scores)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, scores):.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False Positive Rate", ylabel="True Positive Rate")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main() -> None:
    args = parse_args()
    for name in ("knn_k", "score_batch_size", "knn_reference_batch_size", "gmm_n_components",
                 "gmm_max_iter", "gmm_n_init", "score_hist_bins"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if not np.isfinite(args.gmm_reg_covar) or args.gmm_reg_covar <= 0:
        raise ValueError("--gmm-reg-covar must be finite and positive.")
    if not np.isfinite(args.gmm_tol) or args.gmm_tol < 0:
        raise ValueError("--gmm-tol must be finite and non-negative.")
    run_dir = args.run_dir.expanduser().resolve()
    summary_path = run_dir / "summary.json"
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint
        else run_dir / "best_model.pth"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "latent_diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with summary_path.open() as handle:
        summary = json.load(handle)

    device = resolve_device(args.device)
    seed = int(summary.get("base_seed", summary.get("seed", 42)))
    seed_everything(seed)

    backend = DiagnosticDatasetBackend.from_summary(
        summary=summary,
        run_dir=run_dir,
        dataset_override=args.dataset,
        dataset_root_override=args.dataset_root,
        cms_manifest_override=args.cms_split_manifest,
        max_num_particles_override=args.max_num_particles,
    )

    backgrounds = list(dict.fromkeys(summary["background_labels"]))
    signals = list(dict.fromkeys(summary.get("signal_labels", [])))
    backend.validate_requested_labels(list(dict.fromkeys(backgrounds + signals)))
    if not backgrounds:
        raise ValueError("At least one background label is required.")
    signal_was_configured = bool(signals)
    signal_rotation = signals if signal_was_configured else list(backgrounds)

    def display_label(label: str) -> str:
        return label.removeprefix("label_")

    background_display_name = "+".join(display_label(x) for x in backgrounds)
    signal_display_name = "+".join(display_label(x) for x in signals)
    steps = int(args.eval_steps or summary.get("eval_steps", 50))
    batch_size = int(
        args.batch_size
        or summary.get("per_rank_batch_size", summary.get("batch_size", 128))
    )
    num_workers = int(
        args.num_workers
        if args.num_workers is not None
        else summary.get("num_workers", 0)
    )
    if num_workers < 0:
        raise ValueError("--num-workers must be non-negative.")
    cov_eps = float(
        args.mahalanobis_cov_eps
        if args.mahalanobis_cov_eps is not None
        else summary.get("mahalanobis_cov_eps", 1e-4)
    )
    if not np.isfinite(cov_eps) or cov_eps <= 0:
        raise ValueError("Mahalanobis covariance regularization must be finite and positive.")
    precision = str(summary.get("precision", "fp32"))
    max_pair_points = int(summary.get("max_latent_plot_points", 5000))

    print(f"Dataset backend: {backend.dataset_name}")
    print(f"Dataset root: {backend.dataset_root}")
    print(f"Loading {checkpoint_path} on {device}")
    print(f"Sampling {steps} x {batch_size} events from each source stream")
    print("Full-jet density latent space: representation")
    print(f"Training labels: {background_display_name}")
    if signal_was_configured:
        print(f"Visualization signal labels: {signal_display_name}")
    else:
        print("Visualization signal labels: none; rotating every training type")
    print(f"Dataset label axis: {backend.label_axis}")
    print("Particle feature order:")
    for index, name in enumerate(backend.feature_names):
        print(f"  {index:2d}: {name}")
    print(
        "Batch-standardized particle features: "
        f"{backend.batch_standardized_feature_names}"
    )
    if backend.dataset_name == "cms":
        print(
            "CMS split source: "
            f"{backend.cms_manifest_source}; sha256={backend.cms_manifest_sha256}"
        )

    model = build_model(summary, backend, device)
    state_dict = read_state_dict(checkpoint_path, device)
    load_result = model.load_state_dict(state_dict, strict=False)

    feature_stat_suffixes = (
        "_feature_running_mean",
        "_feature_running_var",
        "_feature_num_batches_tracked",
    )
    checkpoint_stat_suffixes = {
        suffix
        for suffix in feature_stat_suffixes
        if any(key.endswith(suffix) for key in state_dict)
    }
    missing_feature_stats = len(checkpoint_stat_suffixes) != len(feature_stat_suffixes)
    unexpected_missing = [
        key
        for key in load_result.missing_keys
        if not key.endswith(feature_stat_suffixes)
    ]
    if unexpected_missing:
        raise RuntimeError(
            "Checkpoint is missing unexpected model parameters: "
            f"{unexpected_missing}"
        )
    if load_result.unexpected_keys:
        raise RuntimeError(
            "Checkpoint contains unexpected model parameters: "
            f"{load_result.unexpected_keys}"
        )
    if missing_feature_stats:
        model._use_frozen_feature_stats_in_eval = False
        warnings.warn(
            "Legacy checkpoint detected: one or more feature running-stat "
            "buffers are absent. Evaluation will use per-batch feature stats.",
            RuntimeWarning,
        )
    else:
        model._use_frozen_feature_stats_in_eval = True
    model.eval()

    loader_common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    print("\nCollecting validation representations once per source...")
    val_z, val_y = collect_full_latents(
        model=model,
        loader=backend.make_loader(split_name="val", labels=backgrounds, seed=seed + 202, **loader_common),
        steps=steps, device=device, precision=precision,
        description="Full latent: background validation",
    )
    if signal_was_configured:
        val_signal_z, val_signal_y = collect_full_latents(
            model=model,
            loader=backend.make_loader(split_name="val", labels=signals, seed=seed + 404, **loader_common),
            steps=steps, device=device, precision=precision,
            description="Full latent: signal validation",
        )
    else:
        val_signal_z = np.empty((0, val_z.shape[1]), dtype=val_z.dtype)
        val_signal_y = np.empty((0, len(backend.label_axis)), dtype=val_y.dtype)
    if len(val_z) == 0:
        raise RuntimeError("Collected empty background validation sample.")
    for label in backgrounds:
        if class_mask(val_y, label, backend.label_axis).sum() < 2:
            raise ValueError(f"Need at least two validation events for background {label}.")
    for label in signals:
        if not class_mask(val_signal_y, label, backend.label_axis).any():
            raise ValueError(f"No validation events for signal {label}.")
    print(f"Collected validation latents: background={len(val_z)}, signal={len(val_signal_z)}")

    def all_type_groups(
        base_z: np.ndarray,
        base_y: np.ndarray,
        base_role: str,
        signal_z: np.ndarray,
        signal_y: np.ndarray,
    ) -> List[Tuple[str, np.ndarray, str]]:
        groups: List[Tuple[str, np.ndarray, str]] = []
        for label in backgrounds:
            values = base_z[class_mask(base_y, label, backend.label_axis)]
            name = display_label(label)
            if signal_was_configured:
                name += f" ({base_role})"
            groups.append((name, values, "o"))
        if signal_was_configured:
            for label in signals:
                values = signal_z[class_mask(signal_y, label, backend.label_axis)]
                groups.append((f"{display_label(label)} (Signal Validation)", values, "x"))
        return groups

    plot_grouped_latent_space(
        groups=all_type_groups(
            val_z, val_y, "Background Validation", val_signal_z, val_signal_y
        ),
        path=output_dir / "02_all_type_validation_pca.png",
        title=(
            "All-type validation latent space with signal validation"
            if signal_was_configured
            else "All-type validation latent space"
        ),
        seed=seed + 440,
        max_points=max_pair_points,
    )

    if signal_was_configured:
        pair_specs = [
            (background, signal)
            for signal in signals
            for background in backgrounds
            if background != signal
        ]
    else:
        pair_specs = [
            (backgrounds[i], backgrounds[j])
            for i in range(len(backgrounds))
            for j in range(i + 1, len(backgrounds))
        ]

    for pair_index, (first_label, second_label) in enumerate(pair_specs):
        first_name = display_label(first_label)
        second_name = display_label(second_label)
        first_z = val_z[class_mask(val_y, first_label, backend.label_axis)]
        if signal_was_configured:
            second_z = val_signal_z[
                class_mask(val_signal_y, second_label, backend.label_axis)
            ]
        else:
            second_z = val_z[class_mask(val_y, second_label, backend.label_axis)]

        plot_pair_latent_space(
            background=first_z,
            signal=second_z,
            background_label=first_name,
            signal_label=second_name,
            path=(
                output_dir
                / f"03_pairwise_pca_{first_name.lower()}_vs_{second_name.lower()}.png"
            ),
            seed=seed + 450 + pair_index,
            max_points=max_pair_points,
            annotate_roles=signal_was_configured,
        )
        plot_pair_lda_residual_pca_space(
            background=first_z,
            signal=second_z,
            background_label=first_name,
            signal_label=second_name,
            path=(
                output_dir
                / f"04_pairwise_lda_residual_pca_{first_name.lower()}_vs_"
                f"{second_name.lower()}.png"
            ),
            seed=seed + 550 + pair_index,
            cov_eps=cov_eps,
            max_points=max_pair_points,
            annotate_roles=signal_was_configured,
        )

    # All scores consume these same cached representations; no encoder calls below.
    score_dtype = torch.float64 if device.type != "mps" else torch.float32
    background_tensor = torch.as_tensor(val_z, dtype=score_dtype, device=device)
    signal_tensor = torch.as_tensor(val_signal_z, dtype=score_dtype, device=device)
    ids = label_ids(val_y, backend.label_axis)
    global_results = {}
    pairwise_results = {}
    global_cache = {}
    class_cache = {}
    percentile_model_cache = {}
    raw_model_cache = {}
    raw_pairwise_results = {}

    def emit_pair(curves, title, stem, xlabel, signal_name):
        plot_comparison(curves, title, output_dir / f"{stem}_roc.png")
        plot_score_distribution(curves, title.replace("ROC", "score distributions"), xlabel,
                                output_dir / f"{stem}_score_distribution.png",
                                signal_name, args.score_hist_bins)

    def result_stats(background_score, signal_score):
        return {"auc_validation_vs_signal": auc(background_score, signal_score),
                "validation_background_scores": score_stats(background_score),
                "validation_signal_scores": score_stats(signal_score)}

    for signal_label in signal_rotation:
        signal_name = display_label(signal_label)
        active_backgrounds = [label for label in backgrounds if label != signal_label]
        if not active_backgrounds:
            raise ValueError("Signal rotation requires at least one other background class.")
        selected = np.isin(ids, [backend.label_axis.index(label) for label in active_backgrounds])
        reference = background_tensor[torch.as_tensor(selected, device=device)]
        reference_labels = torch.as_tensor(ids[selected], device=device)
        if signal_was_configured:
            signal_mask = class_mask(val_signal_y, signal_label, backend.label_axis)
            query = signal_tensor[torch.as_tensor(signal_mask, device=device)]
        else:
            query = background_tensor[torch.as_tensor(class_mask(val_y, signal_label, backend.label_axis), device=device)]
        if len(query) == 0:
            raise ValueError(f"No validation events for signal {signal_label}.")
        cache_key = tuple(active_backgrounds)
        if cache_key not in global_cache:
            print(f"\nFitting validation background scores: {', '.join(active_backgrounds)}")
            mean, matrix, covariance = fit_mahalanobis(reference, cov_eps)
            knn_background = knn_scores(reference, reference, args.knn_k, args.score_batch_size,
                                        args.knn_reference_batch_size, args.knn_reduction, exclude_self=True)
            gmm_kwargs = dict(n_components=args.gmm_n_components, covariance_type=args.gmm_covariance_type,
                              reg_covar=args.gmm_reg_covar, max_iter=args.gmm_max_iter, tol=args.gmm_tol,
                              n_init=args.gmm_n_init, batch_size=args.score_batch_size, seed=seed)
            unlabeled = TorchGaussianMixture(**gmm_kwargs).fit(reference)
            labeled = TorchGaussianMixture(**gmm_kwargs).fit(reference, reference_labels)
            global_cache[cache_key] = (reference, mean, matrix, covariance, unlabeled, labeled, {
                "combined_mahalanobis": mahalanobis_scores(reference, mean, matrix, args.score_batch_size),
                "knn": knn_background, "gmm": unlabeled.scores(reference),
                "labeled_gmm": labeled.scores(reference),
            })
        reference, mean, matrix, covariance, unlabeled, labeled, bg_scores = global_cache[cache_key]
        signal_scores = {
            "combined_mahalanobis": mahalanobis_scores(query, mean, matrix, args.score_batch_size),
            "knn": knn_scores(query, reference, args.knn_k, args.score_batch_size,
                              args.knn_reference_batch_size, args.knn_reduction),
            "gmm": unlabeled.scores(query), "labeled_gmm": labeled.scores(query),
        }
        metadata = {"combined_mahalanobis": covariance,
                    "knn": {"k": args.knn_k, "reduction": args.knn_reduction, "exclude_background_self": True},
                    "gmm": unlabeled.diagnostics(), "labeled_gmm": labeled.diagnostics()}
        metadata["labeled_gmm"]["component_labels"] = [backend.label_axis[int(i)] for i in torch.unique(reference_labels).cpu()]
        per_global = {}
        for name, scores in signal_scores.items():
            curves = [("All background", bg_scores[name], scores)]
            xlabel = "Negative log density" if "gmm" in name else ("Euclidean kNN distance" if name == "knn" else "Squared Mahalanobis distance")
            emit_pair(curves, f"{name} ROC vs {signal_name}",
                      f"05_{name}_signal_{signal_name.lower()}", xlabel, signal_name)
            per_global[name] = {**result_stats(bg_scores[name], scores), "fit": metadata[name]}
            print(f"signal={signal_name}, {name}: AUC={per_global[name]['auc_validation_vs_signal']:.6f}")
        global_results[signal_label] = per_global

        curves = []
        per_background = {}
        joint_name = "joint_class_mahalanobis"
        calibrate_background = joint_name not in bg_scores
        percentile_name = "joint_class_percentile_mahalanobis"
        fit_percentile_models = cache_key not in percentile_model_cache
        raw_name = "joint_class_raw_mahalanobis"
        fit_raw_models = cache_key not in raw_model_cache
        background_raw_columns = []
        signal_raw_columns = []
        background_percentile_columns = []
        signal_percentile_columns = []
        max_background_compatibility = np.zeros(len(reference), dtype=np.float64)
        max_signal_compatibility = np.zeros(len(query), dtype=np.float64)
        for background_label in active_backgrounds:
            if background_label not in class_cache:
                class_values = background_tensor[torch.as_tensor(class_mask(val_y, background_label, backend.label_axis), device=device)]
                class_mean, class_precision, class_cov = fit_mahalanobis(class_values, cov_eps)
                class_scores = mahalanobis_scores(class_values, class_mean, class_precision, args.score_batch_size)
                class_cache[background_label] = class_mean, class_precision, class_cov, class_scores
            class_mean, class_precision, class_cov, class_scores = class_cache[background_label]
            signal_scores = mahalanobis_scores(query, class_mean, class_precision, args.score_batch_size)
            signal_raw_columns.append(signal_scores)
            signal_percentile_columns.append(mahalanobis_percentiles(class_scores, signal_scores))
            max_signal_compatibility = np.maximum(
                max_signal_compatibility,
                mahalanobis_compatibility(class_scores, signal_scores),
            )
            if calibrate_background or fit_percentile_models or fit_raw_models:
                # Every background jet is compared to every active class,
                # regardless of its own label. Calibration uses only that class.
                background_distances = mahalanobis_scores(
                    reference, class_mean, class_precision, args.score_batch_size
                )
                background_raw_columns.append(background_distances)
                background_percentile_columns.append(
                    mahalanobis_percentiles(class_scores, background_distances)
                )
                max_background_compatibility = np.maximum(
                    max_background_compatibility,
                    mahalanobis_compatibility(class_scores, background_distances),
                )
            curves.append((display_label(background_label), class_scores, signal_scores))
            per_background[background_label] = {**result_stats(class_scores, signal_scores), "covariance": class_cov}
        emit_pair(curves, f"Per-class Mahalanobis ROC vs {signal_name}",
                  f"08_multiclass_validation_signal_{signal_name.lower()}", "Squared Mahalanobis distance", signal_name)
        pairwise_results[signal_label] = per_background

        if calibrate_background:
            bg_scores[joint_name] = 1.0 - max_background_compatibility
        joint_signal_scores = 1.0 - max_signal_compatibility
        emit_pair(
            [("All background", bg_scores[joint_name], joint_signal_scores)],
            f"Joint-class Mahalanobis ROC vs {signal_name}",
            f"05_{joint_name}_signal_{signal_name.lower()}",
            "Joint-class Mahalanobis anomaly score", signal_name,
        )
        per_global[joint_name] = {
            **result_stats(bg_scores[joint_name], joint_signal_scores),
            "fit": {
                "component_labels": active_backgrounds,
                "percentile": "(count_less + 0.5 * count_equal) / class_count",
                "compatibility": "p_k = 2 * min(q_k, 1 - q_k)",
                "anomaly_score": "1 - max_k(p_k)",
                "calibration": "own-class background validation distances",
                "num_calibration_events_by_class": {
                    label: len(class_cache[label][3]) for label in active_backgrounds
                },
            },
        }
        print(f"signal={signal_name}, {joint_name}: AUC={per_global[joint_name]['auc_validation_vs_signal']:.6f}")

        # Second layer: model the full K-dimensional percentile vector within
        # each background class, preserving correlations between its entries.
        signal_percentiles = torch.as_tensor(
            np.column_stack(signal_percentile_columns), dtype=score_dtype, device=device
        )
        if fit_percentile_models:
            background_percentiles = torch.as_tensor(
                np.column_stack(background_percentile_columns), dtype=score_dtype, device=device
            )
            percentile_models = {}
            max_compatibility = np.zeros(len(reference), dtype=np.float64)
            for label in active_backgrounds:
                own_class = reference_labels == backend.label_axis.index(label)
                own_percentiles = background_percentiles[own_class]
                p_mean, p_precision, p_cov = fit_mahalanobis(own_percentiles, cov_eps)
                calibration = mahalanobis_scores(
                    own_percentiles, p_mean, p_precision, args.score_batch_size
                )
                distances = mahalanobis_scores(
                    background_percentiles, p_mean, p_precision, args.score_batch_size
                )
                max_compatibility = np.maximum(
                    max_compatibility, mahalanobis_compatibility(calibration, distances)
                )
                percentile_models[label] = p_mean, p_precision, p_cov, calibration
            percentile_model_cache[cache_key] = percentile_models
            bg_scores[percentile_name] = 1.0 - max_compatibility
        percentile_models = percentile_model_cache[cache_key]
        max_compatibility = np.zeros(len(query), dtype=np.float64)
        for label in active_backgrounds:
            p_mean, p_precision, _, calibration = percentile_models[label]
            distances = mahalanobis_scores(
                signal_percentiles, p_mean, p_precision, args.score_batch_size
            )
            max_compatibility = np.maximum(
                max_compatibility, mahalanobis_compatibility(calibration, distances)
            )
        percentile_signal_scores = 1.0 - max_compatibility
        emit_pair(
            [("All background", bg_scores[percentile_name], percentile_signal_scores)],
            f"Joint-class percentile Mahalanobis ROC vs {signal_name}",
            f"05_{percentile_name}_signal_{signal_name.lower()}",
            "Joint-class percentile Mahalanobis anomaly score", signal_name,
        )
        per_global[percentile_name] = {
            **result_stats(bg_scores[percentile_name], percentile_signal_scores),
            "fit": {
                "percentile_vector_labels": active_backgrounds,
                "percentile": "(count_less + 0.5 * count_equal) / class_count",
                "first_layer": "own-class latent Mahalanobis distance percentiles",
                "second_layer": "per-class Mahalanobis of full percentile vectors",
                "calibration": "own-class background validation distances at both layers",
                "compatibility": "p_k = 2 * min(q_k, 1 - q_k)",
                "anomaly_score": "1 - max_k(p_k)",
                "covariance_by_class": {
                    label: percentile_models[label][2] for label in active_backgrounds
                },
            },
        }
        print(f"signal={signal_name}, {percentile_name}: AUC={per_global[percentile_name]['auc_validation_vs_signal']:.6f}")

        # Raw variant: retain squared distances as vector entries, without
        # percentile transforms. Fit each class on its own full score vectors.
        signal_raw_vectors = torch.as_tensor(
            np.column_stack(signal_raw_columns), dtype=score_dtype, device=device
        )
        if fit_raw_models:
            background_raw_vectors = torch.as_tensor(
                np.column_stack(background_raw_columns), dtype=score_dtype, device=device
            )
            raw_models = {}
            max_compatibility = np.zeros(len(reference), dtype=np.float64)
            for label in active_backgrounds:
                own_class = reference_labels == backend.label_axis.index(label)
                own_vectors = background_raw_vectors[own_class]
                raw_mean, raw_precision, raw_cov = fit_mahalanobis(own_vectors, cov_eps)
                calibration = mahalanobis_scores(
                    own_vectors, raw_mean, raw_precision, args.score_batch_size
                )
                distances = mahalanobis_scores(
                    background_raw_vectors, raw_mean, raw_precision, args.score_batch_size
                )
                max_compatibility = np.maximum(
                    max_compatibility, mahalanobis_compatibility(calibration, distances)
                )
                raw_models[label] = raw_mean, raw_precision, raw_cov, calibration
            raw_model_cache[cache_key] = raw_models
            bg_scores[raw_name] = 1.0 - max_compatibility
        raw_models = raw_model_cache[cache_key]
        raw_curves = []
        raw_per_background = {}
        max_compatibility = np.zeros(len(query), dtype=np.float64)
        for label in active_backgrounds:
            raw_mean, raw_precision, raw_cov, calibration = raw_models[label]
            distances = mahalanobis_scores(
                signal_raw_vectors, raw_mean, raw_precision, args.score_batch_size
            )
            # Pairwise diagnostics use only this class's background scores;
            # the global score above evaluates all backgrounds against all classes.
            raw_curves.append((display_label(label), calibration, distances))
            raw_per_background[label] = {
                **result_stats(calibration, distances), "covariance": raw_cov,
                "score_vector_labels": active_backgrounds,
            }
            max_compatibility = np.maximum(
                max_compatibility, mahalanobis_compatibility(calibration, distances)
            )
        emit_pair(
            raw_curves, f"Per-class raw-score-vector Mahalanobis ROC vs {signal_name}",
            f"08_multiclass_raw_mahalanobis_signal_{signal_name.lower()}",
            "Squared Mahalanobis distance in raw score-vector space", signal_name,
        )
        raw_pairwise_results[signal_label] = raw_per_background
        raw_signal_scores = 1.0 - max_compatibility
        emit_pair(
            [("All background", bg_scores[raw_name], raw_signal_scores)],
            f"Joint-class raw Mahalanobis ROC vs {signal_name}",
            f"05_{raw_name}_signal_{signal_name.lower()}",
            "Joint-class raw Mahalanobis anomaly score", signal_name,
        )
        per_global[raw_name] = {
            **result_stats(bg_scores[raw_name], raw_signal_scores),
            "fit": {
                "score_vector_labels": active_backgrounds,
                "first_layer": "raw squared latent Mahalanobis distances to all active classes",
                "second_layer": "per-class Mahalanobis of full raw score vectors",
                "calibration": "own-class background validation second-layer distances",
                "percentile": "(count_less + 0.5 * count_equal) / class_count",
                "compatibility": "p_k = 2 * min(q_k, 1 - q_k)",
                "anomaly_score": "1 - max_k(p_k)",
                "covariance_by_class": {
                    label: raw_models[label][2] for label in active_backgrounds
                },
            },
        }
        print(f"signal={signal_name}, {raw_name}: AUC={per_global[raw_name]['auc_validation_vs_signal']:.6f}")

    results = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "dataset": backend.dataset_name,
        "dataset_root": str(backend.dataset_root),
        "dataset_label_axis": backend.label_axis,
        "particle_features": backend.feature_names,
        "batch_standardized_particle_features": (
            backend.batch_standardized_feature_names
        ),
        "cms_split_manifest_source": backend.cms_manifest_source,
        "cms_split_manifest_sha256": backend.cms_manifest_sha256,
        "device": str(device),
        "sampling": {
            "eval_steps": steps,
            "batch_size": batch_size,
            "events_per_source_stream": steps * batch_size,
            "num_workers": num_workers,
            "split": "val",
            "background_fit_and_evaluation_use_same_sample": True,
            "seed": seed,
            "max_num_particles": backend.max_num_particles,
        },
        "labels": {
            "background": backgrounds,
            "signal": signals,
            "effective_signal_rotation": signal_rotation,
        },
        "signal_was_configured": signal_was_configured,
        "sample_counts": {
            "background_validation_total": len(val_z),
            "signal_validation_total": len(val_signal_z),
        },
        "score_parameters": {key: value for key, value in vars(args).items()
                             if key.startswith(("knn_", "gmm_", "score_"))},
        "global_scores_by_signal": global_results,
        "combined_mahalanobis": {label: values["combined_mahalanobis"]
                                 for label, values in global_results.items()},
        "per_class": (
            pairwise_results[signals[0]]
            if len(signals) == 1
            else pairwise_results
        ),
        "pairwise_mahalanobis_by_signal": pairwise_results,
        "pairwise_raw_mahalanobis_by_signal": raw_pairwise_results,
    }
    with (output_dir / "diagnostic_results.json").open("w") as handle:
        json.dump(safe_json(results), handle, indent=2)

    print(f"\nSaved diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
