#!/usr/bin/env python3
"""Post-training latent diagnostics for JetClass or CMS LeJEPA runs.

The dataset backend, particle feature order, batch-standardized feature list,
label axis, and CMS class × production-family split are reconstructed from the
training ``summary.json``.  A command-line dataset override is available for
explicit recovery workflows, but normal use requires only the run directory.

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

from helpers import cms_streaming
from helpers import jetclass_streaming
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

JETCLASS_DEFAULT_BATCH_STANDARDIZED_FEATURES = (
    "log_pt_fraction",
    "d0_sig",
    "dz_sig",
)

CMS_DEFAULT_BATCH_STANDARDIZED_FEATURES = (
    "log_pt_fraction",
    "Cpfcan_dxysig",
    "log_Cpfcan_dxysig",
    "Cpfcan_dz",
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
    cms_splits: Optional[Dict[str, Dict[str, Dict[str, List[str]]]]] = None
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

        features = _first_summary_list(
            summary,
            ("particle_features", "feature_names"),
        )
        if features is None:
            module = jetclass_streaming if dataset_name == "jetclass" else cms_streaming
            for attr in (
                "DEFAULT_PARTICLE_FEATURES",
                "CMS_PARTICLE_FEATURES",
                "CANONICAL_PARTICLE_FEATURES",
            ):
                value = getattr(module, attr, None)
                if value is not None:
                    features = list(value)
                    break
        if not features:
            raise KeyError(
                "Could not resolve the model particle feature list from summary.json "
                "or the selected dataset module."
            )
        _validate_four_vector_prefix(features, "Current dataset feature list")

        standardized = _first_summary_list(
            summary,
            (
                "batch_standardized_particle_features",
                "batch_normalized_particle_features",
                "standardized_particle_features",
            ),
        )
        if standardized is None:
            standardized = list(
                JETCLASS_DEFAULT_BATCH_STANDARDIZED_FEATURES
                if dataset_name == "jetclass"
                else CMS_DEFAULT_BATCH_STANDARDIZED_FEATURES
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
            for split_name, directory_name in (
                ("train", "train_100M"),
                ("val", "val_5M"),
                ("test", "test_20M"),
            ):
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
            splits = manifest["splits"]
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
            self.cms_splits = self._resolve_manifest_paths(splits)
        else:
            requested_labels = list(dict.fromkeys(
                list(self.summary["background_labels"])
                + list(self.summary["signal_labels"])
            ))
            discovered = cms_streaming.discover_cms_files_by_label_family(
                str(self.dataset_root),
                requested_labels,
            )
            self.cms_splits = cms_streaming.split_cms_files_by_family(
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
                    "Reconstructed CMS split does not match the split used by training. "
                    f"summary={expected_hash}, reconstructed={self.cms_manifest_sha256}. "
                    "Restore cms_split_manifest.json from the run directory."
                )

        assert self.cms_splits is not None
        for split_name in ("train", "val", "test"):
            if split_name not in self.cms_splits:
                raise ValueError(f"CMS split mapping is missing {split_name!r}.")

    def _resolve_manifest_paths(
        self,
        splits: Mapping[str, Mapping[str, Mapping[str, Sequence[str]]]],
    ) -> Dict[str, Dict[str, Dict[str, List[str]]]]:
        old_root_value = self.summary.get("dataset_root")
        old_root = (
            Path(str(old_root_value)).expanduser()
            if old_root_value is not None
            else None
        )
        resolved: Dict[str, Dict[str, Dict[str, List[str]]]] = {}
        missing: List[str] = []
        for split_name, labels in splits.items():
            resolved[split_name] = {}
            for label, families in labels.items():
                resolved[split_name][label] = {}
                for family, paths in families.items():
                    resolved_paths: List[str] = []
                    for raw_path in paths:
                        path = Path(str(raw_path)).expanduser()
                        if path.is_file():
                            candidate = path.resolve()
                        elif old_root is not None:
                            try:
                                relative = path.relative_to(old_root)
                            except ValueError:
                                directory_map = getattr(cms_streaming, "CMS_LABEL_TO_DIRECTORY", {})
                                directory_name = directory_map.get(
                                    label, label.removeprefix("label_").lower()
                                )
                                relative = Path(directory_name) / path.name
                            candidate = (self.dataset_root / relative).resolve()
                        else:
                            candidate = (self.dataset_root / path.name).resolve()
                        if not candidate.is_file():
                            missing.append(str(candidate))
                        resolved_paths.append(str(candidate))
                    resolved[split_name][label][family] = resolved_paths
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
                files_by_label_family=self.cms_splits[split_name],
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
                min_active_families_per_class=int(
                    self.summary.get("cms_min_active_families_per_class", 2)
                ),
                family_sampling=str(
                    self.summary.get("cms_family_sampling", "proportional")
                ),
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
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument(
        "--num-gaussians",
        type=int,
        default=6,
        help="Reserved for the extended multi-Gaussian diagnostic. Default: 6.",
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=30,
        help="Reserved for the extended kNN diagnostic. Default: 30.",
    )
    parser.add_argument("--mahalanobis-cov-eps", type=float, default=None)
    parser.add_argument(
        "--max-num-particles",
        type=int,
        default=None,
        help="Override summary.json; otherwise uses the training value.",
    )
    parser.add_argument(
        "--full-latent-space",
        choices=["representation", "cls"],
        default="representation",
        help=(
            "Which unaugmented latent to use for density scores. "
            "'representation' applies representation_head to the CLS state."
        ),
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
            ("dz_sig", "Cpfcan_dz"),
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
    latent_space: str = "representation",
) -> torch.Tensor:
    cls = model(x, padding_mask=padding_mask)
    if latent_space == "cls":
        return cls
    if latent_space != "representation":
        raise ValueError(f"Unknown latent_space={latent_space!r}.")
    if hasattr(model, "representation_head"):
        return model.representation_head(cls)
    return cls


@torch.no_grad()
def collect_full_latents(
    model: torch.nn.Module,
    loader: DataLoader,
    steps: int,
    device: torch.device,
    precision: str,
    description: str,
    latent_space: str = "representation",
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect unaugmented latents from one loader.

    This helper remains for diagnostics that intentionally use a single source.
    The main Mahalanobis/per-class path uses
    ``collect_interleaved_full_latents`` below so that background and signal
    events share exactly the same forward-pass normalization context.
    """
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
                latent_space=latent_space,
            )
        zs.append(z.detach().float().cpu())
        ys.append(batch["y"].float())
    return torch.cat(zs).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def collect_interleaved_full_latents(
    model: torch.nn.Module,
    background_loader: DataLoader,
    signal_loader: DataLoader,
    steps: int,
    device: torch.device,
    precision: str,
    description: str,
    background_labels: Sequence[str],
    signal_labels: Sequence[str],
    combined_batch_size: int,
    seed: int,
    latent_space: str = "representation",
) -> Tuple[np.ndarray, np.ndarray]:
    """Forward background and signal events together, then return labels.

    This interleaved collection path is retained for compatibility with older
    model versions whose evaluation-time feature standardization depended on
    the current batch composition. The current model uses frozen normalization
    statistics in ``eval`` mode, so interleaving background and signal events is
    no longer required for correctness. It is kept here to preserve the sampling
    and evaluation behavior of earlier diagnostic runs.

    The combined batch budget is split in proportion to the number of requested
    background and signal jet classes. Because both source datasets already
    emit class-balanced streams, this gives approximately equal representation
    to every configured jet type inside each forward pass.
    """
    if combined_batch_size < 2:
        raise ValueError("combined_batch_size must be at least 2.")
    if not background_labels or not signal_labels:
        raise ValueError("Both background_labels and signal_labels are required.")

    num_background_classes = len(background_labels)
    num_signal_classes = len(signal_labels)
    num_classes = num_background_classes + num_signal_classes

    num_background = int(
        round(combined_batch_size * num_background_classes / num_classes)
    )
    num_background = min(max(num_background, 1), combined_batch_size - 1)
    num_signal = combined_batch_size - num_background

    model.eval()
    background_iterator = iter(background_loader)
    signal_iterator = iter(signal_loader)
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(int(seed))

    zs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []

    for _ in tqdm(range(steps), desc=description):
        background_batch = next(background_iterator)
        signal_batch = next(signal_iterator)

        background_available = int(background_batch["x_particles"].shape[0])
        signal_available = int(signal_batch["x_particles"].shape[0])
        if background_available < num_background or signal_available < num_signal:
            raise RuntimeError(
                "A source loader batch is too small for the requested mixed "
                "forward composition: "
                f"need background={num_background}, signal={num_signal}; "
                f"received background={background_available}, "
                f"signal={signal_available}."
            )

        x = torch.cat(
            [
                background_batch["x_particles"][:num_background],
                signal_batch["x_particles"][:num_signal],
            ],
            dim=0,
        )
        padding_mask = torch.cat(
            [
                background_batch["padding_mask"][:num_background],
                signal_batch["padding_mask"][:num_signal],
            ],
            dim=0,
        )
        y = torch.cat(
            [
                background_batch["y"][:num_background],
                signal_batch["y"][:num_signal],
            ],
            dim=0,
        ).float()

        permutation = torch.randperm(
            x.shape[0],
            generator=permutation_generator,
        )
        x = x[permutation].to(device, non_blocking=True)
        padding_mask = padding_mask[permutation].to(device, non_blocking=True)
        y = y[permutation]

        with autocast_context(device, precision):
            z = encode_single_view(
                model,
                x,
                padding_mask=padding_mask,
                latent_space=latent_space,
            )

        zs.append(z.detach().float().cpu())
        ys.append(y.cpu())

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


def stratified_split(
    y: np.ndarray,
    labels: Sequence[str],
    label_axis: Sequence[str],
    fit_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("--fit-fraction must lie strictly between 0 and 1.")
    ids = label_ids(y, label_axis)
    rng = np.random.default_rng(seed)
    fit, heldout = [], []
    for label in labels:
        indices = np.flatnonzero(ids == list(label_axis).index(label))
        if len(indices) < 4:
            raise RuntimeError(f"Only {len(indices)} sampled events found for {label}.")
        rng.shuffle(indices)
        cut = min(max(int(round(len(indices) * fit_fraction)), 2), len(indices) - 2)
        fit.append(indices[:cut])
        heldout.append(indices[cut:])
    fit_idx, heldout_idx = np.concatenate(fit), np.concatenate(heldout)
    rng.shuffle(fit_idx)
    rng.shuffle(heldout_idx)
    return fit_idx, heldout_idx

def fit_mahalanobis(
    latents: np.ndarray, cov_eps: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    x = np.asarray(latents, dtype=np.float64)
    mean = x.mean(axis=0)
    cov = np.asarray(np.cov(x - mean, rowvar=False), dtype=np.float64)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    reg_cov = cov + cov_eps * np.eye(cov.shape[0], dtype=np.float64)
    precision = np.linalg.pinv(reg_cov)

    eig = np.linalg.eigvalsh(cov)
    reg_eig = np.linalg.eigvalsh(reg_cov)
    positive = eig[eig > 0]
    return mean, precision, {
        "num_fit_events": int(len(x)),
        "latent_dim": int(x.shape[1]),
        "raw_min_eigenvalue": float(eig.min()),
        "raw_max_eigenvalue": float(eig.max()),
        "raw_condition_number_positive_spectrum": (
            float(eig.max() / positive.min()) if len(positive) else float("inf")
        ),
        "regularized_min_eigenvalue": float(reg_eig.min()),
        "regularized_max_eigenvalue": float(reg_eig.max()),
        "regularized_condition_number": float(reg_eig.max() / reg_eig.min()),
        "cov_eps": float(cov_eps),
    }


def mahalanobis_scores(
    latents: np.ndarray, mean: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    centered = np.asarray(latents, dtype=np.float64) - mean
    return np.einsum("ni,ij,nj->n", centered, precision, centered)


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
) -> None:
    """Plot one background class against the configured signal in one PCA basis."""

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
        label=f"{background_label} (Background)",
    )
    ax.scatter(
        signal_2d[:, 0],
        signal_2d[:, 1],
        s=18,
        alpha=0.65,
        marker="x",
        label=f"{signal_label} (Signal)",
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
        label=f"{background_label} (Background)",
    )
    ax.scatter(
        lda_scores[num_background:],
        residual_pc1_scores[num_background:],
        s=18,
        alpha=0.65,
        marker="x",
        label=f"{signal_label} (Signal)",
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

def plot_score_distribution(
    background: np.ndarray,
    signal: np.ndarray,
    title: str,
    xlabel: str,
    path: Path,
    background_label: str = "Background (validation)",
    signal_label: str = "signal",
) -> None:
    """Plot two score distributions with robust common histogram limits."""

    background = np.asarray(background, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    combined = np.concatenate([background, signal])

    low, high = np.quantile(combined, [0.0025, 0.9975])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(combined))
        high = float(np.max(combined))
    if high <= low:
        high = low + 1.0

    bins = np.linspace(low, high, 81)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.hist(
        np.clip(background, low, high),
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.7,
        label=background_label,
    )
    ax.hist(
        np.clip(signal, low, high),
        bins=bins,
        density=True,
        histtype="step",
        linewidth=1.7,
        label=signal_label,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(False)
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
    ax.set_title(title)
    ax.legend()
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def centroid_info(train_z: np.ndarray, val_z: np.ndarray) -> Dict[str, object]:
    train_c = train_z.mean(axis=0, dtype=np.float64)
    val_c = val_z.mean(axis=0, dtype=np.float64)
    return {
        "train_count": int(len(train_z)),
        "validation_count": int(len(val_z)),
        "train_centroid": train_c.tolist(),
        "validation_centroid": val_c.tolist(),
        "train_centroid_l2_norm": float(np.linalg.norm(train_c)),
        "validation_centroid_l2_norm": float(np.linalg.norm(val_c)),
        "train_validation_centroid_l2_distance": float(np.linalg.norm(val_c - train_c)),
    }


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

    backgrounds = list(summary["background_labels"])
    signals = list(summary["signal_labels"])
    backend.validate_requested_labels(backgrounds + signals)
    overlap = sorted(set(backgrounds) & set(signals))
    if overlap:
        raise ValueError(f"Background and signal labels overlap: {overlap}")

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
        summary.get("num_workers", 4) if args.num_workers is None else args.num_workers
    )
    cov_eps = float(
        args.mahalanobis_cov_eps
        if args.mahalanobis_cov_eps is not None
        else summary.get("mahalanobis_cov_eps", 1e-4)
    )
    precision = str(summary.get("precision", "fp32"))

    print(f"Dataset backend: {backend.dataset_name}")
    print(f"Dataset root: {backend.dataset_root}")
    print(f"Loading {checkpoint_path} on {device}")
    print(f"Sampling {steps} x {batch_size} events from each source stream")
    print(f"Full-jet density latent space: {args.full_latent_space}")
    print(f"Background labels: {background_display_name}")
    print(f"Signal labels: {signal_display_name}")
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

    print("\nCollecting full-event latents from train, validation, and signal streams...")
    train_mixed_z, train_mixed_y = collect_interleaved_full_latents(
        model=model,
        background_loader=backend.make_loader(
            split_name="train",
            labels=backgrounds,
            seed=seed + 101,
            **loader_common,
        ),
        signal_loader=backend.make_loader(
            split_name="test",
            labels=signals,
            seed=seed + 301,
            **loader_common,
        ),
        steps=steps,
        device=device,
        precision=precision,
        description="Full latent: interleaved train background + signal",
        background_labels=backgrounds,
        signal_labels=signals,
        combined_batch_size=batch_size,
        seed=seed + 401,
        latent_space=args.full_latent_space,
    )
    train_background_mask = labels_mask(
        train_mixed_y,
        backgrounds,
        backend.label_axis,
    )
    train_z = train_mixed_z[train_background_mask]
    train_y = train_mixed_y[train_background_mask]

    validation_mixed_z, validation_mixed_y = collect_interleaved_full_latents(
        model=model,
        background_loader=backend.make_loader(
            split_name="val",
            labels=backgrounds,
            seed=seed + 202,
            **loader_common,
        ),
        signal_loader=backend.make_loader(
            split_name="test",
            labels=signals,
            seed=seed + 302,
            **loader_common,
        ),
        steps=steps,
        device=device,
        precision=precision,
        description="Full latent: interleaved validation background + signal",
        background_labels=backgrounds,
        signal_labels=signals,
        combined_batch_size=batch_size,
        seed=seed + 402,
        latent_space=args.full_latent_space,
    )
    validation_background_mask = labels_mask(
        validation_mixed_y,
        backgrounds,
        backend.label_axis,
    )
    validation_signal_mask = labels_mask(
        validation_mixed_y,
        signals,
        backend.label_axis,
    )
    val_z = validation_mixed_z[validation_background_mask]
    val_y = validation_mixed_y[validation_background_mask]
    signal_z = validation_mixed_z[validation_signal_mask]
    signal_y = validation_mixed_y[validation_signal_mask]

    for background_index, background in enumerate(backgrounds):
        background_name = display_label(background)
        background_pair_z = validation_mixed_z[
            labels_mask(
                validation_mixed_y,
                [background],
                backend.label_axis,
            )
        ]
        max_pair_points = int(summary.get("max_latent_plot_points", 5000))
        plot_pair_latent_space(
            background=background_pair_z,
            signal=signal_z,
            background_label=background_name,
            signal_label=signal_display_name,
            path=(
                output_dir
                / f"01_pairwise_pca_{background_name.lower()}_vs_"
                f"{signal_display_name.lower()}.png"
            ),
            seed=seed + 450 + background_index,
            max_points=max_pair_points,
        )
        plot_pair_lda_residual_pca_space(
            background=background_pair_z,
            signal=signal_z,
            background_label=background_name,
            signal_label=signal_display_name,
            path=(
                output_dir
                / f"02_pairwise_lda_residual_pca_{background_name.lower()}_vs_"
                f"{signal_display_name.lower()}.png"
            ),
            seed=seed + 550 + background_index,
            cov_eps=cov_eps,
            max_points=max_pair_points,
        )

    if len(train_z) == 0 or len(val_z) == 0 or len(signal_z) == 0:
        raise RuntimeError(
            "Interleaved latent collection produced an empty regrouped sample: "
            f"train_background={len(train_z)}, "
            f"validation_background={len(val_z)}, signal={len(signal_z)}."
        )

    print(
        "Regrouped interleaved full latents: "
        f"train background={len(train_z)}, "
        f"validation background={len(val_z)}, signal={len(signal_z)}"
    )

    fit_idx, heldout_idx = stratified_split(
        train_y,
        backgrounds,
        backend.label_axis,
        args.fit_fraction,
        seed + 404,
    )
    mean, precision_matrix, cov_diag = fit_mahalanobis(train_z[fit_idx], cov_eps)
    fit_score = mahalanobis_scores(train_z[fit_idx], mean, precision_matrix)
    heldout_score = mahalanobis_scores(train_z[heldout_idx], mean, precision_matrix)
    val_score = mahalanobis_scores(val_z, mean, precision_matrix)
    signal_score = mahalanobis_scores(signal_z, mean, precision_matrix)

    combined = {
        "auc_fit_subset_vs_signal": auc(fit_score, signal_score),
        "auc_heldout_train_vs_signal": auc(heldout_score, signal_score),
        "auc_validation_vs_signal": auc(val_score, signal_score),
        "fit_background_scores": score_stats(fit_score),
        "heldout_train_background_scores": score_stats(heldout_score),
        "validation_background_scores": score_stats(val_score),
        "signal_scores": score_stats(signal_score),
        "covariance": cov_diag,
    }
    plot_comparison(
        [
            ("Held-out train", heldout_score, signal_score),
            ("Validation", val_score, signal_score),
        ],
        "Mahalanobis ROC from a train-fit background Gaussian",
        output_dir / "03_combined_mahalanobis_comparison.png",
    )
    print(
        f"AUC fit={combined['auc_fit_subset_vs_signal']:.6f}, "
        f"held-out={combined['auc_heldout_train_vs_signal']:.6f}, "
        f"validation={combined['auc_validation_vs_signal']:.6f}"
    )

    print("\nPer-class centroid and Mahalanobis diagnostic")
    per_class: Dict[str, Dict[str, object]] = {}
    train_curves = []
    validation_curves = []
    for label in backgrounds:
        name = display_label(label)
        train_mask = class_mask(train_y, label, backend.label_axis)
        val_mask = class_mask(val_y, label, backend.label_axis)
        class_train_indices = np.flatnonzero(train_mask)
        class_fit_idx = fit_idx[np.isin(fit_idx, class_train_indices)]
        class_heldout_idx = heldout_idx[np.isin(heldout_idx, class_train_indices)]

        class_mean, class_precision, class_cov_diag = fit_mahalanobis(
            train_z[class_fit_idx], cov_eps
        )
        class_fit_score = mahalanobis_scores(
            train_z[class_fit_idx], class_mean, class_precision
        )
        class_heldout_score = mahalanobis_scores(
            train_z[class_heldout_idx], class_mean, class_precision
        )
        class_val_score = mahalanobis_scores(
            val_z[val_mask], class_mean, class_precision
        )
        class_signal_score = mahalanobis_scores(
            signal_z, class_mean, class_precision
        )

        result = centroid_info(train_z[train_mask], val_z[val_mask])
        result.update(
            {
                "auc_fit_subset_vs_signal": auc(class_fit_score, class_signal_score),
                "auc_heldout_train_vs_signal": auc(
                    class_heldout_score, class_signal_score
                ),
                "auc_validation_vs_signal": auc(class_val_score, class_signal_score),
                "fit_background_scores": score_stats(class_fit_score),
                "heldout_train_background_scores": score_stats(class_heldout_score),
                "validation_background_scores": score_stats(class_val_score),
                "signal_scores": score_stats(class_signal_score),
                "covariance": class_cov_diag,
            }
        )
        per_class[label] = result

        plot_comparison(
            [
                ("Held-out train", class_heldout_score, class_signal_score),
                ("Validation", class_val_score, class_signal_score),
            ],
            f"{name}-specific Mahalanobis ROC",
            output_dir / f"04_{name.lower()}_train_validation_comparison.png",
        )
        train_curves.append(
            (f"{name} held-out train", class_heldout_score, class_signal_score)
        )
        validation_curves.append(
            (f"{name} validation", class_val_score, class_signal_score)
        )
        print(
            f"{name}: centroid shift="
            f"{result['train_validation_centroid_l2_distance']:.6g}, "
            f"AUC held-out={result['auc_heldout_train_vs_signal']:.6f}, "
            f"validation={result['auc_validation_vs_signal']:.6f}"
        )

    plot_comparison(
        train_curves,
        "Class-specific held-out train Mahalanobis ROC",
        output_dir / "05_per_class_train_comparison.png",
    )
    plot_comparison(
        validation_curves,
        "Class-specific validation Mahalanobis ROC",
        output_dir / "06_per_class_validation_comparison.png",
    )

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
            "fit_fraction": float(args.fit_fraction),
            "num_gaussians": int(args.num_gaussians),
            "knn_k": int(args.knn_k),
            "seed": seed,
            "max_num_particles": backend.max_num_particles,
        },
        "labels": {"background": backgrounds, "signal": signals},
        "full_latent_space": args.full_latent_space,
        "sample_counts": {
            "background_train_total": int(len(train_z)),
            "background_train_fit": int(len(fit_idx)),
            "background_train_heldout": int(len(heldout_idx)),
            "background_validation_total": int(len(val_z)),
            "signal_total": int(len(signal_z)),
        },
        "combined_mahalanobis": combined,
        "per_class": per_class,
    }
    with (output_dir / "diagnostic_results.json").open("w") as handle:
        json.dump(safe_json(results), handle, indent=2)

    print(f"\nSaved diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
