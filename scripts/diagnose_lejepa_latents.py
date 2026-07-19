#!/usr/bin/env python3
"""Post-training latent diagnostics for a JetClass LeJEPA run.

This script reuses the dataset/preprocessing pipeline from
run_train_lejepa_part_jetclass.py, reads model settings from summary.json,
loads best_model.pth, and runs:

1. Unaugmented full-jet Mahalanobis diagnostics with a stratified
   train-fit / train-held-out split.
2. Augmented multi-view local-global consistency diagnostics.
3. Separate background-class centroid and class-specific Mahalanobis diagnostics.
4. Multi-Gaussian nearest-component Mahalanobis diagnostics.
5. Standardized latent-space k-nearest-neighbor distance diagnostics.
6. Class-conditional background Gaussian-mixture likelihood diagnostics.
7. Two-sided local-global consistency diagnostics.
8. Full-jet/view latent norms plus cosine and unit-normalized local-global diagnostics.

Example:
    python -u scripts/diagnose_lejepa_latents.py \
        plots/run-lejepa-semi-sup-jetclass-ddp
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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

try:
    from run_train_lejepa_part_jetclass import (
        JETCLASS_LABELS,
        JetClassIterableDataset,
        collate_jetclass_tensors,
    )
except ImportError:
    from scripts.run_train_lejepa_part_jetclass import (
        JETCLASS_LABELS,
        JetClassIterableDataset,
        collate_jetclass_tensors,
    )

from models.part_jetclass import (
    CorruptedNegativeAugmentationConfig,
    LeJEPALossConfig,
    LeJEPAParticleTransformerRepresentation,
    LeJEPASemiSupervisedParticleTransformerRepresentation,
    LeJEPASemiSupervisedTripletParticleTransformerRepresentation,
    LeJEPATripletParticleTransformerRepresentation,
    MultiViewAugmentationConfig,
    ParticleTransformerConfig,
    SemiSupervisedLossConfig,
    TripletLossConfig,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose latent geometry and anomaly scores of a trained LeJEPA run."
    )
    parser.add_argument(
        "run_dir", type=Path, help="Directory containing summary.json and best_model.pth."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=None)
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
        help=(
            "Number of KMeans-defined Gaussian components used by the "
            "nearest-component Mahalanobis diagnostic. Default: 6."
        ),
    )
    parser.add_argument(
        "--knn-k",
        type=int,
        default=30,
        help=(
            "Number of standardized latent-space neighbors averaged by the "
            "kNN anomaly score. Default: 30."
        ),
    )
    parser.add_argument("--mahalanobis-cov-eps", type=float, default=None)
    parser.add_argument(
        "--max-num-particles",
        type=int,
        default=128,
        help="Training default is 128; summary.json currently does not store it.",
    )
    parser.add_argument(
        "--full-latent-space",
        choices=["representation", "cls"],
        default="representation",
        help=(
            "Which unaugmented latent to use for density scores. "
            "'representation' applies representation_head to the CLS state and "
            "matches LeJEPA/triplet losses; 'cls' keeps old raw-CLS behavior."
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


def make_loader(
    split_dir: Path,
    labels: Sequence[str],
    particle_features: Sequence[str],
    max_num_particles: int,
    shuffle_active_shards: int,
    seed: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = JetClassIterableDataset(
        split_dir=str(split_dir),
        labels_to_load=labels,
        particle_features=particle_features,
        max_num_particles=max_num_particles,
        max_events=None,
        shuffle_files=True,
        shuffle_active_shards=shuffle_active_shards,
        infinite=True,
        seed=seed,
        rank=0,
        world_size=1,
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_jetclass_tensors,
        "persistent_workers": False,
        "drop_last": True,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 1
    return DataLoader(**kwargs)


def build_model(summary: Dict[str, object], device: torch.device) -> torch.nn.Module:
    features = list(summary["particle_features"])
    if "part_eta" in features or "part_phi" in features:
        raise ValueError("part_eta/part_phi must not appear in the revised JetClass pipeline.")
    precision = str(summary.get("precision", "fp32"))

    expected_features = [
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
    if features != expected_features:
        raise ValueError(
            "The run summary does not use the revised JetClass feature pipeline. "
            f"Expected {expected_features}, found {features}. "
            "Use the older diagnostic script for checkpoints trained with the old "
            "part_eta/part_phi and raw impact-parameter inputs."
        )

    model_config = ParticleTransformerConfig(
        input_dim=len(features),
        input_feature_names=tuple(features),
        embed_dim=int(summary["embed_dim"]),
        num_heads=int(summary["num_heads"]),
        num_layers=int(summary["num_layers"]),
        ffn_mult=int(summary.get("ffn_mult", 4)),
        dropout=float(summary.get("dropout", 0.1)),
        representation_dim=int(summary["representation_dim"]),
        use_pairwise_bias=bool(summary.get("use_pairwise_bias", True)),
        pairwise_hidden_dim=int(summary.get("pairwise_hidden_dim", 64)),
        pairwise_num_features=int(summary.get("pairwise_num_features", 4)),
        compute_dtype=precision_to_dtype(precision),
        use_internal_autocast=False,
        eps=float(summary.get("eps", 1e-8)),
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

    model_name = str(summary.get("model", "semi-sup"))

    negative_augmentation_config = None
    triplet_loss_config = None
    if model_name in {"triplet", "semi-sup-triplet"}:
        negative_augmentation_config = CorruptedNegativeAugmentationConfig(
            num_negative_views=int(summary.get("num_negative_views", 4)),
            batch_mix_prob=float(summary.get("batch_mix_prob", 0.45)),
            pt_resample_prob=float(summary.get("pt_resample_prob", 0.25)),
            node_deta_dphi_rotation_prob=float(
                summary.get("node_deta_dphi_rotation_prob", 0.20)
            ),
            deta_dphi_shuffle_prob=float(
                summary.get("deta_dphi_shuffle_prob", 0.05)
            ),
            identity_shuffle_prob=float(summary.get("identity_shuffle_prob", 0.05)),
            min_nodes=int(summary.get("min_nodes", 4)),
            eps=float(summary.get("eps", 1e-8)),
            deta_index=features.index("part_deta"),
            dphi_index=features.index("part_dphi"),
            pt_index=features.index("part_pt"),
            log_pt_fraction_index=features.index("log_pt_fraction"),
            d0_sig_index=features.index("d0_sig"),
            dz_sig_index=features.index("dz_sig"),
            charge_index=features.index("part_charge"),
            identity_start_index=features.index("part_isChargedHadron"),
            identity_end_index=features.index("part_isMuon") + 1,
            corrupt_node_frac=float(summary.get("corrupt_node_frac", 0.5)),
            batch_mix_anchor_frac_min=float(
                summary.get("batch_mix_anchor_frac_min", 0.3)
            ),
            batch_mix_anchor_frac_max=float(
                summary.get("batch_mix_anchor_frac_max", 0.7)
            ),
            renormalize_pt_sum=bool(
                summary.get("renormalize_negative_pt_sum", True)
            ),
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

    if model_name == "semi-sup":
        backgrounds = list(summary["background_labels"])
        model = LeJEPASemiSupervisedParticleTransformerRepresentation(
            model_config=model_config,
            augmentation_config=augmentation_config,
            loss_config=loss_config,
            semi_supervised_config=SemiSupervisedLossConfig(
                classification_weight=float(summary.get("classification_weight", 0.1)),
                num_classes=int(
                    summary.get("num_classification_classes", len(backgrounds))
                ),
            ),
        )
    elif model_name == "semi-sup-triplet":
        backgrounds = list(summary["background_labels"])
        model = LeJEPASemiSupervisedTripletParticleTransformerRepresentation(
            model_config=model_config,
            augmentation_config=augmentation_config,
            negative_augmentation_config=negative_augmentation_config,
            loss_config=loss_config,
            triplet_loss_config=triplet_loss_config,
            semi_supervised_config=SemiSupervisedLossConfig(
                classification_weight=float(summary.get("classification_weight", 0.1)),
                num_classes=int(
                    summary.get("num_classification_classes", len(backgrounds))
                ),
            ),
        )
    elif model_name == "triplet":
        model = LeJEPATripletParticleTransformerRepresentation(
            model_config=model_config,
            augmentation_config=augmentation_config,
            negative_augmentation_config=negative_augmentation_config,
            loss_config=loss_config,
            triplet_loss_config=triplet_loss_config,
        )
    elif model_name == "lejepa":
        model = LeJEPAParticleTransformerRepresentation(
            model_config=model_config,
            augmentation_config=augmentation_config,
            loss_config=loss_config,
        )
    else:
        raise ValueError(
            "This diagnostic script supports model='lejepa', 'semi-sup', "
            "'triplet', and 'semi-sup-triplet'; "
            f"found {model_name!r}."
        )
    return model.to(device)


def read_state_dict(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state_dict)}")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
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



def label_ids(y: np.ndarray) -> np.ndarray:
    if y.ndim != 2 or y.shape[1] != len(JETCLASS_LABELS):
        raise ValueError(f"Unexpected one-hot label shape: {y.shape}")
    return np.argmax(y, axis=1).astype(np.int64)


def class_mask(y: np.ndarray, label: str) -> np.ndarray:
    return label_ids(y) == JETCLASS_LABELS.index(label)


def labels_mask(y: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    ids = label_ids(y)
    requested_ids = np.asarray(
        [JETCLASS_LABELS.index(label) for label in labels],
        dtype=np.int64,
    )
    return np.isin(ids, requested_ids)


def stratified_split(
    y: np.ndarray,
    labels: Sequence[str],
    fit_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("--fit-fraction must lie strictly between 0 and 1.")
    ids = label_ids(y)
    rng = np.random.default_rng(seed)
    fit, heldout = [], []
    for label in labels:
        indices = np.flatnonzero(ids == JETCLASS_LABELS.index(label))
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

    with summary_path.open() as f:
        summary = json.load(f)

    device = resolve_device(args.device)
    seed = int(summary.get("base_seed", 42))
    seed_everything(seed)

    dataset_root = (
        args.dataset_root.expanduser().resolve()
        if args.dataset_root
        else Path(summary["dataset_root"]).expanduser().resolve()
    )
    train_dir, val_dir, test_dir = (
        dataset_root / "train_100M",
        dataset_root / "val_5M",
        dataset_root / "test_20M",
    )
    for directory in (train_dir, val_dir, test_dir):
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing JetClass split: {directory}")

    backgrounds = list(summary["background_labels"])
    signals = list(summary["signal_labels"])
    features = list(summary["particle_features"])

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
    model_name = str(summary.get("model", "semi-sup"))
    num_global_views = int(summary.get("num_global_views", 2))

    print(f"Loading {checkpoint_path} on {device}")
    print(f"Sampling {steps} x {batch_size} events from each split")
    print(f"Full-jet density latent space: {args.full_latent_space}")
    print(f"Background labels: {background_display_name}")
    print(f"Signal labels: {signal_display_name}")
    print(
        "Full-jet latent collection retains the interleaved background-signal "
        "path for compatibility with older models; current eval-mode "
        "normalization uses frozen statistics."
    )
    model = build_model(summary, device)
    model.load_state_dict(read_state_dict(checkpoint_path, device), strict=True)
    model.eval()

    common = {
        "particle_features": features,
        "max_num_particles": args.max_num_particles,
        "shuffle_active_shards": int(summary.get("shuffle_active_shards", 3)),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }

    print("\nCollecting latents for full-events from train, val, and signals...")

    train_mixed_z, train_mixed_y = collect_interleaved_full_latents(
        model=model,
        background_loader=make_loader(
            train_dir, backgrounds, seed=seed + 101, **common
        ),
        signal_loader=make_loader(
            test_dir, signals, seed=seed + 301, **common
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
    train_background_mask = labels_mask(train_mixed_y, backgrounds)
    train_z = train_mixed_z[train_background_mask]
    train_y = train_mixed_y[train_background_mask]

    validation_mixed_z, validation_mixed_y = collect_interleaved_full_latents(
        model=model,
        background_loader=make_loader(
            val_dir, backgrounds, seed=seed + 202, **common
        ),
        signal_loader=make_loader(
            test_dir, signals, seed=seed + 302, **common
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
    validation_background_mask = labels_mask(validation_mixed_y, backgrounds)
    validation_signal_mask = labels_mask(validation_mixed_y, signals)
    val_z = validation_mixed_z[validation_background_mask]
    val_y = validation_mixed_y[validation_background_mask]
    signal_z = validation_mixed_z[validation_signal_mask]
    signal_y = validation_mixed_y[validation_signal_mask]
    
    # Plot pair-wise latent space for each background class against the signal
    for background_index, background in enumerate(backgrounds):
        background_name = display_label(background)
        background_pair_z = validation_mixed_z[
            labels_mask(validation_mixed_y, [background])
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
        train_y, backgrounds, args.fit_fraction, seed + 404
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
        name = label.removeprefix("label_")
        train_mask = class_mask(train_y, label)
        val_mask = class_mask(val_y, label)
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
                "heldout_train_background_scores": score_stats(
                    class_heldout_score
                ),
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
            f"{name}: centroid shift={result['train_validation_centroid_l2_distance']:.6g}, "
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
        "dataset_root": str(dataset_root),
        "device": str(device),
        "sampling": {
            "eval_steps": steps,
            "batch_size": batch_size,
            "events_per_dataset": steps * batch_size,
            "num_workers": num_workers,
            "fit_fraction": float(args.fit_fraction),
            "num_gaussians": int(args.num_gaussians),
            "knn_k": int(args.knn_k),
            "seed": seed,
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
    with (output_dir / "diagnostic_results.json").open("w") as f:
        json.dump(safe_json(results), f, indent=2)

    print(f"\nSaved diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
