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
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors
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
from visualize.plot_metrics import plot_roc_curve


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


JETCLASS_PIPELINE_VERSION = "relative16-no-eta-phi-v2"

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

    The model batch-standardizes selected particle features inside ``forward``.
    Consequently, separately forwarding a mixed-background batch and a
    signal-only batch produces representations under different normalization
    contexts. This collector fixes that diagnostic mismatch by constructing one
    combined batch, randomly interleaving its events, forwarding it once, and
    retaining the original one-hot labels for regrouping afterward.

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


def project_labels(
    y: torch.Tensor, model_name: str, background_labels: Sequence[str]
) -> torch.Tensor:
    if model_name not in {"semi-sup", "semi-sup-triplet"}:
        return y
    indices = [JETCLASS_LABELS.index(label) for label in background_labels]
    return y[:, indices]


def local_global_metrics(
    z_views: torch.Tensor,
    num_global_views: int,
    eps: float = 1e-12,
) -> Dict[str, torch.Tensor]:
    """Compute several event-level local/global view diagnostics.

    z_views has shape (V, B, D). The first G views are global and the
    remaining views are local. Every returned tensor has shape (B,).

    normalized_mse is the MSE after L2-normalizing both vectors. Therefore it
    is a constant rescaling of cosine distance for fixed latent dimension:

        normalized_mse = 2 * cosine_distance / D.
    """

    if z_views.ndim != 3 or not 1 <= num_global_views < z_views.size(0):
        raise ValueError(
            f"Invalid z_views/G: shape={tuple(z_views.shape)}, "
            f"G={num_global_views}."
        )

    z_views = z_views.float()
    global_views = z_views[:num_global_views]
    local_views = z_views[num_global_views:]
    anchor = global_views.mean(dim=0)
    anchor_expanded = anchor.unsqueeze(0).expand_as(local_views)

    mse = (local_views - anchor_expanded).square().mean(dim=-1).mean(dim=0)

    cosine_distance = (
        1.0
        - F.cosine_similarity(
            local_views,
            anchor_expanded,
            dim=-1,
            eps=eps,
        )
    ).mean(dim=0)

    local_unit = F.normalize(local_views, p=2, dim=-1, eps=eps)
    anchor_unit = F.normalize(anchor, p=2, dim=-1, eps=eps)
    normalized_mse = (
        local_unit - anchor_unit.unsqueeze(0)
    ).square().mean(dim=-1).mean(dim=0)

    return {
        "mse": mse,
        "cosine_distance": cosine_distance,
        "normalized_mse": normalized_mse,
        "global_anchor_norm": anchor.norm(p=2, dim=-1),
        "mean_global_view_norm": global_views.norm(p=2, dim=-1).mean(dim=0),
        "mean_local_view_norm": local_views.norm(p=2, dim=-1).mean(dim=0),
        "mean_all_view_norm": z_views.norm(p=2, dim=-1).mean(dim=0),
    }


@torch.no_grad()
def collect_local_global_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    steps: int,
    device: torch.device,
    precision: str,
    model_name: str,
    background_labels: Sequence[str],
    num_global_views: int,
    description: str,
) -> Dict[str, np.ndarray]:
    model.eval()
    collected: Dict[str, List[torch.Tensor]] = {}
    iterator = iter(loader)

    for _ in tqdm(range(steps), desc=description):
        batch = next(iterator)
        x = batch["x_particles"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        y = project_labels(
            batch["y"].to(device, non_blocking=True),
            model_name,
            background_labels,
        )
        with autocast_context(device, precision):
            output = model.forward_pretrain(x, y, padding_mask=mask)

        metrics = local_global_metrics(
            output["z_views"],
            num_global_views,
        )
        for key, value in metrics.items():
            collected.setdefault(key, []).append(value.detach().float().cpu())

    return {
        key: torch.cat(values).numpy().astype(np.float64)
        for key, values in collected.items()
    }



def triplet_response_metrics(
    z_views: torch.Tensor,
    z_negatives: torch.Tensor,
    num_global_views: int,
    margin: float,
    use_global_views_as_positives: bool,
    normalize: bool = False,
) -> Dict[str, torch.Tensor]:
    """Event-level diagnostics matching the triplet training geometry.

    Returns shape-(B,) tensors. Larger `margin_score` or `hinge_violation`
    means the event violates the learned positive-vs-negative separation more.
    """

    if z_views.ndim != 3 or z_negatives.ndim != 3:
        raise ValueError(
            f"Expected z_views/z_negatives as (V,B,D)/(K,B,D), got "
            f"{tuple(z_views.shape)} and {tuple(z_negatives.shape)}."
        )
    if not 1 <= num_global_views <= z_views.size(0):
        raise ValueError(
            f"Invalid num_global_views={num_global_views} for z_views={tuple(z_views.shape)}."
        )

    z_views = z_views.float()
    z_negatives = z_negatives.float()
    if normalize:
        z_views = F.normalize(z_views, p=2, dim=-1)
        z_negatives = F.normalize(z_negatives, p=2, dim=-1)

    anchor = z_views[:num_global_views].mean(dim=0)
    positives = z_views[:num_global_views] if use_global_views_as_positives else z_views

    d_pos = (positives - anchor.unsqueeze(0)).square().mean(dim=-1)  # (P, B)
    d_neg = (z_negatives - anchor.unsqueeze(0)).square().mean(dim=-1)  # (K, B)

    # Per-event average distances.
    d_pos_mean = d_pos.mean(dim=0)
    d_neg_mean = d_neg.mean(dim=0)

    # AUC is invariant under adding the constant margin, but keeping it makes
    # the number directly interpretable as a margin violation before ReLU.
    margin_score = d_pos_mean - d_neg_mean + float(margin)

    # Exact event-wise analogue of the training triplet loss before batch mean.
    hinge_violation = F.relu(
        d_pos.unsqueeze(0) - d_neg.unsqueeze(1) + float(margin)
    ).mean(dim=(0, 1))

    cosine_pos = (
        1.0
        - F.cosine_similarity(positives, anchor.unsqueeze(0), dim=-1, eps=1e-12)
    ).mean(dim=0)
    cosine_neg = (
        1.0
        - F.cosine_similarity(z_negatives, anchor.unsqueeze(0), dim=-1, eps=1e-12)
    ).mean(dim=0)
    cosine_margin_score = cosine_pos - cosine_neg + float(margin)

    return {
        "triplet_pos_distance": d_pos_mean,
        "triplet_neg_distance": d_neg_mean,
        "triplet_distance_gap": d_neg_mean - d_pos_mean,
        "triplet_margin_score": margin_score,
        "triplet_hinge_violation": hinge_violation,
        "triplet_pos_cosine_distance": cosine_pos,
        "triplet_neg_cosine_distance": cosine_neg,
        "triplet_cosine_margin_score": cosine_margin_score,
    }


@torch.no_grad()
def collect_triplet_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    steps: int,
    device: torch.device,
    precision: str,
    model_name: str,
    background_labels: Sequence[str],
    num_global_views: int,
    margin: float,
    use_global_views_as_positives: bool,
    normalize: bool,
    description: str,
) -> Dict[str, np.ndarray]:
    model.eval()
    collected: Dict[str, List[torch.Tensor]] = {}
    iterator = iter(loader)

    for _ in tqdm(range(steps), desc=description):
        batch = next(iterator)
        x = batch["x_particles"].to(device, non_blocking=True)
        mask = batch["padding_mask"].to(device, non_blocking=True)
        y = project_labels(
            batch["y"].to(device, non_blocking=True),
            model_name,
            background_labels,
        )
        with autocast_context(device, precision):
            output = model.forward_pretrain(x, y, padding_mask=mask)

        if "z_negatives" not in output:
            raise RuntimeError(
                "The loaded model did not return z_negatives. "
                "Triplet diagnostics require model='triplet' or 'semi-sup-triplet'."
            )
        metrics = triplet_response_metrics(
            output["z_views"],
            output["z_negatives"],
            num_global_views=num_global_views,
            margin=margin,
            use_global_views_as_positives=use_global_views_as_positives,
            normalize=normalize,
        )
        for key, value in metrics.items():
            collected.setdefault(key, []).append(value.detach().float().cpu())

    return {
        key: torch.cat(values).numpy().astype(np.float64)
        for key, values in collected.items()
    }

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


def fit_multi_gaussian_mahalanobis(
    latents: np.ndarray,
    num_gaussians: int,
    cov_eps: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Partition train latents with KMeans and fit one full-covariance Gaussian
    inside each cluster.

    This is intentionally a nearest-ellipsoid diagnostic rather than a true
    Gaussian-mixture negative log likelihood. Component weights and
    log-determinant terms are not included in the final anomaly score.
    """

    x = np.asarray(latents, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"Expected latents shape (N, D), got {x.shape}.")
    if num_gaussians < 1:
        raise ValueError("--num-gaussians must be at least 1.")
    if num_gaussians > len(x):
        raise ValueError(
            f"--num-gaussians={num_gaussians} exceeds fit events={len(x)}."
        )

    kmeans = KMeans(
        n_clusters=num_gaussians,
        random_state=seed,
        n_init=20,
        algorithm="lloyd",
    )
    assignments = kmeans.fit_predict(x)

    means: List[np.ndarray] = []
    precisions: List[np.ndarray] = []
    components: List[Dict[str, object]] = []

    for component_idx in range(num_gaussians):
        component = x[assignments == component_idx]

        if len(component) < 2:
            raise RuntimeError(
                "A KMeans component contains fewer than two events: "
                f"component={component_idx}, count={len(component)}. "
                "Reduce --num-gaussians or collect more train events."
            )

        mean, precision, covariance_diag = fit_mahalanobis(
            component,
            cov_eps,
        )
        means.append(mean)
        precisions.append(precision)

        components.append(
            {
                "component": int(component_idx),
                "count": int(len(component)),
                "fraction": float(len(component) / len(x)),
                "kmeans_center_l2_norm": float(
                    np.linalg.norm(kmeans.cluster_centers_[component_idx])
                ),
                "covariance": covariance_diag,
            }
        )

    diagnostics: Dict[str, object] = {
        "num_gaussians": int(num_gaussians),
        "num_fit_events": int(len(x)),
        "kmeans_inertia": float(kmeans.inertia_),
        "kmeans_iterations": int(kmeans.n_iter_),
        "components": components,
    }

    return (
        np.stack(means, axis=0),
        np.stack(precisions, axis=0),
        assignments.astype(np.int64),
        diagnostics,
    )


def nearest_gaussian_mahalanobis_scores(
    latents: np.ndarray,
    means: np.ndarray,
    precisions: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Score each event by its minimum squared Mahalanobis distance to any
    fitted Gaussian component.

    Returns:
        scores:
            Shape (N,), minimum component distance.
        nearest_components:
            Shape (N,), index of the nearest component.
    """

    x = np.asarray(latents, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    precisions = np.asarray(precisions, dtype=np.float64)

    if x.ndim != 2:
        raise ValueError(f"Expected latents shape (N, D), got {x.shape}.")
    if means.ndim != 2:
        raise ValueError(f"Expected means shape (K, D), got {means.shape}.")
    if precisions.ndim != 3:
        raise ValueError(
            f"Expected precisions shape (K, D, D), got {precisions.shape}."
        )
    if means.shape[0] != precisions.shape[0]:
        raise ValueError(
            "Component count differs between means and precisions: "
            f"{means.shape[0]} vs {precisions.shape[0]}."
        )
    if x.shape[1] != means.shape[1]:
        raise ValueError(
            f"Latent dimension mismatch: x={x.shape}, means={means.shape}."
        )

    centered = x[:, None, :] - means[None, :, :]
    all_scores = np.einsum(
        "nki,kij,nkj->nk",
        centered,
        precisions,
        centered,
        optimize=True,
    )

    nearest = np.argmin(all_scores, axis=1)
    scores = all_scores[np.arange(len(x)), nearest]

    return scores.astype(np.float64), nearest.astype(np.int64)


def component_counts(assignments: np.ndarray, num_components: int) -> List[int]:
    return np.bincount(
        np.asarray(assignments, dtype=np.int64),
        minlength=num_components,
    ).astype(np.int64).tolist()



def fit_latent_standardizer(
    reference: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-dimension mean/std using only train-fit background latents."""

    x = np.asarray(reference, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected reference shape (N, D), got {x.shape}.")

    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale = np.where(scale > eps, scale, 1.0)
    return mean, scale


def standardize_latents(
    latents: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    return (np.asarray(latents, dtype=np.float64) - mean) / scale


def fit_knn_reference(
    reference: np.ndarray,
    requested_k: int,
) -> Tuple[NearestNeighbors, int]:
    """Fit a Euclidean kNN index on standardized train-fit latents."""

    x = np.asarray(reference, dtype=np.float64)
    if requested_k < 1:
        raise ValueError("--knn-k must be at least 1.")
    if len(x) < 2:
        raise ValueError("At least two train-fit events are required for kNN.")

    effective_k = min(int(requested_k), len(x) - 1)
    model = NearestNeighbors(
        n_neighbors=effective_k + 1,
        metric="euclidean",
        algorithm="auto",
        n_jobs=-1,
    )
    model.fit(x)
    return model, effective_k


def knn_distance_scores(
    model: NearestNeighbors,
    latents: np.ndarray,
    k: int,
    reference_query: bool = False,
) -> np.ndarray:
    """
    Average Euclidean distance to the k nearest train-fit background events.

    For the train-fit reference itself, request k+1 neighbors and remove the
    self-match. Held-out/validation/signal queries use the nearest k directly.
    """

    x = np.asarray(latents, dtype=np.float64)
    n_neighbors = k + 1 if reference_query else k
    distances, _ = model.kneighbors(x, n_neighbors=n_neighbors)
    if reference_query:
        distances = distances[:, 1:]
    return distances.mean(axis=1).astype(np.float64)


def fit_gaussian_density_component(
    latents: np.ndarray,
    cov_eps: float,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[str, float]]:
    """Fit one regularized full-covariance Gaussian and its log determinant."""

    x = np.asarray(latents, dtype=np.float64)
    mean, precision, diagnostics = fit_mahalanobis(x, cov_eps)
    centered = x - mean
    cov = np.asarray(np.cov(centered, rowvar=False), dtype=np.float64)
    if cov.ndim == 0:
        cov = np.asarray([[float(cov)]], dtype=np.float64)
    reg_cov = cov + cov_eps * np.eye(cov.shape[0], dtype=np.float64)
    sign, logdet = np.linalg.slogdet(reg_cov)
    if sign <= 0 or not np.isfinite(logdet):
        raise RuntimeError(
            "Regularized covariance is not positive definite enough for "
            f"Gaussian likelihood; sign={sign}, logdet={logdet}."
        )
    return mean, precision, float(logdet), diagnostics


def fit_class_conditional_gaussian_mixture(
    train_z: np.ndarray,
    train_y: np.ndarray,
    fit_idx: np.ndarray,
    labels: Sequence[str],
    cov_eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """
    Fit one Gaussian per known background class and empirical class priors.

    This is a supervised density model only with respect to the known
    background subclasses; The configured signal sample is never used during fitting.
    """

    ids = label_ids(train_y)
    means: List[np.ndarray] = []
    precisions: List[np.ndarray] = []
    logdets: List[float] = []
    counts: List[int] = []
    components: List[Dict[str, object]] = []

    for label in labels:
        label_id = JETCLASS_LABELS.index(label)
        class_fit_idx = fit_idx[ids[fit_idx] == label_id]
        if len(class_fit_idx) < 2:
            raise RuntimeError(
                f"Only {len(class_fit_idx)} train-fit events found for {label}."
            )

        mean, precision, logdet, covariance_diag = (
            fit_gaussian_density_component(train_z[class_fit_idx], cov_eps)
        )
        means.append(mean)
        precisions.append(precision)
        logdets.append(logdet)
        counts.append(int(len(class_fit_idx)))
        components.append(
            {
                "label": label,
                "count": int(len(class_fit_idx)),
                "log_determinant": float(logdet),
                "covariance": covariance_diag,
            }
        )

    counts_array = np.asarray(counts, dtype=np.float64)
    priors = counts_array / counts_array.sum()
    for component, prior in zip(components, priors):
        component["prior"] = float(prior)

    diagnostics: Dict[str, object] = {
        "num_components": int(len(labels)),
        "components": components,
    }
    return (
        np.stack(means, axis=0),
        np.stack(precisions, axis=0),
        np.asarray(logdets, dtype=np.float64),
        priors,
        diagnostics,
    )


def stable_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(values - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(result, axis=axis)


def class_conditional_gaussian_mixture_nll(
    latents: np.ndarray,
    means: np.ndarray,
    precisions: np.ndarray,
    logdets: np.ndarray,
    priors: np.ndarray,
) -> np.ndarray:
    """Negative log density under the QCD/Hbb/Hcc Gaussian mixture."""

    x = np.asarray(latents, dtype=np.float64)
    centered = x[:, None, :] - means[None, :, :]
    quadratic = np.einsum(
        "nki,kij,nkj->nk",
        centered,
        precisions,
        centered,
        optimize=True,
    )
    dimension = x.shape[1]
    component_log_prob = -0.5 * (
        dimension * np.log(2.0 * np.pi)
        + logdets[None, :]
        + quadratic
    )
    mixture_log_prob = stable_logsumexp(
        component_log_prob + np.log(priors)[None, :],
        axis=1,
    )
    return (-mixture_log_prob).astype(np.float64)


def random_split_indices(
    n: int,
    fit_fraction: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if n < 4:
        raise ValueError("At least four events are required for a fit/held-out split.")
    if not 0.0 < fit_fraction < 1.0:
        raise ValueError("--fit-fraction must lie strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    cut = min(max(int(round(n * fit_fraction)), 2), n - 2)
    return indices[:cut], indices[cut:]


def empirical_two_sided_scores(
    reference: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a scalar diagnostic into a two-sided background-tail anomaly score.

    For value x, compute lower and upper empirical tail probabilities against
    train-fit background, use p_two = min(1, 2 * min(p_lower, p_upper)), and
    return -log(p_two). Both unusually small and unusually large values score
    as anomalous.
    """

    ref = np.sort(np.asarray(reference, dtype=np.float64))
    x = np.asarray(values, dtype=np.float64)
    n = len(ref)
    if n < 2:
        raise ValueError("At least two reference scores are required.")

    num_le = np.searchsorted(ref, x, side="right")
    num_ge = n - np.searchsorted(ref, x, side="left")

    p_lower = (num_le + 1.0) / (n + 1.0)
    p_upper = (num_ge + 1.0) / (n + 1.0)
    p_two = np.minimum(1.0, 2.0 * np.minimum(p_lower, p_upper))
    anomaly = -np.log(np.maximum(p_two, np.finfo(np.float64).tiny))
    return anomaly.astype(np.float64), p_two.astype(np.float64)


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


def plot_one_roc(
    background: np.ndarray,
    signal: np.ndarray,
    background_label: str,
    path: Path,
) -> None:
    plot_roc_curve(
        background,
        signal,
        background_label=background_label,
        signal_label="signal",
        savepath=str(path),
        examples=False,
        loss_fn=None,
    )


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
        "Full-jet latent collection will interleave background and signal events "
        "inside every forward pass, then regroup representations by their "
        "original labels. This gives all Mahalanobis and per-class comparisons "
        "a shared batch-standardization context."
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

    print("\n[1/8] Unaugmented full-jet Mahalanobis diagnostic")

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
    plot_one_roc(
        heldout_score,
        signal_score,
        f"{background_display_name} (held-out train)",
        output_dir / "01_mahalanobis_heldout_train_vs_signal.png",
    )
    plot_one_roc(
        val_score,
        signal_score,
        f"{background_display_name} (validation)",
        output_dir / "01_mahalanobis_validation_vs_signal.png",
    )
    plot_comparison(
        [
            ("Held-out train", heldout_score, signal_score),
            ("Validation", val_score, signal_score),
        ],
        "Mahalanobis ROC from a train-fit background Gaussian",
        output_dir / "01_mahalanobis_comparison.png",
    )
    print(
        f"AUC fit={combined['auc_fit_subset_vs_signal']:.6f}, "
        f"held-out={combined['auc_heldout_train_vs_signal']:.6f}, "
        f"validation={combined['auc_validation_vs_signal']:.6f}"
    )

    print("\n[2/8] Full-jet latent norm diagnostic")
    train_full_norm = np.linalg.norm(train_z, axis=1)
    val_full_norm = np.linalg.norm(val_z, axis=1)
    signal_full_norm = np.linalg.norm(signal_z, axis=1)
    latent_norms = {
        "background_train": score_stats(train_full_norm),
        "background_validation": score_stats(val_full_norm),
        "signal": score_stats(signal_full_norm),
    }
    plot_score_distribution(
        val_full_norm,
        signal_full_norm,
        "Full-jet latent norm distribution",
        r"$\|z\|_2$",
        output_dir / "02_full_latent_norm_distribution.png",
    )
    print(
        "Full latent norm mean/median: "
        f"train bg={train_full_norm.mean():.6f}/{np.median(train_full_norm):.6f}, "
        f"val bg={val_full_norm.mean():.6f}/{np.median(val_full_norm):.6f}, "
        f"signal={signal_full_norm.mean():.6f}/{np.median(signal_full_norm):.6f}"
    )

    print("\n[3/8] Augmented local-global diagnostics")
    train_lg_metrics = collect_local_global_metrics(
        model,
        make_loader(train_dir, backgrounds, seed=seed + 101, **common),
        steps,
        device,
        precision,
        model_name,
        backgrounds,
        num_global_views,
        "Local-global: background train",
    )
    val_lg_metrics = collect_local_global_metrics(
        model,
        make_loader(val_dir, backgrounds, seed=seed + 202, **common),
        steps,
        device,
        precision,
        model_name,
        backgrounds,
        num_global_views,
        "Local-global: background validation",
    )
    signal_lg_metrics = collect_local_global_metrics(
        model,
        make_loader(test_dir, signals, seed=seed + 303, **common),
        steps,
        device,
        precision,
        model_name,
        backgrounds,
        num_global_views,
        f"Local-global: {signal_display_name}",
    )

    train_lg = train_lg_metrics["mse"]
    val_lg = val_lg_metrics["mse"]
    signal_lg = signal_lg_metrics["mse"]
    train_lg_cos = train_lg_metrics["cosine_distance"]
    val_lg_cos = val_lg_metrics["cosine_distance"]
    signal_lg_cos = signal_lg_metrics["cosine_distance"]
    train_lg_nmse = train_lg_metrics["normalized_mse"]
    val_lg_nmse = val_lg_metrics["normalized_mse"]
    signal_lg_nmse = signal_lg_metrics["normalized_mse"]

    local_global = {
        "definition": "mean local-to-global-anchor MSE in raw latent coordinates",
        "auc_train_vs_signal": auc(train_lg, signal_lg),
        "auc_validation_vs_signal": auc(val_lg, signal_lg),
        "train_background_scores": score_stats(train_lg),
        "validation_background_scores": score_stats(val_lg),
        "signal_scores": score_stats(signal_lg),
    }
    local_global_cosine = {
        "definition": "mean 1-cosine-similarity from local views to the global anchor",
        "auc_train_vs_signal": auc(train_lg_cos, signal_lg_cos),
        "auc_validation_vs_signal": auc(val_lg_cos, signal_lg_cos),
        "train_background_scores": score_stats(train_lg_cos),
        "validation_background_scores": score_stats(val_lg_cos),
        "signal_scores": score_stats(signal_lg_cos),
    }
    local_global_normalized_mse = {
        "definition": (
            "mean MSE after L2-normalizing local views and the global anchor; "
            "equal to 2*cosine_distance/latent_dim up to floating-point error"
        ),
        "auc_train_vs_signal": auc(train_lg_nmse, signal_lg_nmse),
        "auc_validation_vs_signal": auc(val_lg_nmse, signal_lg_nmse),
        "train_background_scores": score_stats(train_lg_nmse),
        "validation_background_scores": score_stats(val_lg_nmse),
        "signal_scores": score_stats(signal_lg_nmse),
    }

    augmented_latent_norms = {
        "background_train": {
            key: score_stats(train_lg_metrics[key])
            for key in (
                "global_anchor_norm",
                "mean_global_view_norm",
                "mean_local_view_norm",
                "mean_all_view_norm",
            )
        },
        "background_validation": {
            key: score_stats(val_lg_metrics[key])
            for key in (
                "global_anchor_norm",
                "mean_global_view_norm",
                "mean_local_view_norm",
                "mean_all_view_norm",
            )
        },
        "signal": {
            key: score_stats(signal_lg_metrics[key])
            for key in (
                "global_anchor_norm",
                "mean_global_view_norm",
                "mean_local_view_norm",
                "mean_all_view_norm",
            )
        },
    }

    plot_one_roc(
        train_lg,
        signal_lg,
        f"{background_display_name} (train)",
        output_dir / "03_local_global_mse_train_vs_signal.png",
    )
    plot_one_roc(
        val_lg,
        signal_lg,
        f"{background_display_name} (validation)",
        output_dir / "03_local_global_mse_validation_vs_signal.png",
    )
    plot_comparison(
        [("Train", train_lg, signal_lg), ("Validation", val_lg, signal_lg)],
        "Raw local-global MSE ROC",
        output_dir / "03_local_global_mse_comparison.png",
    )
    plot_score_distribution(
        val_lg,
        signal_lg,
        "Raw local-global MSE distribution",
        "Mean local-to-global MSE",
        output_dir / "03_local_global_mse_distribution.png",
    )

    plot_one_roc(
        val_lg_cos,
        signal_lg_cos,
        f"{background_display_name} (validation, cosine distance)",
        output_dir / "09_local_global_cosine_validation_vs_signal.png",
    )
    plot_score_distribution(
        val_lg_cos,
        signal_lg_cos,
        "Local-global cosine-distance distribution",
        "Mean 1 - cosine similarity",
        output_dir / "09_local_global_cosine_distribution.png",
    )

    plot_one_roc(
        val_lg_nmse,
        signal_lg_nmse,
        f"{background_display_name} (validation, unit-normalized MSE)",
        output_dir / "10_local_global_normalized_mse_validation_vs_signal.png",
    )
    plot_score_distribution(
        val_lg_nmse,
        signal_lg_nmse,
        "Unit-normalized local-global MSE distribution",
        "Mean MSE after L2 normalization",
        output_dir / "10_local_global_normalized_mse_distribution.png",
    )

    plot_score_distribution(
        val_lg_metrics["global_anchor_norm"],
        signal_lg_metrics["global_anchor_norm"],
        "Augmented global-anchor latent norm distribution",
        r"$\|z_{global\ anchor}\|_2$",
        output_dir / "10_augmented_global_anchor_norm_distribution.png",
    )

    print(
        f"Raw MSE AUC train={local_global['auc_train_vs_signal']:.6f}, "
        f"validation={local_global['auc_validation_vs_signal']:.6f}"
    )
    print(
        f"Cosine AUC train={local_global_cosine['auc_train_vs_signal']:.6f}, "
        f"validation={local_global_cosine['auc_validation_vs_signal']:.6f}"
    )
    print(
        "Normalized-MSE AUC "
        f"train={local_global_normalized_mse['auc_train_vs_signal']:.6f}, "
        f"validation={local_global_normalized_mse['auc_validation_vs_signal']:.6f}"
    )
    print(
        "Global-anchor norm mean/median: "
        f"train bg={train_lg_metrics['global_anchor_norm'].mean():.6f}/"
        f"{np.median(train_lg_metrics['global_anchor_norm']):.6f}, "
        f"val bg={val_lg_metrics['global_anchor_norm'].mean():.6f}/"
        f"{np.median(val_lg_metrics['global_anchor_norm']):.6f}, "
        f"{signal_display_name}={signal_lg_metrics['global_anchor_norm'].mean():.6f}/"
        f"{np.median(signal_lg_metrics['global_anchor_norm']):.6f}"
    )



    triplet_response = None
    triplet_metric_arrays: Dict[str, Dict[str, np.ndarray]] = {}
    if model_name in {"triplet", "semi-sup-triplet"}:
        print("\n[4/9] Triplet positive-vs-corrupted-negative response diagnostic")
        triplet_margin = float(summary.get("triplet_margin", 1.0))
        triplet_use_global_positives = not bool(
            summary.get("use_all_views_as_triplet_positives", False)
        )
        triplet_normalize = bool(
            summary.get("normalize_representations_for_triplet", False)
        )

        train_triplet_metrics = collect_triplet_metrics(
            model,
            make_loader(train_dir, backgrounds, seed=seed + 101, **common),
            steps,
            device,
            precision,
            model_name,
            backgrounds,
            num_global_views,
            triplet_margin,
            triplet_use_global_positives,
            triplet_normalize,
            "Triplet response: background train",
        )
        val_triplet_metrics = collect_triplet_metrics(
            model,
            make_loader(val_dir, backgrounds, seed=seed + 202, **common),
            steps,
            device,
            precision,
            model_name,
            backgrounds,
            num_global_views,
            triplet_margin,
            triplet_use_global_positives,
            triplet_normalize,
            "Triplet response: background validation",
        )
        signal_triplet_metrics = collect_triplet_metrics(
            model,
            make_loader(test_dir, signals, seed=seed + 303, **common),
            steps,
            device,
            precision,
            model_name,
            backgrounds,
            num_global_views,
            triplet_margin,
            triplet_use_global_positives,
            triplet_normalize,
            f"Triplet response: {signal_display_name}",
        )

        triplet_metric_arrays = {
            "train": train_triplet_metrics,
            "validation": val_triplet_metrics,
            "signal": signal_triplet_metrics,
        }

        triplet_response = {
            "definition": (
                "event-level triplet response from generated positive views and "
                "corrupted negative views; larger margin/hinge scores indicate "
                "weaker learned separation"
            ),
            "triplet_margin": triplet_margin,
            "use_global_views_as_positives": triplet_use_global_positives,
            "normalize_representations_for_triplet": triplet_normalize,
            "metrics": {},
        }
        for key in (
            "triplet_margin_score",
            "triplet_hinge_violation",
            "triplet_distance_gap",
            "triplet_pos_distance",
            "triplet_neg_distance",
            "triplet_cosine_margin_score",
            "triplet_pos_cosine_distance",
            "triplet_neg_cosine_distance",
        ):
            train_values = train_triplet_metrics[key]
            val_values = val_triplet_metrics[key]
            signal_values = signal_triplet_metrics[key]
            # For distance_gap, normal events should usually have larger
            # negative-positive gap, so use the sign that makes larger=anomaly.
            if key == "triplet_distance_gap":
                train_score_for_auc = -train_values
                val_score_for_auc = -val_values
                signal_score_for_auc = -signal_values
                score_description = "negative distance gap, i.e. d_pos - d_neg"
            else:
                train_score_for_auc = train_values
                val_score_for_auc = val_values
                signal_score_for_auc = signal_values
                score_description = key

            triplet_response["metrics"][key] = {
                "score_used_for_auc": score_description,
                "auc_train_vs_signal": auc(train_score_for_auc, signal_score_for_auc),
                "auc_validation_vs_signal": auc(val_score_for_auc, signal_score_for_auc),
                "train_background_scores": score_stats(train_values),
                "validation_background_scores": score_stats(val_values),
                "signal_scores": score_stats(signal_values),
            }

        triplet_margin_train = train_triplet_metrics["triplet_margin_score"]
        triplet_margin_val = val_triplet_metrics["triplet_margin_score"]
        triplet_margin_signal = signal_triplet_metrics["triplet_margin_score"]
        triplet_hinge_train = train_triplet_metrics["triplet_hinge_violation"]
        triplet_hinge_val = val_triplet_metrics["triplet_hinge_violation"]
        triplet_hinge_signal = signal_triplet_metrics["triplet_hinge_violation"]

        plot_one_roc(
            triplet_margin_val,
            triplet_margin_signal,
            f"{background_display_name} (validation, triplet margin score)",
            output_dir / "11_triplet_margin_score_validation_vs_signal.png",
        )
        plot_score_distribution(
            triplet_margin_val,
            triplet_margin_signal,
            "Triplet margin-score distribution",
            r"$d_+ - d_- + m$",
            output_dir / "11_triplet_margin_score_distribution.png",
        )
        plot_one_roc(
            triplet_hinge_val,
            triplet_hinge_signal,
            f"{background_display_name} (validation, triplet hinge violation)",
            output_dir / "12_triplet_hinge_violation_validation_vs_signal.png",
        )
        plot_score_distribution(
            triplet_hinge_val,
            triplet_hinge_signal,
            "Triplet hinge-violation distribution",
            r"mean ReLU$(d_+ - d_- + m)$",
            output_dir / "12_triplet_hinge_violation_distribution.png",
        )

        print(
            "Triplet margin score: "
            f"AUC train={triplet_response['metrics']['triplet_margin_score']['auc_train_vs_signal']:.6f}, "
            f"validation={triplet_response['metrics']['triplet_margin_score']['auc_validation_vs_signal']:.6f}"
        )
        print(
            "Triplet hinge violation: "
            f"AUC train={triplet_response['metrics']['triplet_hinge_violation']['auc_train_vs_signal']:.6f}, "
            f"validation={triplet_response['metrics']['triplet_hinge_violation']['auc_validation_vs_signal']:.6f}"
        )

    print("\n[5/9] Per-class centroid and Mahalanobis diagnostic")
    per_class: Dict[str, Dict[str, object]] = {}
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

        plot_one_roc(
            class_heldout_score,
            class_signal_score,
            f"{name} (held-out train)",
            output_dir / f"03_{name.lower()}_heldout_train_vs_signal.png",
        )
        plot_one_roc(
            class_val_score,
            class_signal_score,
            f"{name} (validation)",
            output_dir / f"03_{name.lower()}_validation_vs_signal.png",
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
        validation_curves,
        "Class-specific validation Mahalanobis ROC",
        output_dir / "03_per_class_validation_comparison.png",
    )

    print(
        "\n[6/9] Multi-Gaussian nearest-component Mahalanobis diagnostic"
    )
    (
        multi_means,
        multi_precisions,
        multi_fit_components,
        multi_fit_diagnostics,
    ) = fit_multi_gaussian_mahalanobis(
        train_z[fit_idx],
        num_gaussians=args.num_gaussians,
        cov_eps=cov_eps,
        seed=seed + 505,
    )

    multi_fit_score, multi_fit_nearest = (
        nearest_gaussian_mahalanobis_scores(
            train_z[fit_idx],
            multi_means,
            multi_precisions,
        )
    )
    multi_heldout_score, multi_heldout_nearest = (
        nearest_gaussian_mahalanobis_scores(
            train_z[heldout_idx],
            multi_means,
            multi_precisions,
        )
    )
    multi_val_score, multi_val_nearest = (
        nearest_gaussian_mahalanobis_scores(
            val_z,
            multi_means,
            multi_precisions,
        )
    )
    multi_signal_score, multi_signal_nearest = (
        nearest_gaussian_mahalanobis_scores(
            signal_z,
            multi_means,
            multi_precisions,
        )
    )

    multi_gaussian = {
        "num_gaussians": int(args.num_gaussians),
        "auc_fit_subset_vs_signal": auc(
            multi_fit_score,
            multi_signal_score,
        ),
        "auc_heldout_train_vs_signal": auc(
            multi_heldout_score,
            multi_signal_score,
        ),
        "auc_validation_vs_signal": auc(
            multi_val_score,
            multi_signal_score,
        ),
        "fit_background_scores": score_stats(multi_fit_score),
        "heldout_train_background_scores": score_stats(
            multi_heldout_score
        ),
        "validation_background_scores": score_stats(multi_val_score),
        "signal_scores": score_stats(multi_signal_score),
        "nearest_component_counts": {
            "fit_background": component_counts(
                multi_fit_nearest,
                args.num_gaussians,
            ),
            "heldout_train_background": component_counts(
                multi_heldout_nearest,
                args.num_gaussians,
            ),
            "validation_background": component_counts(
                multi_val_nearest,
                args.num_gaussians,
            ),
            "signal": component_counts(
                multi_signal_nearest,
                args.num_gaussians,
            ),
        },
        "fit": multi_fit_diagnostics,
    }

    plot_one_roc(
        multi_heldout_score,
        multi_signal_score,
        (
            f"{background_display_name} "
            f"(held-out train, nearest of {args.num_gaussians} Gaussians)"
        ),
        output_dir / "04_multi_gaussian_heldout_train_vs_signal.png",
    )
    plot_one_roc(
        multi_val_score,
        multi_signal_score,
        (
            f"{background_display_name} "
            f"(validation, nearest of {args.num_gaussians} Gaussians)"
        ),
        output_dir / "04_multi_gaussian_validation_vs_signal.png",
    )
    plot_comparison(
        [
            (
                f"Held-out train, K={args.num_gaussians}",
                multi_heldout_score,
                multi_signal_score,
            ),
            (
                f"Validation, K={args.num_gaussians}",
                multi_val_score,
                multi_signal_score,
            ),
        ],
        (
            "Nearest-component Mahalanobis ROC "
            f"with {args.num_gaussians} train Gaussians"
        ),
        output_dir / "04_multi_gaussian_comparison.png",
    )
    plot_comparison(
        [
            (
                "Single Gaussian validation",
                val_score,
                signal_score,
            ),
            (
                f"Nearest of {args.num_gaussians} Gaussians validation",
                multi_val_score,
                multi_signal_score,
            ),
        ],
        "Single- vs multi-Gaussian validation Mahalanobis ROC",
        output_dir / "04_single_vs_multi_gaussian_validation.png",
    )

    print(
        f"Multi-Gaussian K={args.num_gaussians}: "
        f"AUC fit={multi_gaussian['auc_fit_subset_vs_signal']:.6f}, "
        f"held-out={multi_gaussian['auc_heldout_train_vs_signal']:.6f}, "
        f"validation={multi_gaussian['auc_validation_vs_signal']:.6f}"
    )


    print("\n[7/9] Standardized latent-space kNN diagnostic")
    knn_mean, knn_scale = fit_latent_standardizer(train_z[fit_idx])
    knn_reference_z = standardize_latents(
        train_z[fit_idx],
        knn_mean,
        knn_scale,
    )
    knn_model, effective_knn_k = fit_knn_reference(
        knn_reference_z,
        requested_k=args.knn_k,
    )
    knn_fit_score = knn_distance_scores(
        knn_model,
        knn_reference_z,
        effective_knn_k,
        reference_query=True,
    )
    knn_heldout_score = knn_distance_scores(
        knn_model,
        standardize_latents(train_z[heldout_idx], knn_mean, knn_scale),
        effective_knn_k,
    )
    knn_val_score = knn_distance_scores(
        knn_model,
        standardize_latents(val_z, knn_mean, knn_scale),
        effective_knn_k,
    )
    knn_signal_score = knn_distance_scores(
        knn_model,
        standardize_latents(signal_z, knn_mean, knn_scale),
        effective_knn_k,
    )

    knn_diagnostic = {
        "requested_k": int(args.knn_k),
        "effective_k": int(effective_knn_k),
        "standardization": "per-dimension train-fit background z-score",
        "distance": "mean Euclidean distance to k nearest train-fit events",
        "auc_fit_subset_vs_signal": auc(knn_fit_score, knn_signal_score),
        "auc_heldout_train_vs_signal": auc(
            knn_heldout_score,
            knn_signal_score,
        ),
        "auc_validation_vs_signal": auc(knn_val_score, knn_signal_score),
        "fit_background_scores": score_stats(knn_fit_score),
        "heldout_train_background_scores": score_stats(knn_heldout_score),
        "validation_background_scores": score_stats(knn_val_score),
        "signal_scores": score_stats(knn_signal_score),
    }
    plot_one_roc(
        knn_heldout_score,
        knn_signal_score,
        f"{background_display_name} (held-out train, kNN k={effective_knn_k})",
        output_dir / "05_knn_heldout_train_vs_signal.png",
    )
    plot_one_roc(
        knn_val_score,
        knn_signal_score,
        f"{background_display_name} (validation, kNN k={effective_knn_k})",
        output_dir / "05_knn_validation_vs_signal.png",
    )
    plot_comparison(
        [
            ("Held-out train", knn_heldout_score, knn_signal_score),
            ("Validation", knn_val_score, knn_signal_score),
        ],
        f"Standardized latent kNN ROC (k={effective_knn_k})",
        output_dir / "05_knn_comparison.png",
    )
    print(
        f"kNN k={effective_knn_k}: "
        f"AUC fit={knn_diagnostic['auc_fit_subset_vs_signal']:.6f}, "
        f"held-out={knn_diagnostic['auc_heldout_train_vs_signal']:.6f}, "
        f"validation={knn_diagnostic['auc_validation_vs_signal']:.6f}"
    )

    print("\n[8/9] Class-conditional Gaussian-mixture likelihood diagnostic")
    (
        class_mix_means,
        class_mix_precisions,
        class_mix_logdets,
        class_mix_priors,
        class_mix_fit_diagnostics,
    ) = fit_class_conditional_gaussian_mixture(
        train_z,
        train_y,
        fit_idx,
        backgrounds,
        cov_eps,
    )
    class_mix_fit_score = class_conditional_gaussian_mixture_nll(
        train_z[fit_idx],
        class_mix_means,
        class_mix_precisions,
        class_mix_logdets,
        class_mix_priors,
    )
    class_mix_heldout_score = class_conditional_gaussian_mixture_nll(
        train_z[heldout_idx],
        class_mix_means,
        class_mix_precisions,
        class_mix_logdets,
        class_mix_priors,
    )
    class_mix_val_score = class_conditional_gaussian_mixture_nll(
        val_z,
        class_mix_means,
        class_mix_precisions,
        class_mix_logdets,
        class_mix_priors,
    )
    class_mix_signal_score = class_conditional_gaussian_mixture_nll(
        signal_z,
        class_mix_means,
        class_mix_precisions,
        class_mix_logdets,
        class_mix_priors,
    )

    class_conditional_mixture = {
        "definition": (
            "negative log likelihood under an empirical-prior mixture of "
            "one full-covariance Gaussian per background class"
        ),
        "auc_fit_subset_vs_signal": auc(
            class_mix_fit_score,
            class_mix_signal_score,
        ),
        "auc_heldout_train_vs_signal": auc(
            class_mix_heldout_score,
            class_mix_signal_score,
        ),
        "auc_validation_vs_signal": auc(
            class_mix_val_score,
            class_mix_signal_score,
        ),
        "fit_background_scores": score_stats(class_mix_fit_score),
        "heldout_train_background_scores": score_stats(
            class_mix_heldout_score
        ),
        "validation_background_scores": score_stats(class_mix_val_score),
        "signal_scores": score_stats(class_mix_signal_score),
        "fit": class_mix_fit_diagnostics,
    }
    plot_one_roc(
        class_mix_heldout_score,
        class_mix_signal_score,
        f"{background_display_name} (held-out train, class-conditional mixture NLL)",
        output_dir / "06_class_conditional_gmm_heldout_train_vs_signal.png",
    )
    plot_one_roc(
        class_mix_val_score,
        class_mix_signal_score,
        f"{background_display_name} (validation, class-conditional mixture NLL)",
        output_dir / "06_class_conditional_gmm_validation_vs_signal.png",
    )
    plot_comparison(
        [
            (
                "Held-out train",
                class_mix_heldout_score,
                class_mix_signal_score,
            ),
            (
                "Validation",
                class_mix_val_score,
                class_mix_signal_score,
            ),
        ],
        "Class-conditional Gaussian-mixture likelihood ROC",
        output_dir / "06_class_conditional_gmm_comparison.png",
    )
    print(
        "Class-conditional mixture NLL: "
        f"AUC fit={class_conditional_mixture['auc_fit_subset_vs_signal']:.6f}, "
        f"held-out={class_conditional_mixture['auc_heldout_train_vs_signal']:.6f}, "
        f"validation={class_conditional_mixture['auc_validation_vs_signal']:.6f}"
    )

    print("\n[9/9] Two-sided local-global consistency diagnostic")
    lg_fit_idx, lg_heldout_idx = random_split_indices(
        len(train_lg),
        args.fit_fraction,
        seed + 606,
    )
    lg_reference = train_lg[lg_fit_idx]
    lg_two_fit_score, lg_two_fit_p = empirical_two_sided_scores(
        lg_reference,
        train_lg[lg_fit_idx],
    )
    lg_two_heldout_score, lg_two_heldout_p = empirical_two_sided_scores(
        lg_reference,
        train_lg[lg_heldout_idx],
    )
    lg_two_val_score, lg_two_val_p = empirical_two_sided_scores(
        lg_reference,
        val_lg,
    )
    lg_two_signal_score, lg_two_signal_p = empirical_two_sided_scores(
        lg_reference,
        signal_lg,
    )

    two_sided_local_global = {
        "definition": (
            "-log of the two-sided empirical tail probability under the "
            "train-fit background local-global score distribution"
        ),
        "reference_count": int(len(lg_reference)),
        "auc_fit_subset_vs_signal": auc(
            lg_two_fit_score,
            lg_two_signal_score,
        ),
        "auc_heldout_train_vs_signal": auc(
            lg_two_heldout_score,
            lg_two_signal_score,
        ),
        "auc_validation_vs_signal": auc(
            lg_two_val_score,
            lg_two_signal_score,
        ),
        "fit_background_scores": score_stats(lg_two_fit_score),
        "heldout_train_background_scores": score_stats(
            lg_two_heldout_score
        ),
        "validation_background_scores": score_stats(lg_two_val_score),
        "signal_scores": score_stats(lg_two_signal_score),
    }
    plot_one_roc(
        lg_two_heldout_score,
        lg_two_signal_score,
        f"{background_display_name} (held-out train, two-sided local-global)",
        output_dir / "07_two_sided_local_global_heldout_train_vs_signal.png",
    )
    plot_one_roc(
        lg_two_val_score,
        lg_two_signal_score,
        f"{background_display_name} (validation, two-sided local-global)",
        output_dir / "07_two_sided_local_global_validation_vs_signal.png",
    )
    plot_comparison(
        [
            (
                "Held-out train",
                lg_two_heldout_score,
                lg_two_signal_score,
            ),
            (
                "Validation",
                lg_two_val_score,
                lg_two_signal_score,
            ),
        ],
        "Two-sided local-global empirical-tail ROC",
        output_dir / "07_two_sided_local_global_comparison.png",
    )
    comparison_curves = [
        ("Single Gaussian Mahalanobis", val_score, signal_score),
        (f"kNN k={effective_knn_k}", knn_val_score, knn_signal_score),
        (
            "Class-conditional mixture NLL",
            class_mix_val_score,
            class_mix_signal_score,
        ),
        ("Raw local-global MSE", val_lg, signal_lg),
        ("Local-global cosine distance", val_lg_cos, signal_lg_cos),
        ("Local-global normalized MSE", val_lg_nmse, signal_lg_nmse),
        (
            "Two-sided local-global",
            lg_two_val_score,
            lg_two_signal_score,
        ),
    ]
    if triplet_response is not None:
        comparison_curves.extend(
            [
                (
                    "Triplet margin score",
                    triplet_metric_arrays["validation"]["triplet_margin_score"],
                    triplet_metric_arrays["signal"]["triplet_margin_score"],
                ),
                (
                    "Triplet hinge violation",
                    triplet_metric_arrays["validation"]["triplet_hinge_violation"],
                    triplet_metric_arrays["signal"]["triplet_hinge_violation"],
                ),
            ]
        )
    plot_comparison(
        comparison_curves,
        "Validation anomaly-score ROC comparison",
        output_dir / "08_validation_score_comparison.png",
    )
    print(
        "Two-sided local-global: "
        f"AUC fit={two_sided_local_global['auc_fit_subset_vs_signal']:.6f}, "
        f"held-out={two_sided_local_global['auc_heldout_train_vs_signal']:.6f}, "
        f"validation={two_sided_local_global['auc_validation_vs_signal']:.6f}"
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
        "multi_gaussian_mahalanobis": multi_gaussian,
        "knn_distance": knn_diagnostic,
        "class_conditional_gaussian_mixture": class_conditional_mixture,
        "latent_norms": latent_norms,
        "augmented_latent_norms": augmented_latent_norms,
        "local_global": local_global,
        "local_global_cosine": local_global_cosine,
        "local_global_normalized_mse": local_global_normalized_mse,
        "two_sided_local_global": two_sided_local_global,
        "triplet_response": triplet_response,
        "per_class": per_class,
    }
    with (output_dir / "diagnostic_results.json").open("w") as f:
        json.dump(safe_json(results), f, indent=2)

    np.savez_compressed(
        output_dir / "diagnostic_scores.npz",
        combined_fit_background=fit_score,
        combined_heldout_train_background=heldout_score,
        combined_validation_background=val_score,
        combined_signal=signal_score,
        full_latent_norm_train_background=train_full_norm,
        full_latent_norm_validation_background=val_full_norm,
        full_latent_norm_signal=signal_full_norm,
        local_global_train_background=train_lg,
        local_global_validation_background=val_lg,
        local_global_signal=signal_lg,
        local_global_cosine_train_background=train_lg_cos,
        local_global_cosine_validation_background=val_lg_cos,
        local_global_cosine_signal=signal_lg_cos,
        local_global_normalized_mse_train_background=train_lg_nmse,
        local_global_normalized_mse_validation_background=val_lg_nmse,
        local_global_normalized_mse_signal=signal_lg_nmse,
        augmented_global_anchor_norm_train_background=train_lg_metrics["global_anchor_norm"],
        augmented_global_anchor_norm_validation_background=val_lg_metrics["global_anchor_norm"],
        augmented_global_anchor_norm_signal=signal_lg_metrics["global_anchor_norm"],
        augmented_mean_local_view_norm_train_background=train_lg_metrics["mean_local_view_norm"],
        augmented_mean_local_view_norm_validation_background=val_lg_metrics["mean_local_view_norm"],
        augmented_mean_local_view_norm_signal=signal_lg_metrics["mean_local_view_norm"],
        multi_gaussian_fit_background=multi_fit_score,
        multi_gaussian_heldout_train_background=multi_heldout_score,
        multi_gaussian_validation_background=multi_val_score,
        multi_gaussian_signal=multi_signal_score,
        multi_gaussian_fit_nearest_component=multi_fit_nearest,
        multi_gaussian_heldout_nearest_component=multi_heldout_nearest,
        multi_gaussian_validation_nearest_component=multi_val_nearest,
        multi_gaussian_signal_nearest_component=multi_signal_nearest,
        knn_fit_background=knn_fit_score,
        knn_heldout_train_background=knn_heldout_score,
        knn_validation_background=knn_val_score,
        knn_signal=knn_signal_score,
        class_conditional_gmm_fit_background=class_mix_fit_score,
        class_conditional_gmm_heldout_train_background=class_mix_heldout_score,
        class_conditional_gmm_validation_background=class_mix_val_score,
        class_conditional_gmm_signal=class_mix_signal_score,
        two_sided_local_global_fit_background=lg_two_fit_score,
        two_sided_local_global_heldout_train_background=lg_two_heldout_score,
        two_sided_local_global_validation_background=lg_two_val_score,
        two_sided_local_global_signal=lg_two_signal_score,
        two_sided_local_global_fit_pvalue=lg_two_fit_p,
        two_sided_local_global_heldout_pvalue=lg_two_heldout_p,
        two_sided_local_global_validation_pvalue=lg_two_val_p,
        two_sided_local_global_signal_pvalue=lg_two_signal_p,
    )
    if triplet_response is not None:
        np.savez_compressed(
            output_dir / "triplet_diagnostic_scores.npz",
            **{
                f"{split}_{key}": values
                for split, metrics in triplet_metric_arrays.items()
                for key, values in metrics.items()
            },
        )

    print(f"\nSaved diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
