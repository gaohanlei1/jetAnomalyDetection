#!/usr/bin/env python3
"""Post-training latent diagnostics for CMS AK8 LeJEPA semi-sup runs.

Loads ``summary.json`` + ``best_model.pth``, streams DeepNTuplizer ROOT data,
and writes overall + per-background-class Mahalanobis AUCs and ROC plots into
``<run_dir>/latent_diagnostics/``.

Example:
    python -u scripts/diagnose_lejepa_latents_cms.py \\
        plots/run-lejepa-ak8-semisup-m1.0-hbb \\
        --eval-steps 100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helpers.cms_streaming import (  # noqa: E402
    CMSIterableDataset,
    discover_cms_files_by_class,
    split_files_train_val,
)
from models.part import (  # noqa: E402
    CorruptedNegativeAugmentationConfig,
    LeJEPALossConfig,
    LeJEPASemiSupervisedTripletParticleTransformerRepresentation,
    MultiViewAugmentationConfig,
    ParticleTransformerConfig,
    SemiSupervisedLossConfig,
    TripletLossConfig,
)
from scripts.run_train_lejepa_part import (  # noqa: E402
    collate_node_tensors,
    precision_to_dtype,
)

SIGNAL_KEY_FROM_NAME = {
    "WJets": "wjets",
    "ZJets": "zjets",
    "TTbar": "ttbar",
    "Hbb": "hbb",
}
BG_KEY_FROM_NAME = {
    "QCD": "qcd",
    "WJets": "wjets",
    "ZJets": "zjets",
    "TTbar": "ttbar",
    "Hbb": "hbb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CMS AK8 latent diagnostics with per-class Mahalanobis AUCs."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory with summary.json and best_model.pth.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--ntuple-root",
        type=Path,
        default=Path("/HEP/export/home/hgao50/jet-anomaly-data/ak8-v2"),
    )
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--mahalanobis-cov-eps", type=float, default=None)
    parser.add_argument("--max-num-particles", type=int, default=128)
    parser.add_argument("--stream-val-fraction", type=float, default=0.1)
    parser.add_argument("--keep-stage-dir", action="store_true")
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


def stage_class_dir(ntuple_root: Path, stage_dir: Path, key: str) -> Path:
    dest = stage_dir / key
    dest.mkdir(parents=True, exist_ok=True)
    patterns = {
        "qcd": [
            "qcd/qcd_PT170to300_*.root",
            "qcd/qcd_PT300to470_*.root",
        ],
        "wjets": ["wjets/wjets_Wto2Q_PTQQ200_*.root"],
        "zjets": ["zjets/zjets_Zto2Q_PTQQ200_*.root"],
        "ttbar": ["ttbar/ttbar_TTto4Q_*.root"],
        "hbb": [
            "hbb/hbb_WminusH_WtoLNu_*.root",
            "hbb/hbb_WplusH_WtoLNu_*.root",
            "hbb/hbb_ZH_Zto2L_*.root",
        ],
    }[key]
    for pattern in patterns:
        for path in sorted(ntuple_root.glob(pattern)):
            link = dest / path.name
            if link.exists() or link.is_symlink():
                continue
            os.symlink(path, link)
    return dest


def read_state_dict(checkpoint_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state_dict)}")
    if state_dict and all(str(k).startswith("module.") for k in state_dict):
        state_dict = {
            str(k).removeprefix("module."): v for k, v in state_dict.items()
        }
    return state_dict


def build_model(summary: dict, device: torch.device) -> torch.nn.Module:
    features = list(summary["node_features"])
    precision = str(summary.get("precision", "bf16"))
    model_config = ParticleTransformerConfig(
        input_dim=len(features),
        embed_dim=int(summary["embed_dim"]),
        num_heads=int(summary["num_heads"]),
        num_layers=int(summary["num_layers"]),
        ffn_mult=int(summary.get("ffn_mult", 4)),
        dropout=float(summary.get("dropout", 0.1)),
        representation_dim=int(summary["representation_dim"]),
        use_pairwise_bias=bool(summary.get("use_pairwise_bias", True)),
        pairwise_hidden_dim=int(summary.get("pairwise_hidden_dim", 16)),
        pairwise_num_features=int(summary.get("pairwise_num_features", 4)),
        compute_dtype=precision_to_dtype(precision),
        use_internal_autocast=False,
        eps=float(summary.get("eps", 1e-8)),
        input_feature_names=tuple(features),
        standardized_feature_names=tuple(
            summary.get(
                "standardized_feature_names",
                ["log_pt", "d0/d0Err", "dz/dzErr"],
            )
        ),
        feature_norm_momentum=float(summary.get("feature_norm_momentum", 0.1)),
    )
    global_range = summary.get("global_drop_pt_frac_range", [0.0, 0.3])
    local_range = summary.get("local_drop_pt_frac_range", [0.3, 0.75])
    augmentation_config = MultiViewAugmentationConfig(
        num_global_views=int(summary.get("num_global_views", 2)),
        num_local_views=int(summary.get("num_local_views", 3)),
        global_drop_pt_frac_range=(float(global_range[0]), float(global_range[1])),
        local_drop_pt_frac_range=(float(local_range[0]), float(local_range[1])),
        min_nodes=int(summary.get("min_nodes", 4)),
        pt_index=features.index("pt"),
        eps=float(summary.get("eps", 1e-8)),
        pt_drop_power=float(summary.get("pt_drop_power", 1.0)),
        zero_dropped_features=not bool(summary.get("keep_dropped_features", False)),
    )
    negative_augmentation_config = CorruptedNegativeAugmentationConfig(
        num_negative_views=int(summary.get("num_negative_views", 4)),
        batch_mix_prob=float(summary.get("batch_mix_prob", 0.45)),
        pt_resample_prob=float(summary.get("pt_resample_prob", 0.25)),
        node_eta_phi_rotation_prob=float(
            summary.get("node_eta_phi_rotation_prob", 0.20)
        ),
        eta_phi_shuffle_prob=float(summary.get("eta_phi_shuffle_prob", 0.05)),
        identity_shuffle_prob=float(summary.get("identity_shuffle_prob", 0.05)),
        min_nodes=int(summary.get("min_nodes", 4)),
        eps=float(summary.get("eps", 1e-8)),
        eta_index=features.index("eta"),
        phi_index=features.index("phi"),
        pt_index=features.index("pt"),
        d0_index=features.index("d0/d0Err"),
        dz_index=features.index("dz/dzErr"),
        charge_index=features.index("charge"),
        mass_index=features.index("mass"),
        log_pt_index=features.index("log_pt"),
        pdg_start_index=features.index("pdgId_-211"),
        pdg_end_index=features.index("pdgId_211") + 1,
        corrupt_node_frac=float(summary.get("corrupt_node_frac", 0.5)),
        batch_mix_anchor_frac_min=float(
            summary.get("batch_mix_anchor_frac_min", 0.3)
        ),
        batch_mix_anchor_frac_max=float(
            summary.get("batch_mix_anchor_frac_max", 0.7)
        ),
        renormalize_pt_sum=bool(summary.get("renormalize_negative_pt_sum", True)),
        renormalize_log_pt_stats=bool(
            summary.get("renormalize_negative_log_pt_stats", True)
        ),
    )
    loss_config = LeJEPALossConfig(
        invariant_weight=float(summary.get("invariant_weight", 1.0)),
        sigreg_weight=float(summary.get("sigreg_weight", 0.05)),
        epps_pulley_num_points=int(summary.get("epps_pulley_num_points", 17)),
        num_slices=int(summary.get("num_slices", 1024)),
        normalize_representations_for_invariant=bool(
            summary.get("normalize_invariant_representations", False)
        ),
        normalize_representations_for_sigreg=bool(
            summary.get("normalize_sigreg_reps", False)
            or summary.get("normalize_sigreg_nxt", False)
        ),
    )
    triplet_loss_config = TripletLossConfig(
        triplet_weight=float(summary.get("triplet_weight", 0.1)),
        triplet_margin=float(summary.get("triplet_margin", 1.0)),
        normalize_representations_for_triplet=bool(
            summary.get("normalize_triplet_reps", False)
            or summary.get("normalize_triplet_nxt", False)
        ),
        use_global_views_as_positives=not bool(
            summary.get("use_all_views_as_triplet_positives", False)
        ),
    )
    backgrounds = list(summary["background_names"])
    model = LeJEPASemiSupervisedTripletParticleTransformerRepresentation(
        model_config=model_config,
        augmentation_config=augmentation_config,
        negative_augmentation_config=negative_augmentation_config,
        loss_config=loss_config,
        triplet_loss_config=triplet_loss_config,
        semi_supervised_config=SemiSupervisedLossConfig(
            classification_weight=float(summary.get("classification_weight", 0.05)),
            num_classes=int(
                summary.get("num_background_classes", len(backgrounds))
            ),
        ),
    )
    return model.to(device)


def make_loader(
    files_by_class: Dict[int, List[str]],
    particle_features: Sequence[str],
    *,
    batch_size: int,
    num_workers: int,
    max_num_particles: int,
    lowerpt: float,
    upperpt: float,
    seed: int,
) -> DataLoader:
    dataset = CMSIterableDataset(
        files_by_class=files_by_class,
        particle_features=particle_features,
        max_num_particles=max_num_particles,
        min_nodes=4,
        lowerpt=lowerpt,
        upperpt=upperpt,
        shuffle_files=True,
        shuffle_active_shards=4,
        infinite=True,
        seed=seed,
    )
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": True,
        "collate_fn": collate_node_tensors,
        "persistent_workers": False,
        "drop_last": True,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(**kwargs)


@torch.no_grad()
def collect_latents_and_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    precision: str,
    description: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    dtype = precision_to_dtype(precision)
    use_autocast = device.type == "cuda" and precision in {"bf16", "fp16"}
    zs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    data_iter = iter(loader)
    for _ in tqdm(range(steps), total=steps, desc=description):
        batch = next(data_iter)
        x = batch["x"].to(device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(device, non_blocking=True)
        y = batch["y"]
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=use_autocast,
        ):
            z = model(x, padding_mask=padding_mask, normalize_output=False)
        zs.append(z.detach().float().cpu())
        ys.append(y.detach().cpu())
    return torch.cat(zs, dim=0).numpy(), torch.cat(ys, dim=0).numpy()


def fit_mahalanobis(
    latents: np.ndarray, cov_eps: float
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    latents = np.asarray(latents, dtype=np.float64)
    mean = latents.mean(axis=0)
    centered = latents - mean
    cov = np.cov(centered, rowvar=False)
    cov = np.atleast_2d(cov)
    cov = cov + cov_eps * np.eye(cov.shape[0], dtype=np.float64)
    precision = np.linalg.pinv(cov)
    return mean, precision, {
        "dim": int(cov.shape[0]),
        "trace": float(np.trace(cov)),
        "eps": float(cov_eps),
    }


def mahalanobis_scores(
    latents: np.ndarray, mean: np.ndarray, precision: np.ndarray
) -> np.ndarray:
    latents = np.asarray(latents, dtype=np.float64)
    centered = latents - mean
    return np.einsum("ni,ij,nj->n", centered, precision, centered).astype(np.float64)


def auc(background: np.ndarray, signal: np.ndarray) -> float:
    y = np.concatenate(
        [np.zeros(len(background), dtype=np.int64), np.ones(len(signal), dtype=np.int64)]
    )
    return float(roc_auc_score(y, np.concatenate([background, signal])))


def score_stats(scores: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64)
    return {
        "n": int(len(scores)),
        "mean": float(np.mean(scores)) if len(scores) else float("nan"),
        "median": float(np.median(scores)) if len(scores) else float("nan"),
        "std": float(np.std(scores)) if len(scores) else float("nan"),
    }


def stratified_fit_heldout(
    y: np.ndarray, n_classes: int, fit_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    fit_parts: List[np.ndarray] = []
    held_parts: List[np.ndarray] = []
    for class_id in range(n_classes):
        idx = np.flatnonzero(y == class_id)
        rng.shuffle(idx)
        n_fit = max(1, int(round(len(idx) * fit_fraction)))
        n_fit = min(n_fit, len(idx) - 1) if len(idx) > 1 else len(idx)
        fit_parts.append(idx[:n_fit])
        held_parts.append(idx[n_fit:])
    return np.concatenate(fit_parts), np.concatenate(held_parts)


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
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pairwise_pca(
    background: np.ndarray,
    signal: np.ndarray,
    background_label: str,
    signal_label: str,
    path: Path,
    seed: int,
    max_points: int = 4000,
) -> None:
    rng = np.random.default_rng(seed)
    bg = np.asarray(background, dtype=np.float64)
    sg = np.asarray(signal, dtype=np.float64)
    if len(bg) > max_points:
        bg = bg[rng.choice(len(bg), max_points, replace=False)]
    if len(sg) > max_points:
        sg = sg[rng.choice(len(sg), max_points, replace=False)]
    combined = np.concatenate([bg, sg], axis=0)
    reduced = PCA(n_components=2, random_state=seed).fit_transform(combined)
    bg2, sg2 = reduced[: len(bg)], reduced[len(bg) :]
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(bg2[:, 0], bg2[:, 1], s=6, alpha=0.35, label=f"{background_label} (BG)")
    ax.scatter(sg2[:, 0], sg2[:, 1], s=6, alpha=0.35, label=signal_label)
    ax.set_title(f"Pairwise PCA: {background_label} vs {signal_label}")
    ax.legend(markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
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
    seed = int(summary.get("seed", 42))
    seed_everything(seed)

    backgrounds = list(summary["background_names"])
    signal_name = str(summary["signal_name"])
    signal_key = SIGNAL_KEY_FROM_NAME[signal_name]
    bg_keys = [BG_KEY_FROM_NAME[name] for name in backgrounds]
    features = list(summary["node_features"])
    steps = int(args.eval_steps or summary.get("eval_steps", 100))
    batch_size = int(args.batch_size or summary.get("batch_size", 128))
    cov_eps = float(
        args.mahalanobis_cov_eps
        if args.mahalanobis_cov_eps is not None
        else summary.get("mahalanobis_cov_eps", 1e-4)
    )
    lowerpt = float(summary.get("lowerpt", 200))
    upperpt = float(summary.get("upperpt", 400))
    precision = str(summary.get("precision", "bf16"))

    stage_dir = Path(
        os.environ.get("TMPDIR", "/tmp")
    ) / f"ak8_diag_{run_dir.name}_{os.getpid()}"
    stage_dir.mkdir(parents=True, exist_ok=True)
    try:
        bg_dirs = [
            str(stage_class_dir(args.ntuple_root, stage_dir, key)) for key in bg_keys
        ]
        sg_dir = stage_class_dir(args.ntuple_root, stage_dir, signal_key)

        files_by_class = discover_cms_files_by_class(bg_dirs)
        train_files, val_files = split_files_train_val(
            files_by_class,
            val_fraction=args.stream_val_fraction,
            seed=seed,
        )
        # Signal uses a single class id 0 for its own loader.
        signal_files = discover_cms_files_by_class([str(sg_dir)])

        print(f"Loading {checkpoint_path} on {device}")
        print(f"Backgrounds: {backgrounds}")
        print(f"Signal: {signal_name}")
        print(f"Sampling {steps} x {batch_size} events per split")

        model = build_model(summary, device)
        state_dict = read_state_dict(checkpoint_path, device)
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys in checkpoint: {load_result.unexpected_keys}"
            )
        missing = [
            k
            for k in load_result.missing_keys
            if k
            not in {
                "_feature_running_mean",
                "_feature_running_var",
                "_feature_num_batches_tracked",
            }
        ]
        if missing:
            raise RuntimeError(f"Missing keys in checkpoint: {missing}")
        if hasattr(model, "_use_frozen_feature_stats_in_eval"):
            model._use_frozen_feature_stats_in_eval = True
        model.eval()

        train_loader = make_loader(
            train_files,
            features,
            batch_size=batch_size,
            num_workers=args.num_workers,
            max_num_particles=args.max_num_particles,
            lowerpt=lowerpt,
            upperpt=upperpt,
            seed=seed + 101,
        )
        val_loader = make_loader(
            val_files,
            features,
            batch_size=batch_size,
            num_workers=args.num_workers,
            max_num_particles=args.max_num_particles,
            lowerpt=lowerpt,
            upperpt=upperpt,
            seed=seed + 202,
        )
        signal_loader = make_loader(
            signal_files,
            features,
            batch_size=batch_size,
            num_workers=args.num_workers,
            max_num_particles=args.max_num_particles,
            lowerpt=lowerpt,
            upperpt=upperpt,
            seed=seed + 303,
        )

        train_z, train_y = collect_latents_and_labels(
            model, train_loader, device, steps, precision, "Train background latents"
        )
        val_z, val_y = collect_latents_and_labels(
            model, val_loader, device, steps, precision, "Val background latents"
        )
        signal_z, _ = collect_latents_and_labels(
            model, signal_loader, device, steps, precision, "Signal latents"
        )

        n_classes = len(backgrounds)
        fit_idx, heldout_idx = stratified_fit_heldout(
            train_y, n_classes, args.fit_fraction, seed + 404
        )

        mean, precision_mat, cov_diag = fit_mahalanobis(train_z[fit_idx], cov_eps)
        fit_score = mahalanobis_scores(train_z[fit_idx], mean, precision_mat)
        heldout_score = mahalanobis_scores(train_z[heldout_idx], mean, precision_mat)
        val_score = mahalanobis_scores(val_z, mean, precision_mat)
        signal_score = mahalanobis_scores(signal_z, mean, precision_mat)

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
            f"Combined AUC fit={combined['auc_fit_subset_vs_signal']:.6f}, "
            f"held-out={combined['auc_heldout_train_vs_signal']:.6f}, "
            f"validation={combined['auc_validation_vs_signal']:.6f}"
        )

        print("\nPer-class Mahalanobis diagnostic")
        per_class: Dict[str, Dict[str, object]] = {}
        train_curves = []
        validation_curves = []
        for class_id, name in enumerate(backgrounds):
            train_mask = train_y == class_id
            val_mask = val_y == class_id
            class_train_indices = np.flatnonzero(train_mask)
            class_fit_idx = fit_idx[np.isin(fit_idx, class_train_indices)]
            class_heldout_idx = heldout_idx[np.isin(heldout_idx, class_train_indices)]
            if len(class_fit_idx) < 2 or len(val_mask.nonzero()[0]) < 1:
                print(f"Skipping {name}: insufficient events")
                continue

            class_mean, class_precision, class_cov = fit_mahalanobis(
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

            result = {
                "auc_fit_subset_vs_signal": auc(class_fit_score, class_signal_score),
                "auc_heldout_train_vs_signal": auc(
                    class_heldout_score, class_signal_score
                ),
                "auc_validation_vs_signal": auc(class_val_score, class_signal_score),
                "fit_background_scores": score_stats(class_fit_score),
                "heldout_train_background_scores": score_stats(class_heldout_score),
                "validation_background_scores": score_stats(class_val_score),
                "signal_scores": score_stats(class_signal_score),
                "covariance": class_cov,
            }
            per_class[name] = result
            plot_comparison(
                [
                    ("Held-out train", class_heldout_score, class_signal_score),
                    ("Validation", class_val_score, class_signal_score),
                ],
                f"{name}-specific Mahalanobis ROC",
                output_dir / f"04_{name.lower()}_train_validation_comparison.png",
            )
            plot_pairwise_pca(
                val_z[val_mask],
                signal_z,
                background_label=name,
                signal_label=signal_name,
                path=output_dir
                / f"01_pairwise_pca_{name.lower()}_vs_{signal_name.lower()}.png",
                seed=seed + 450 + class_id,
            )
            train_curves.append(
                (f"{name} held-out train", class_heldout_score, class_signal_score)
            )
            validation_curves.append(
                (f"{name} validation", class_val_score, class_signal_score)
            )
            print(
                f"{name}: AUC held-out={result['auc_heldout_train_vs_signal']:.6f}, "
                f"validation={result['auc_validation_vs_signal']:.6f}"
            )

        if train_curves:
            plot_comparison(
                train_curves,
                "Class-specific held-out train Mahalanobis ROC",
                output_dir / "05_per_class_train_comparison.png",
            )
        if validation_curves:
            plot_comparison(
                validation_curves,
                "Class-specific validation Mahalanobis ROC",
                output_dir / "06_per_class_validation_comparison.png",
            )

        results = {
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "ntuple_root": str(args.ntuple_root),
            "labels": {"background": backgrounds, "signal": [signal_name]},
            "counts": {
                "background_train_total": int(len(train_z)),
                "background_train_fit": int(len(fit_idx)),
                "background_train_heldout": int(len(heldout_idx)),
                "background_validation_total": int(len(val_z)),
                "signal_total": int(len(signal_z)),
            },
            "combined": combined,
            "per_class": per_class,
            "eval_steps": steps,
            "batch_size": batch_size,
        }
        with (output_dir / "diagnostic_results.json").open("w") as f:
            json.dump(jsonable(results), f, indent=2)
        print(f"\nWrote {output_dir / 'diagnostic_results.json'}")
    finally:
        if not args.keep_stage_dir and stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
