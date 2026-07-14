#!/usr/bin/env python3
"""Train a normalized autoencoder (NAE) on frozen ParticleTransformer representations.

This script is the pandas/PKL counterpart of ``run_train_lejepa_part.py``.
It reconstructs the original background/signal datasets from the backbone
``summary.json``, reproduces the seeded background train/validation split,
and computes every representation online with the frozen backbone. No latent
cache is created.

Training has two phases:

1. Vanilla AE pretraining:
       loss = mean(E(z_data))

2. NAE contrastive-divergence training:
       loss = mean(E(z_data)) - mean(E(z_model))

   ``z_model`` is obtained by a short Langevin chain initialized from the
   current background representations, following the data-initialized CD
   procedure described in the DarkCLR paper.

The reconstruction energy is

    E(z) = ||z - AE(z)||_2^2.

The best AE-pretraining checkpoint minimizes validation reconstruction energy.
The best NAE checkpoint minimizes the absolute validation energy difference,
matching the checkpoint rule used in the paper.

Example:

python -u scripts/run_train_nae_part.py \
    --backbone-dir "plots/run-lejepa-trip-part-prob-aug-fracb-augb-4n" \
    --ae-pretrain-epochs 100 \
    --nae-epochs 100 \
    --batch-size 128 \
    --learning-rate 1e-4 \
    --output-dir "plots/run-nae-trip-part"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root for local imports when launched from scripts/.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.normalized_autoencoder import (  # noqa: E402
    NormalizedAutoencoder,
    NormalizedAutoencoderConfig,
)


# -----------------------------------------------------------------------------
# Data pipeline: intentionally mirrors run_train_lejepa_part.py
# -----------------------------------------------------------------------------


def row_to_node_tensor(
    row: pd.Series,
    node_feature_names: Sequence[str],
    min_nodes: int,
) -> torch.Tensor:
    arrays: List[np.ndarray] = []
    expected_length: Optional[int] = None

    for name in node_feature_names:
        if name not in row:
            raise ValueError(f"Missing node feature: {name}")

        values = np.asarray(row[name], dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(f"Feature {name} is not one-dimensional.")

        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError(
                f"Length mismatch for feature {name}: "
                f"expected {expected_length}, got {len(values)}."
            )
        arrays.append(values)

    if expected_length is None or expected_length < min_nodes:
        raise ValueError(f"Too few nodes: {expected_length}.")

    stacked = np.column_stack(arrays).astype(np.float32)
    valid_mask = np.isfinite(stacked).all(axis=1)

    if "pt" in node_feature_names:
        pt_index = node_feature_names.index("pt")
        valid_mask &= stacked[:, pt_index] > 0

    stacked = stacked[valid_mask]
    if stacked.shape[0] < min_nodes:
        raise ValueError(f"Too few valid nodes after cleaning: {stacked.shape[0]}.")

    return torch.tensor(stacked, dtype=torch.float32)


def dataframe_to_node_tensors(
    df: pd.DataFrame,
    node_feature_names: Sequence[str],
    label: int,
    min_nodes: int,
    desc: str,
) -> Tuple[List[torch.Tensor], List[int]]:
    node_tensors: List[torch.Tensor] = []
    labels: List[int] = []

    for index in tqdm(range(len(df)), desc=desc):
        try:
            node_tensors.append(
                row_to_node_tensor(
                    row=df.iloc[index],
                    node_feature_names=node_feature_names,
                    min_nodes=min_nodes,
                )
            )
            labels.append(label)
        except Exception as exc:
            print(f"Skipping event {index} due to error: {exc}")

    return node_tensors, labels


class JetNodeDataset(Dataset):
    def __init__(self, node_tensors: Sequence[torch.Tensor], labels: Sequence[int]):
        if len(node_tensors) != len(labels):
            raise ValueError("node_tensors and labels must have the same length.")
        self.node_tensors = list(node_tensors)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.node_tensors)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.node_tensors[index], self.labels[index]


def collate_node_tensors(
    batch: Sequence[Tuple[torch.Tensor, int]],
) -> Dict[str, torch.Tensor]:
    xs, labels = zip(*batch)
    batch_size = len(xs)
    max_nodes = max(x.size(0) for x in xs)
    feature_dim = xs[0].size(1)

    padded = torch.zeros(batch_size, max_nodes, feature_dim, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool)

    for index, x in enumerate(xs):
        num_nodes = x.size(0)
        padded[index, :num_nodes] = x
        padding_mask[index, :num_nodes] = False

    return {
        "x": padded,
        "padding_mask": padding_mask,
        "y": torch.tensor(labels, dtype=torch.long),
    }


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_to_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unknown precision: {precision}")


def parse_hidden_dims(value: str) -> Tuple[int, ...]:
    dims = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("--hidden-dims must contain positive comma-separated integers.")
    return dims


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    final_lr_ratio: float,
) -> LambdaLR:
    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        if total_steps <= warmup_steps:
            return 1.0
        progress = float(step - warmup_steps) / float(total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_ratio + (1.0 - final_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def atomic_json_dump(payload: Dict[str, object], path: Path) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temp_path, path)


def safe_torch_load(path: Path, device: torch.device):
    """Load a trusted project checkpoint saved as a full model or state dict."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# -----------------------------------------------------------------------------
# Trainer
# -----------------------------------------------------------------------------


class TrainNAEPart:
    TRAIN_SPLIT = 0.8

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device(
            args.device if args.device is not None else (
                "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            )
        )
        seed_everything(args.seed)

        self.backbone_dir = Path(args.backbone_dir)
        self.backbone_summary_path = self.backbone_dir / "summary.json"
        self.backbone_checkpoint_path = self.backbone_dir / args.backbone_checkpoint

        if not self.backbone_summary_path.exists():
            raise FileNotFoundError(self.backbone_summary_path)
        if not self.backbone_checkpoint_path.exists():
            raise FileNotFoundError(self.backbone_checkpoint_path)

        with self.backbone_summary_path.open() as handle:
            self.backbone_summary: Dict[str, object] = json.load(handle)

        self.output_dir = Path(args.output_dir)
        self.roc_dir = self.output_dir / "roc"
        self.score_dir = self.output_dir / "anomaly_scores"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.roc_dir.mkdir(parents=True, exist_ok=True)
        self.score_dir.mkdir(parents=True, exist_ok=True)

        self.node_feature_names = list(self.backbone_summary["node_features"])
        self.min_nodes = int(self.backbone_summary.get("min_nodes", 4))
        self.precision = args.precision or str(self.backbone_summary.get("precision", "bf16"))
        self.autocast_dtype = precision_to_dtype(self.precision)
        self.use_autocast = self.precision in {"bf16", "fp16"} and self.device.type in {"cuda", "cpu"}

        self.background_path = Path(
            args.background or str(self.backbone_summary["background"])
        )
        self.signal_path = Path(
            args.signal or str(self.backbone_summary["signal"])
        )

        self.summary_path = self.output_dir / "summary.json"
        self.history_path = self.output_dir / "history.json"

    def load_data(self) -> None:
        print(f"Loading background from {self.background_path}")
        print(f"Loading signal from {self.signal_path}")

        background_df = pd.read_pickle(self.background_path)
        signal_df = pd.read_pickle(self.signal_path)

        if self.args.max_background_events is not None:
            background_df = background_df.head(self.args.max_background_events)
        if self.args.max_signal_events is not None:
            signal_df = signal_df.head(self.args.max_signal_events)

        bg_nodes, bg_labels = dataframe_to_node_tensors(
            background_df,
            self.node_feature_names,
            label=0,
            min_nodes=self.min_nodes,
            desc="Loading background node tensors",
        )
        sg_nodes, sg_labels = dataframe_to_node_tensors(
            signal_df,
            self.node_feature_names,
            label=1,
            min_nodes=self.min_nodes,
            desc="Loading signal node tensors",
        )

        if len(bg_nodes) < 2:
            raise ValueError("Need at least two valid background events.")
        if not sg_nodes:
            raise ValueError("No valid signal events were loaded.")

        # Reproduce the original backbone script's seeded shuffle and 80/20 split.
        split_rng = random.Random(int(self.backbone_summary.get("seed", 42)))
        bg_indices = list(range(len(bg_nodes)))
        split_rng.shuffle(bg_indices)
        bg_nodes = [bg_nodes[index] for index in bg_indices]
        bg_labels = [bg_labels[index] for index in bg_indices]

        train_size = int(self.TRAIN_SPLIT * len(bg_nodes))
        train_size = max(1, min(train_size, len(bg_nodes) - 1))

        self.bg_train_dataset = JetNodeDataset(
            bg_nodes[:train_size], bg_labels[:train_size]
        )
        self.bg_val_dataset = JetNodeDataset(
            bg_nodes[train_size:], bg_labels[train_size:]
        )
        self.sg_dataset = JetNodeDataset(sg_nodes, sg_labels)

        print(f"Background train events: {len(self.bg_train_dataset)}")
        print(f"Background val events: {len(self.bg_val_dataset)}")
        print(f"Signal events: {len(self.sg_dataset)}")
        print(f"Node features: {self.node_feature_names}")

    def make_dataloaders(self) -> None:
        common = {
            "batch_size": self.args.batch_size,
            "num_workers": self.args.num_workers,
            "pin_memory": self.args.pin_memory,
            "collate_fn": collate_node_tensors,
            "persistent_workers": self.args.num_workers > 0,
        }
        if self.args.num_workers > 0 and self.args.prefetch_factor is not None:
            common["prefetch_factor"] = self.args.prefetch_factor

        self.train_loader = DataLoader(
            self.bg_train_dataset,
            shuffle=True,
            drop_last=self.args.drop_last,
            **common,
        )
        self.val_loader = DataLoader(
            self.bg_val_dataset,
            shuffle=False,
            drop_last=False,
            **common,
        )
        self.signal_loader = DataLoader(
            self.sg_dataset,
            shuffle=False,
            drop_last=False,
            **common,
        )

    def load_backbone(self) -> None:
        checkpoint = safe_torch_load(self.backbone_checkpoint_path, self.device)

        # The old training script saves the full model object with torch.save(model).
        if isinstance(checkpoint, torch.nn.Module):
            self.backbone = checkpoint
        elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("model"), torch.nn.Module):
            self.backbone = checkpoint["model"]
        else:
            raise TypeError(
                "This script expects best_model.pth from run_train_lejepa_part.py, "
                "which saves a full torch.nn.Module. Received checkpoint type "
                f"{type(checkpoint)}."
            )

        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        self.representation_dim = int(self.backbone_summary["representation_dim"])
        print(f"Loaded frozen backbone from {self.backbone_checkpoint_path}")
        print(f"Representation dimension: {self.representation_dim}")

    def build_nae(self) -> None:
        config = NormalizedAutoencoderConfig(
            input_dim=self.representation_dim,
            hidden_dims=parse_hidden_dims(self.args.hidden_dims),
            bottleneck_dim=self.args.bottleneck_dim,
            activation=self.args.activation,
            output_activation="identity",
        )
        self.nae = NormalizedAutoencoder(config).to(self.device)
        self.num_trainable_parameters = sum(
            parameter.numel() for parameter in self.nae.parameters() if parameter.requires_grad
        )
        print(f"NAE model:\n{self.nae}")
        print(f"NAE trainable parameters: {self.num_trainable_parameters}")

    @torch.no_grad()
    def backbone_forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x"].to(self.device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(self.device, non_blocking=True)

        self.backbone.eval()
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.use_autocast,
        ):
            z = self.backbone(
                x,
                padding_mask=padding_mask,
                normalize_output=self.args.normalize_backbone_output,
            )
        return z.detach().float()

    def energy_per_sample(self, z: torch.Tensor) -> torch.Tensor:
        # Use squared L2, matching E(z)=||z-AE(z)||^2 in the paper.
        return self.nae.energy_per_sample(z.float())

    def langevin_sample(self, z_initial: torch.Tensor) -> torch.Tensor:
        """Short data-initialized Langevin chain for contrastive divergence.

        Standard overdamped Langevin update for p(z) proportional to exp(-E(z)):

            z <- z - step_size * grad_z E(z) + noise_scale * Normal(0, I)

        The returned sample is detached, so the NAE update does not backpropagate
        through the sampling trajectory.
        """
        z = z_initial.detach().float().clone()
        if self.args.negative_init_noise > 0:
            z = z + self.args.negative_init_noise * torch.randn_like(z)

        was_training = self.nae.training
        self.nae.eval()

        for _ in range(self.args.langevin_steps):
            z.requires_grad_(True)
            energy = self.energy_per_sample(z).sum()
            gradient = torch.autograd.grad(
                energy,
                z,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]

            if self.args.langevin_grad_clip is not None:
                flat = gradient.flatten(1)
                norms = flat.norm(dim=1, keepdim=True).clamp_min(1e-12)
                scale = (self.args.langevin_grad_clip / norms).clamp(max=1.0)
                gradient = gradient * scale.view(-1, *([1] * (gradient.ndim - 1)))

            with torch.no_grad():
                z = z - self.args.langevin_step_size * gradient
                if self.args.langevin_noise_scale > 0:
                    z = z + self.args.langevin_noise_scale * torch.randn_like(z)
                if self.args.langevin_clip is not None:
                    z = z.clamp(-self.args.langevin_clip, self.args.langevin_clip)

        self.nae.train(was_training)
        return z.detach()

    def run_pretrain_epoch(self, epoch: int) -> Dict[str, float]:
        self.nae.train()
        energies: List[float] = []

        pbar = tqdm(
            self.train_loader,
            desc=f"AE Pretrain {epoch}/{self.args.ae_pretrain_epochs}",
        )
        for batch in pbar:
            z_pos = self.backbone_forward(batch)
            self.optimizer.zero_grad(set_to_none=True)

            e_pos = self.energy_per_sample(z_pos).mean()
            e_pos.backward()

            if self.args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.nae.parameters(), self.args.grad_clip_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            value = float(e_pos.detach().cpu())
            energies.append(value)
            pbar.set_postfix(energy=f"{value:.5g}")

        return {"positive_energy": float(np.mean(energies))}

    def run_nae_epoch(self, epoch: int) -> Dict[str, float]:
        self.nae.train()
        positive_values: List[float] = []
        negative_values: List[float] = []
        difference_values: List[float] = []

        pbar = tqdm(
            self.train_loader,
            desc=f"NAE/CD {epoch}/{self.args.nae_epochs}",
        )
        for batch in pbar:
            z_pos = self.backbone_forward(batch)
            z_neg = self.langevin_sample(z_pos)

            self.nae.train()
            self.optimizer.zero_grad(set_to_none=True)
            e_pos = self.energy_per_sample(z_pos).mean()
            e_neg = self.energy_per_sample(z_neg).mean()
            difference = e_pos - e_neg
            difference.backward()

            if self.args.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.nae.parameters(), self.args.grad_clip_norm
                )

            self.optimizer.step()
            self.scheduler.step()

            pos_value = float(e_pos.detach().cpu())
            neg_value = float(e_neg.detach().cpu())
            diff_value = float(difference.detach().cpu())
            positive_values.append(pos_value)
            negative_values.append(neg_value)
            difference_values.append(diff_value)
            pbar.set_postfix(
                Epos=f"{pos_value:.4g}",
                Eneg=f"{neg_value:.4g}",
                diff=f"{diff_value:.4g}",
            )

        return {
            "positive_energy": float(np.mean(positive_values)),
            "negative_energy": float(np.mean(negative_values)),
            "energy_difference": float(np.mean(difference_values)),
        }

    def evaluate_background_energy(self, loader: DataLoader, with_negative: bool) -> Dict[str, float]:
        self.nae.eval()
        positive_values: List[float] = []
        negative_values: List[float] = []

        for batch in tqdm(loader, desc="Validation", leave=False):
            z_pos = self.backbone_forward(batch)
            if with_negative:
                with torch.enable_grad():
                    z_neg = self.langevin_sample(z_pos)
            else:
                z_neg = None

            self.nae.eval()
            with torch.no_grad():
                positive_values.extend(
                    self.energy_per_sample(z_pos).detach().cpu().numpy().tolist()
                )
                if z_neg is not None:
                    negative_values.extend(
                        self.energy_per_sample(z_neg).detach().cpu().numpy().tolist()
                    )

        result = {"positive_energy": float(np.mean(positive_values))}
        if negative_values:
            result["negative_energy"] = float(np.mean(negative_values))
            result["energy_difference"] = (
                result["positive_energy"] - result["negative_energy"]
            )
        return result

    @torch.no_grad()
    def collect_anomaly_scores(self, loader: DataLoader) -> np.ndarray:
        self.nae.eval()
        scores: List[np.ndarray] = []
        for batch in tqdm(loader, desc="Collecting NAE scores", leave=False):
            z = self.backbone_forward(batch)
            # MSE and squared-L2 differ only by the constant representation_dim,
            # so they produce exactly the same ROC ordering. Report MSE for scale.
            score = self.nae.mse_per_sample(z.float())
            scores.append(score.detach().cpu().numpy())
        return np.concatenate(scores, axis=0)

    @staticmethod
    def compute_auc(background_scores: np.ndarray, signal_scores: np.ndarray) -> float:
        labels = np.concatenate(
            [
                np.zeros(len(background_scores), dtype=np.int64),
                np.ones(len(signal_scores), dtype=np.int64),
            ]
        )
        scores = np.concatenate([background_scores, signal_scores])
        return float(roc_auc_score(labels, scores))

    def evaluate_roc(self, phase: str, epoch: int) -> Dict[str, float]:
        train_scores = self.collect_anomaly_scores(self.train_eval_loader)
        val_scores = self.collect_anomaly_scores(self.val_loader)
        signal_scores = self.collect_anomaly_scores(self.signal_loader)

        train_auc = self.compute_auc(train_scores, signal_scores)
        val_auc = self.compute_auc(val_scores, signal_scores)

        tag = f"{phase}_epoch_{epoch:04d}"
        self.plot_score_distribution(
            val_scores,
            signal_scores,
            self.score_dir / f"scores_{tag}.png",
            title=f"NAE Reconstruction Score: {phase}, Epoch {epoch}",
        )
        self.plot_roc_curve(
            train_scores,
            val_scores,
            signal_scores,
            self.roc_dir / f"roc_{tag}.png",
            title=f"NAE ROC: {phase}, Epoch {epoch}",
        )

        np.savez_compressed(
            self.score_dir / f"scores_{tag}.npz",
            background_train=train_scores,
            background_val=val_scores,
            signal=signal_scores,
        )

        metrics = {
            "phase": phase,
            "epoch": int(epoch),
            "auc_bgtrain_vs_signal": train_auc,
            "auc_bgval_vs_signal": val_auc,
            "background_train_score_mean": float(np.mean(train_scores)),
            "background_val_score_mean": float(np.mean(val_scores)),
            "signal_score_mean": float(np.mean(signal_scores)),
        }
        with (self.roc_dir / f"metrics_{tag}.json").open("w") as handle:
            json.dump(metrics, handle, indent=2)

        print(f"NAE AUC, background train vs signal: {train_auc:.6f}")
        print(f"NAE AUC, background val vs signal: {val_auc:.6f}")
        return metrics

    @staticmethod
    def plot_score_distribution(
        background_scores: np.ndarray,
        signal_scores: np.ndarray,
        path: Path,
        title: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 5))
        combined = np.concatenate([background_scores, signal_scores])
        finite = combined[np.isfinite(combined)]
        if finite.size == 0:
            plt.close(fig)
            return
        low, high = np.quantile(finite, [0.001, 0.995])
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            low, high = float(finite.min()), float(finite.max() + 1e-8)
        bins = np.linspace(low, high, 80)
        ax.hist(background_scores, bins=bins, density=True, histtype="step", label="Background Val")
        ax.hist(signal_scores, bins=bins, density=True, histtype="step", label="Signal")
        ax.set_xlabel("NAE reconstruction MSE")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    @staticmethod
    def plot_roc_curve(
        train_scores: np.ndarray,
        val_scores: np.ndarray,
        signal_scores: np.ndarray,
        path: Path,
        title: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        for background_scores, label in (
            (train_scores, "Background Train vs Signal"),
            (val_scores, "Background Val vs Signal"),
        ):
            y_true = np.concatenate(
                [np.zeros(len(background_scores)), np.ones(len(signal_scores))]
            )
            y_score = np.concatenate([background_scores, signal_scores])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            auc = roc_auc_score(y_true, y_score)
            ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.legend()
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    def plot_progress(self, history: Dict[str, List[float]]) -> None:
        if not history["train_positive_energy"]:
            return

        epochs = np.arange(1, len(history["train_positive_energy"]) + 1)
        phase_boundary = self.args.ae_pretrain_epochs

        fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

        axes[0].plot(epochs, history["train_positive_energy"], label="Train")
        axes[0].plot(epochs, history["val_positive_energy"], label="Validation")
        axes[0].set_ylabel("Positive energy")
        axes[0].set_yscale("log")
        axes[0].legend()

        axes[1].plot(epochs, history["train_negative_energy"], label="Train")
        axes[1].plot(epochs, history["val_negative_energy"], label="Validation")
        axes[1].set_ylabel("Negative energy")
        axes[1].legend()

        axes[2].plot(epochs, history["train_energy_difference"], label="Train")
        axes[2].plot(epochs, history["val_energy_difference"], label="Validation")
        axes[2].axhline(0.0, linestyle="--", linewidth=1)
        axes[2].set_ylabel("E(data) - E(model)")
        axes[2].legend()

        roc_epochs = np.asarray(history["roc_total_epoch"], dtype=np.int64)
        if roc_epochs.size > 0:
            axes[3].plot(
                roc_epochs,
                history["auc_bgtrain_vs_signal"],
                marker="o",
                label="Background Train vs Signal",
            )
            axes[3].plot(
                roc_epochs,
                history["auc_bgval_vs_signal"],
                marker="o",
                label="Background Val vs Signal",
            )
        axes[3].set_ylabel("ROC AUC")
        axes[3].set_ylim(0, 1)
        axes[3].legend()

        if 0 < phase_boundary < len(epochs):
            for ax in axes:
                ax.axvline(phase_boundary + 0.5, linestyle="--", linewidth=1)

        for ax in axes:
            ax.grid(False)
        axes[-1].set_xlabel("Total epoch")
        fig.suptitle("Normalized Autoencoder Training Progress")
        fig.tight_layout()
        fig.savefig(self.output_dir / "training_progress.png")
        plt.close(fig)

    def save_checkpoint(self, path: Path, phase: str, epoch: int, metrics: Dict[str, float]) -> None:
        torch.save(
            {
                "model_state_dict": self.nae.state_dict(),
                "model_config": {
                    "input_dim": self.representation_dim,
                    "hidden_dims": list(parse_hidden_dims(self.args.hidden_dims)),
                    "bottleneck_dim": self.args.bottleneck_dim,
                    "activation": self.args.activation,
                    "output_activation": "identity",
                },
                "phase": phase,
                "epoch": int(epoch),
                "metrics": metrics,
                "backbone_dir": str(self.backbone_dir),
                "backbone_checkpoint": self.args.backbone_checkpoint,
            },
            path,
        )

    def train(self) -> None:
        self.load_data()
        self.make_dataloaders()
        self.load_backbone()
        self.build_nae()

        # Fixed, non-shuffled train subset loader for fair epoch-level ROC.
        train_eval_size = min(self.args.max_train_eval_events, len(self.bg_train_dataset))
        train_eval_dataset = torch.utils.data.Subset(
            self.bg_train_dataset, range(train_eval_size)
        )
        loader_kwargs = {
            "batch_size": self.args.batch_size,
            "shuffle": False,
            "num_workers": self.args.num_workers,
            "pin_memory": self.args.pin_memory,
            "collate_fn": collate_node_tensors,
            "persistent_workers": self.args.num_workers > 0,
        }
        if self.args.num_workers > 0 and self.args.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = self.args.prefetch_factor
        self.train_eval_loader = DataLoader(train_eval_dataset, **loader_kwargs)

        total_epochs = self.args.ae_pretrain_epochs + self.args.nae_epochs
        total_steps = total_epochs * max(1, len(self.train_loader))
        warmup_steps = self.args.warmup_epochs * max(1, len(self.train_loader))

        self.optimizer = torch.optim.Adam(
            self.nae.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = make_warmup_cosine_scheduler(
            self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            final_lr_ratio=self.args.final_lr_ratio,
        )

        history: Dict[str, List[float]] = {
            "train_positive_energy": [],
            "val_positive_energy": [],
            "train_negative_energy": [],
            "val_negative_energy": [],
            "train_energy_difference": [],
            "val_energy_difference": [],
            "roc_total_epoch": [],
            "auc_bgtrain_vs_signal": [],
            "auc_bgval_vs_signal": [],
        }

        run_summary: Dict[str, object] = {
            "status": "training",
            "backbone_dir": str(self.backbone_dir),
            "backbone_checkpoint": self.args.backbone_checkpoint,
            "background": str(self.background_path),
            "signal": str(self.signal_path),
            "node_features": self.node_feature_names,
            "background_train_events": len(self.bg_train_dataset),
            "background_val_events": len(self.bg_val_dataset),
            "signal_events": len(self.sg_dataset),
            "representation_dim": self.representation_dim,
            "hidden_dims": list(parse_hidden_dims(self.args.hidden_dims)),
            "bottleneck_dim": self.args.bottleneck_dim,
            "activation": self.args.activation,
            "num_trainable_parameters": self.num_trainable_parameters,
            "ae_pretrain_epochs": self.args.ae_pretrain_epochs,
            "nae_epochs": self.args.nae_epochs,
            "learning_rate": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
            "batch_size": self.args.batch_size,
            "langevin_steps": self.args.langevin_steps,
            "langevin_step_size": self.args.langevin_step_size,
            "langevin_noise_scale": self.args.langevin_noise_scale,
            "negative_init_noise": self.args.negative_init_noise,
            "seed": self.args.seed,
            "device": str(self.device),
            "current_phase": None,
            "current_epoch": 0,
            "best_pretrain_val_energy": None,
            "best_nae_abs_val_energy_difference": None,
        }
        atomic_json_dump(run_summary, self.summary_path)

        best_pretrain_energy = float("inf")
        best_abs_nae_difference = float("inf")
        total_epoch = 0

        # Phase 1: ordinary AE pretraining.
        for phase_epoch in range(1, self.args.ae_pretrain_epochs + 1):
            total_epoch += 1
            train_metrics = self.run_pretrain_epoch(phase_epoch)
            val_metrics = self.evaluate_background_energy(self.val_loader, with_negative=False)

            history["train_positive_energy"].append(train_metrics["positive_energy"])
            history["val_positive_energy"].append(val_metrics["positive_energy"])
            history["train_negative_energy"].append(np.nan)
            history["val_negative_energy"].append(np.nan)
            history["train_energy_difference"].append(np.nan)
            history["val_energy_difference"].append(np.nan)

            if val_metrics["positive_energy"] < best_pretrain_energy:
                best_pretrain_energy = val_metrics["positive_energy"]
                self.save_checkpoint(
                    self.output_dir / "best_pretrain_model.pth",
                    phase="ae_pretrain",
                    epoch=phase_epoch,
                    metrics=val_metrics,
                )

            if self.args.roc_eval_every > 0 and (
                phase_epoch % self.args.roc_eval_every == 0
                or phase_epoch == self.args.ae_pretrain_epochs
            ):
                roc_metrics = self.evaluate_roc("ae_pretrain", phase_epoch)
                history["roc_total_epoch"].append(total_epoch)
                history["auc_bgtrain_vs_signal"].append(
                    roc_metrics["auc_bgtrain_vs_signal"]
                )
                history["auc_bgval_vs_signal"].append(
                    roc_metrics["auc_bgval_vs_signal"]
                )

            run_summary.update(
                current_phase="ae_pretrain",
                current_epoch=phase_epoch,
                total_completed_epochs=total_epoch,
                best_pretrain_val_energy=best_pretrain_energy,
                latest_train_metrics=train_metrics,
                latest_val_metrics=val_metrics,
            )
            atomic_json_dump(run_summary, self.summary_path)
            atomic_json_dump(history, self.history_path)
            self.plot_progress(history)
            print(f"AE pretrain epoch {phase_epoch}: train={train_metrics}, val={val_metrics}")

        # Start NAE from the best reconstruction checkpoint, not necessarily the last.
        best_pretrain_path = self.output_dir / "best_pretrain_model.pth"
        if best_pretrain_path.exists():
            checkpoint = safe_torch_load(best_pretrain_path, self.device)
            self.nae.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded best AE-pretraining checkpoint from {best_pretrain_path}")

        # Phase 2: NAE contrastive-divergence training.
        for phase_epoch in range(1, self.args.nae_epochs + 1):
            total_epoch += 1
            train_metrics = self.run_nae_epoch(phase_epoch)
            val_metrics = self.evaluate_background_energy(self.val_loader, with_negative=True)

            history["train_positive_energy"].append(train_metrics["positive_energy"])
            history["val_positive_energy"].append(val_metrics["positive_energy"])
            history["train_negative_energy"].append(train_metrics["negative_energy"])
            history["val_negative_energy"].append(val_metrics["negative_energy"])
            history["train_energy_difference"].append(train_metrics["energy_difference"])
            history["val_energy_difference"].append(val_metrics["energy_difference"])

            abs_difference = abs(val_metrics["energy_difference"])
            if abs_difference < best_abs_nae_difference:
                best_abs_nae_difference = abs_difference
                self.save_checkpoint(
                    self.output_dir / "best_model.pth",
                    phase="nae",
                    epoch=phase_epoch,
                    metrics=val_metrics,
                )
                print(
                    "Saved new best NAE checkpoint: "
                    f"|validation energy difference|={abs_difference:.6g}"
                )

            self.save_checkpoint(
                self.output_dir / "last_model.pth",
                phase="nae",
                epoch=phase_epoch,
                metrics=val_metrics,
            )

            if self.args.roc_eval_every > 0 and (
                phase_epoch % self.args.roc_eval_every == 0
                or phase_epoch == self.args.nae_epochs
            ):
                roc_metrics = self.evaluate_roc("nae", phase_epoch)
                history["roc_total_epoch"].append(total_epoch)
                history["auc_bgtrain_vs_signal"].append(
                    roc_metrics["auc_bgtrain_vs_signal"]
                )
                history["auc_bgval_vs_signal"].append(
                    roc_metrics["auc_bgval_vs_signal"]
                )

            run_summary.update(
                current_phase="nae",
                current_epoch=phase_epoch,
                total_completed_epochs=total_epoch,
                best_nae_abs_val_energy_difference=best_abs_nae_difference,
                latest_train_metrics=train_metrics,
                latest_val_metrics=val_metrics,
            )
            atomic_json_dump(run_summary, self.summary_path)
            atomic_json_dump(history, self.history_path)
            self.plot_progress(history)
            print(f"NAE epoch {phase_epoch}: train={train_metrics}, val={val_metrics}")

        # Evaluate the selected best NAE checkpoint once more.
        best_model_path = self.output_dir / "best_model.pth"
        if best_model_path.exists():
            checkpoint = safe_torch_load(best_model_path, self.device)
            self.nae.load_state_dict(checkpoint["model_state_dict"])
            final_metrics = self.evaluate_roc("best_nae", int(checkpoint["epoch"]))
        else:
            final_metrics = {}

        run_summary.update(
            status="completed",
            current_phase="completed",
            total_completed_epochs=total_epoch,
            best_pretrain_val_energy=best_pretrain_energy,
            best_nae_abs_val_energy_difference=best_abs_nae_difference,
            final_best_model_roc=final_metrics,
        )
        atomic_json_dump(run_summary, self.summary_path)
        atomic_json_dump(history, self.history_path)
        self.plot_progress(history)
        print(f"Completed NAE training. Outputs saved to {self.output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a normalized autoencoder on frozen LeJEPA representations."
    )

    parser.add_argument(
        "--backbone-dir",
        type=str,
        default="plots/run-lejepa-trip-part-prob-aug-fracb-augb-4n",
        help="Directory containing backbone summary.json and best_model.pth.",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default="best_model.pth",
    )
    parser.add_argument("--background", type=str, default=None)
    parser.add_argument("--signal", type=str, default=None)

    parser.add_argument("--hidden-dims", type=str, default="64,32,16,8")
    parser.add_argument("--bottleneck-dim", type=int, default=3)
    parser.add_argument(
        "--activation",
        choices=["relu", "leaky_relu", "silu", "gelu", "tanh"],
        default="relu",
    )

    parser.add_argument("--ae-pretrain-epochs", type=int, default=100)
    parser.add_argument("--nae-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--final-lr-ratio", type=float, default=1e-2)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)

    parser.add_argument("--langevin-steps", type=int, default=20)
    parser.add_argument("--langevin-step-size", type=float, default=1e-2)
    parser.add_argument("--langevin-noise-scale", type=float, default=1e-2)
    parser.add_argument("--negative-init-noise", type=float, default=1e-2)
    parser.add_argument("--langevin-grad-clip", type=float, default=10.0)
    parser.add_argument("--langevin-clip", type=float, default=None)

    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default=None,
        help="Backbone forward precision. Defaults to backbone summary.json.",
    )
    parser.add_argument(
        "--normalize-backbone-output",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="L2-normalize backbone representations before NAE training.",
    )

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
    )
    parser.add_argument(
        "--drop-last",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--roc-eval-every", type=int, default=10)
    parser.add_argument("--max-train-eval-events", type=int, default=5000)
    parser.add_argument("--max-background-events", type=int, default=None)
    parser.add_argument("--max-signal-events", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="plots/run-nae-trip-part")

    return parser


if __name__ == "__main__":
    trainer = TrainNAEPart(build_parser().parse_args())
    trainer.train()
