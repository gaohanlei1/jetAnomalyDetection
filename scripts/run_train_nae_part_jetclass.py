#!/usr/bin/env python3
"""Train a representation-space normalized autoencoder on a frozen LeJEPA backbone.

The data are streamed online from JetClass ROOT shards.  ParticleTransformer
representations are recomputed on every batch; no latent cache is created.

Training has two phases:

1. Vanilla autoencoder pretraining for ``--ae-pretrain-epochs`` epochs.
2. NAE contrastive-divergence training for ``--nae-epochs`` epochs, using
   short Langevin chains and

       loss = mean(E_positive) - mean(E_negative).

The NAE checkpoint is selected by the validation energy difference whose
absolute value is closest to zero, matching the DarkCLR paper.

Example:

python -u scripts/run_train_nae_part_jetclass.py \
    --backbone-dir plots/run-lejepa-semi-sup-triplet-jetclass-ddp \
    --ae-pretrain-epochs 100 \
    --nae-epochs 100 \
    --batch-size 256 \
    --steps-per-epoch 1000 \
    --val-steps 50 \
    --eval-steps 50 \
    --output-dir plots/run-nae-lejepa-triplet

DDP example:

torchrun --standalone --nproc-per-node=2 \
    scripts/run_train_nae_part_jetclass.py \
    --backbone-dir plots/run-lejepa-semi-sup-triplet-jetclass-ddp \
    --batch-size 256 \
    --steps-per-epoch 1000 \
    --val-steps 50 \
    --eval-steps 50 \
    --output-dir plots/run-nae-lejepa-triplet-ddp
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, PROJECT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from run_train_lejepa_part_jetclass import (
        JetClassIterableDataset,
        collate_jetclass_tensors,
    )
except ImportError:
    from scripts.run_train_lejepa_part_jetclass import (
        JetClassIterableDataset,
        collate_jetclass_tensors,
    )

from models.normalized_autoencoder import (
    NormalizedAutoencoder,
    NormalizedAutoencoderConfig,
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


def parse_csv_list(value: str) -> List[str]:
    result = [item.strip() for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("Expected a non-empty comma-separated list.")
    return result


def parse_int_list(value: str) -> Tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise ValueError("Expected positive comma-separated integer dimensions.")
    return result


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
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(float(progress), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_ratio + (1.0 - final_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def read_state_dict(path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state_dict)}")

    prefixes = ("module.model.", "module.", "model.")
    for prefix in prefixes:
        if state_dict and all(str(key).startswith(prefix) for key in state_dict):
            state_dict = {
                str(key).removeprefix(prefix): value
                for key, value in state_dict.items()
            }
            break
    return state_dict


def build_backbone_from_summary(
    summary: Dict[str, object],
    device: torch.device,
) -> nn.Module:
    features = list(summary["particle_features"])
    model_name = str(summary["model"])

    model_config = ParticleTransformerConfig(
        input_dim=len(features),
        embed_dim=int(summary["embed_dim"]),
        num_heads=int(summary["num_heads"]),
        num_layers=int(summary["num_layers"]),
        ffn_mult=int(summary.get("ffn_mult", 4)),
        dropout=float(summary.get("dropout", 0.0)),
        representation_dim=int(summary["representation_dim"]),
        use_pairwise_bias=bool(summary.get("use_pairwise_bias", True)),
        pairwise_hidden_dim=int(summary.get("pairwise_hidden_dim", 64)),
        pairwise_num_features=int(summary.get("pairwise_num_features", 4)),
        compute_dtype=precision_to_dtype(str(summary.get("precision", "bf16"))),
        use_internal_autocast=False,
        eps=float(summary.get("eps", 1e-8)),
    )

    augmentation_config = MultiViewAugmentationConfig(
        num_global_views=int(summary.get("num_global_views", 2)),
        num_local_views=int(summary.get("num_local_views", 6)),
        global_drop_pt_frac_range=tuple(
            summary.get("global_drop_pt_frac_range", [0.0, 0.5])
        ),
        local_drop_pt_frac_range=tuple(
            summary.get("local_drop_pt_frac_range", [0.5, 0.95])
        ),
        min_nodes=int(summary.get("min_nodes", 4)),
        px_index=features.index("part_px"),
        py_index=features.index("part_py"),
        pz_index=features.index("part_pz"),
        energy_index=features.index("part_energy"),
        eta_index=features.index("part_eta"),
        phi_index=features.index("part_phi"),
        pt_index=features.index("part_pt"),
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

    negative_config = None
    triplet_config = None
    if model_name in {"triplet", "semi-sup-triplet"}:
        negative_config = CorruptedNegativeAugmentationConfig(
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
            eta_index=features.index("part_eta"),
            phi_index=features.index("part_phi"),
            deta_index=features.index("part_deta"),
            dphi_index=features.index("part_dphi"),
            pt_index=features.index("part_pt"),
            d0_index=features.index("part_d0val"),
            dz_index=features.index("part_dzval"),
            charge_index=features.index("part_charge"),
            identity_start_index=features.index("part_isChargedHadron"),
            identity_end_index=features.index("part_isMuon") + 1,
            corrupt_node_frac=float(summary.get("corrupt_node_frac", 1.0)),
            batch_mix_anchor_frac_min=float(
                summary.get("batch_mix_anchor_frac_min", 0.1)
            ),
            batch_mix_anchor_frac_max=float(
                summary.get("batch_mix_anchor_frac_max", 0.9)
            ),
            renormalize_pt_sum=bool(
                summary.get("renormalize_negative_pt_sum", True)
            ),
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

    if model_name == "lejepa":
        model = LeJEPAParticleTransformerRepresentation(
            model_config=model_config,
            augmentation_config=augmentation_config,
            loss_config=loss_config,
        )
    elif model_name == "triplet":
        model = LeJEPATripletParticleTransformerRepresentation(
            model_config=model_config,
            augmentation_config=augmentation_config,
            negative_augmentation_config=negative_config,
            loss_config=loss_config,
            triplet_loss_config=triplet_config,
        )
    elif model_name == "semi-sup":
        model = LeJEPASemiSupervisedParticleTransformerRepresentation(
            model_config=model_config,
            augmentation_config=augmentation_config,
            loss_config=loss_config,
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
    elif model_name == "semi-sup-triplet":
        model = LeJEPASemiSupervisedTripletParticleTransformerRepresentation(
            model_config=model_config,
            augmentation_config=augmentation_config,
            negative_augmentation_config=negative_config,
            loss_config=loss_config,
            triplet_loss_config=triplet_config,
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
    else:
        raise ValueError(f"Unsupported backbone model in summary.json: {model_name!r}")

    return model.to(device)


class TrainNAE:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if self.distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("DDP mode currently requires CUDA.")
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            self.device = torch.device("cuda", self.local_rank)
        else:
            if args.device == "auto":
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(args.device)

        self.is_main = self.rank == 0
        if args.batch_size % self.world_size != 0:
            raise ValueError(
                f"--batch-size={args.batch_size} must be divisible by world_size={self.world_size}."
            )
        self.per_rank_batch_size = args.batch_size // self.world_size

        seed_everything(args.seed + self.rank)
        self.backbone_dir = Path(args.backbone_dir)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.roc_dir = self.output_dir / "roc"
        self.score_dir = self.output_dir / "anomaly_scores"
        self.roc_dir.mkdir(exist_ok=True)
        self.score_dir.mkdir(exist_ok=True)

        summary_path = self.backbone_dir / "summary.json"
        with summary_path.open() as file:
            self.backbone_summary = json.load(file)

        self.dataset_root = Path(
            args.dataset_root or str(self.backbone_summary["dataset_root"])
        )
        self.background_labels = list(self.backbone_summary["background_labels"])
        self.signal_labels = (
            parse_csv_list(args.signal_labels)
            if args.signal_labels is not None
            else list(self.backbone_summary["signal_labels"])
        )
        self.particle_features = list(self.backbone_summary["particle_features"])
        self.representation_dim = int(self.backbone_summary["representation_dim"])

        self._build_backbone()
        self._build_nae()
        self._build_datasets_and_loaders()

    def _build_backbone(self) -> None:
        checkpoint = (
            Path(self.args.backbone_checkpoint)
            if self.args.backbone_checkpoint
            else self.backbone_dir / "best_model.pth"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing backbone checkpoint: {checkpoint}")

        self.backbone = build_backbone_from_summary(
            self.backbone_summary,
            self.device,
        )
        state_dict = read_state_dict(checkpoint, self.device)
        self.backbone.load_state_dict(state_dict, strict=True)
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        if self.is_main:
            print(f"Loaded frozen backbone from {checkpoint}")
            print(f"Backbone model: {self.backbone_summary['model']}")
            print(f"Representation dimension: {self.representation_dim}")

    def _build_nae(self) -> None:
        config = NormalizedAutoencoderConfig(
            input_dim=self.representation_dim,
            hidden_dims=self.args.hidden_dims,
            bottleneck_dim=self.args.bottleneck_dim,
            activation=self.args.activation,
        )
        core_model = NormalizedAutoencoder(config).to(self.device)
        self.nae_core = core_model
        if self.distributed:
            self.nae = DDP(
                core_model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
            )
        else:
            self.nae = core_model

        if self.is_main:
            num_parameters = sum(parameter.numel() for parameter in core_model.parameters())
            print(f"NAE model:\n{core_model}")
            print(f"NAE trainable parameters: {num_parameters}")

    def _build_datasets_and_loaders(self) -> None:
        train_dir = self.dataset_root / "train_100M"
        val_dir = self.dataset_root / "val_5M"
        test_dir = self.dataset_root / "test_20M"
        for directory in (train_dir, val_dir, test_dir):
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing JetClass split directory: {directory}")

        common = dict(
            particle_features=self.particle_features,
            max_num_particles=self.args.max_num_particles,
            shuffle_active_shards=self.args.shuffle_active_shards,
            infinite=True,
            shuffle_files=True,
            rank=self.rank,
            world_size=self.world_size,
        )
        self.train_dataset = JetClassIterableDataset(
            split_dir=str(train_dir),
            labels_to_load=self.background_labels,
            max_events=self.args.max_train_events,
            seed=self.args.seed,
            **common,
        )
        self.train_eval_dataset = JetClassIterableDataset(
            split_dir=str(train_dir),
            labels_to_load=self.background_labels,
            max_events=self.args.max_val_events,
            seed=self.args.seed + 11,
            **common,
        )
        self.val_dataset = JetClassIterableDataset(
            split_dir=str(val_dir),
            labels_to_load=self.background_labels,
            max_events=self.args.max_val_events,
            seed=self.args.seed + 23,
            **common,
        )
        self.signal_dataset = JetClassIterableDataset(
            split_dir=str(test_dir),
            labels_to_load=self.signal_labels,
            max_events=self.args.max_test_signal_events,
            seed=self.args.seed + 37,
            **common,
        )

        train_kwargs = dict(
            batch_size=self.per_rank_batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=collate_jetclass_tensors,
            persistent_workers=self.args.num_workers > 0,
            drop_last=True,
        )
        if self.args.num_workers > 0:
            train_kwargs["prefetch_factor"] = self.args.prefetch_factor

        eval_workers = 1 if self.args.num_workers > 0 else 0
        eval_kwargs = dict(
            batch_size=self.per_rank_batch_size,
            num_workers=eval_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=collate_jetclass_tensors,
            persistent_workers=eval_workers > 0,
            drop_last=True,
        )
        if eval_workers > 0:
            eval_kwargs["prefetch_factor"] = 1

        self.train_loader = DataLoader(self.train_dataset, **train_kwargs)
        self.train_eval_loader = DataLoader(self.train_eval_dataset, **eval_kwargs)
        self.val_loader = DataLoader(self.val_dataset, **eval_kwargs)
        self.signal_loader = DataLoader(self.signal_dataset, **eval_kwargs)

    def _backbone_representation(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = batch["x_particles"].to(self.device, non_blocking=True)
        padding_mask = batch["padding_mask"].to(self.device, non_blocking=True)
        dtype = precision_to_dtype(self.args.backbone_precision)
        use_autocast = self.device.type == "cuda" and self.args.backbone_precision != "fp32"

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=dtype,
                enabled=use_autocast,
            ):
                cls = self.backbone(x, padding_mask=padding_mask)
                z = self.backbone.representation_head(cls)
        return z.float()

    def _all_reduce_mean(self, value: torch.Tensor) -> float:
        result = value.detach().float().clone()
        if self.distributed:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
            result /= self.world_size
        return float(result.cpu())

    def _all_gather_rows(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach().to(self.device).contiguous()
        if not self.distributed:
            return tensor.cpu()

        local_size = torch.tensor([tensor.size(0)], device=self.device, dtype=torch.long)
        sizes = [torch.zeros_like(local_size) for _ in range(self.world_size)]
        dist.all_gather(sizes, local_size)
        sizes_int = [int(size.item()) for size in sizes]
        max_size = max(sizes_int)
        if tensor.size(0) < max_size:
            padding = torch.zeros(
                (max_size - tensor.size(0), *tensor.shape[1:]),
                device=self.device,
                dtype=tensor.dtype,
            )
            tensor = torch.cat([tensor, padding], dim=0)
        gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, tensor)
        return torch.cat(
            [rank_tensor[:size] for rank_tensor, size in zip(gathered, sizes_int)],
            dim=0,
        ).cpu()

    def _langevin_negative(self, z_positive: torch.Tensor) -> torch.Tensor:
        """Short CD chain initialized near the current background representation."""
        z = (
            z_positive.detach()
            + self.args.negative_init_noise * torch.randn_like(z_positive)
        )

        for _ in range(self.args.langevin_steps):
            z = z.detach().requires_grad_(True)
            energy = self.nae_core.energy_per_sample(z).sum()
            gradient = torch.autograd.grad(
                energy,
                z,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
            with torch.no_grad():
                z = (
                    z
                    - self.args.langevin_step_size * gradient
                    + self.args.langevin_noise_scale * torch.randn_like(z)
                )
                if self.args.langevin_clip is not None:
                    z = z.clamp(
                        min=-self.args.langevin_clip,
                        max=self.args.langevin_clip,
                    )
        return z.detach()

    def _run_epoch(
        self,
        loader: DataLoader,
        steps: int,
        optimizer: Optional[torch.optim.Optimizer],
        phase: str,
        epoch: int,
        scheduler: Optional[LambdaLR] = None,
    ) -> Dict[str, float]:
        training = optimizer is not None
        self.nae.train(training)
        iterator = iter(loader)
        values: Dict[str, List[float]] = {
            "loss": [],
            "positive_energy": [],
            "negative_energy": [],
            "energy_difference": [],
        }

        pbar = tqdm(
            range(steps),
            desc=f"{phase} Epoch {epoch}",
            disable=not self.is_main,
        )

        for _ in pbar:
            batch = next(iterator)
            z_positive = self._backbone_representation(batch)

            if phase == "ae_pretrain":
                context = nullcontext() if training else torch.no_grad()
                with context:
                    reconstruction = self.nae(z_positive)
                    positive_energy = (reconstruction - z_positive).pow(2).sum(dim=-1).mean()
                    loss = positive_energy
                    negative_energy = torch.zeros_like(positive_energy)
            else:
                # Langevin sampling requires gradients with respect to z even in validation.
                with torch.enable_grad():
                    z_negative = self._langevin_negative(z_positive)

                context = nullcontext() if training else torch.no_grad()
                with context:
                    positive_reconstruction = self.nae(z_positive)
                    negative_reconstruction = self.nae(z_negative)
                    positive_energy = (
                        positive_reconstruction - z_positive
                    ).pow(2).sum(dim=-1).mean()
                    negative_energy = (
                        negative_reconstruction - z_negative
                    ).pow(2).sum(dim=-1).mean()
                    loss = (
                        positive_energy
                        - self.args.negative_energy_weight * negative_energy
                    )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.nae.parameters(), self.args.grad_clip_norm
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            energy_difference = positive_energy - negative_energy
            step_metrics = {
                "loss": self._all_reduce_mean(loss),
                "positive_energy": self._all_reduce_mean(positive_energy),
                "negative_energy": self._all_reduce_mean(negative_energy),
                "energy_difference": self._all_reduce_mean(energy_difference),
            }
            for key, value in step_metrics.items():
                values[key].append(value)

            pbar.set_postfix(
                {
                    "loss": f"{step_metrics['loss']:.4g}",
                    "E+": f"{step_metrics['positive_energy']:.4g}",
                    "E-": f"{step_metrics['negative_energy']:.4g}",
                    "dE": f"{step_metrics['energy_difference']:.4g}",
                },
                refresh=False,
            )

        return {key: float(np.mean(items)) for key, items in values.items()}

    @torch.no_grad()
    def _collect_scores(self, loader: DataLoader, steps: int) -> np.ndarray:
        self.nae_core.eval()
        scores = []
        iterator = iter(loader)
        for _ in tqdm(
            range(steps),
            desc="Collecting NAE scores",
            disable=not self.is_main,
            leave=False,
        ):
            batch = next(iterator)
            z = self._backbone_representation(batch)
            scores.append(self.nae_core.mse_per_sample(z))
        local = torch.cat(scores, dim=0)
        return self._all_gather_rows(local).numpy()

    def _evaluate_roc(self, epoch: int) -> Tuple[float, float]:
        train_scores = self._collect_scores(self.train_eval_loader, self.args.eval_steps)
        val_scores = self._collect_scores(self.val_loader, self.args.eval_steps)
        signal_scores = self._collect_scores(self.signal_loader, self.args.eval_steps)

        if not self.is_main:
            return float("nan"), float("nan")

        def auc(background: np.ndarray) -> float:
            labels = np.concatenate(
                [np.zeros(len(background)), np.ones(len(signal_scores))]
            )
            scores = np.concatenate([background, signal_scores])
            return float(roc_auc_score(labels, scores))

        train_auc = auc(train_scores)
        val_auc = auc(val_scores)

        fig, ax = plt.subplots(figsize=(6.5, 6.0))
        for name, background, value in (
            ("Background train", train_scores, train_auc),
            ("Background validation", val_scores, val_auc),
        ):
            labels = np.concatenate(
                [np.zeros(len(background)), np.ones(len(signal_scores))]
            )
            scores = np.concatenate([background, signal_scores])
            fpr, tpr, _ = roc_curve(labels, scores)
            ax.plot(fpr, tpr, label=f"{name} (AUC={value:.4f})")
        ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
        ax.set_xlabel("Background false-positive rate")
        ax.set_ylabel("Signal true-positive rate")
        ax.set_title(f"NAE Reconstruction-Energy ROC — Epoch {epoch}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.roc_dir / f"roc_epoch_{epoch:04d}.png", dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        all_scores = np.concatenate([train_scores, val_scores, signal_scores])
        low, high = np.quantile(all_scores, [0.001, 0.995])
        bins = np.linspace(low, high, 80)
        ax.hist(train_scores, bins=bins, density=True, histtype="step", label="Background train")
        ax.hist(val_scores, bins=bins, density=True, histtype="step", label="Background validation")
        ax.hist(signal_scores, bins=bins, density=True, histtype="step", label="Signal")
        ax.set_xlabel("NAE reconstruction MSE")
        ax.set_ylabel("Density")
        ax.set_title(f"NAE Anomaly-Score Distribution — Epoch {epoch}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.score_dir / f"scores_epoch_{epoch:04d}.png", dpi=160)
        plt.close(fig)

        return train_auc, val_auc

    def _plot_progress(
        self,
        history: Dict[str, List[float]],
        phase_boundary: int,
    ) -> None:
        if not self.is_main or not history["train_loss"]:
            return
        epochs = np.arange(1, len(history["train_loss"]) + 1)
        fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
        panels = (
            ("loss", "Training Objective"),
            ("positive_energy", "Positive/Data Energy"),
            ("negative_energy", "Negative/Model Energy"),
            ("energy_difference", "Energy Difference E+ - E-"),
        )
        for axis, (key, title) in zip(axes, panels):
            axis.plot(epochs, history[f"train_{key}"], label="Train")
            axis.plot(epochs, history[f"val_{key}"], label="Validation")
            axis.axvline(phase_boundary + 0.5, linestyle="--", label="NAE phase starts")
            axis.set_ylabel(title)
            axis.legend()
        axes[-1].set_xlabel("Epoch")
        fig.suptitle("Normalized Autoencoder Training Progress")
        fig.tight_layout()
        fig.savefig(self.output_dir / "training_progress.png", dpi=160)
        plt.close(fig)

        if history["val_auc"]:
            roc_epochs = np.asarray(history["roc_epochs"])
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(roc_epochs, history["train_auc"], marker="o", label="Train background vs signal")
            ax.plot(roc_epochs, history["val_auc"], marker="o", label="Validation background vs signal")
            ax.axhline(0.5, linestyle="--")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("ROC AUC")
            ax.set_title("NAE Anomaly-Detection AUC")
            ax.legend()
            fig.tight_layout()
            fig.savefig(self.output_dir / "auc_progress.png", dpi=160)
            plt.close(fig)

    def train(self) -> None:
        total_epochs = self.args.ae_pretrain_epochs + self.args.nae_epochs
        total_steps = total_epochs * self.args.steps_per_epoch
        optimizer = torch.optim.Adam(
            self.nae.parameters(),
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.args.weight_decay,
        )
        warmup_steps = self.args.warmup_epochs * self.args.steps_per_epoch
        scheduler = make_warmup_cosine_scheduler(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            final_lr_ratio=self.args.final_lr_ratio,
        )

        history: Dict[str, List[float]] = {
            f"{split}_{metric}": []
            for split in ("train", "val")
            for metric in ("loss", "positive_energy", "negative_energy", "energy_difference")
        }
        history.update(
            {
                "roc_epochs": [],
                "train_auc": [],
                "val_auc": [],
            }
        )

        best_pretrain_val = float("inf")
        best_nae_abs_difference = float("inf")
        best_nae_epoch: Optional[int] = None
        best_pretrain_path = self.output_dir / "best_pretrain_model.pth"
        best_nae_path = self.output_dir / "best_model.pth"
        last_path = self.output_dir / "last_model.pth"
        summary_path = self.output_dir / "summary.json"

        summary = {
            "status": "training",
            "backbone_dir": str(self.backbone_dir),
            "backbone_checkpoint": str(
                self.args.backbone_checkpoint
                or self.backbone_dir / "best_model.pth"
            ),
            "backbone_model": self.backbone_summary["model"],
            "dataset_root": str(self.dataset_root),
            "background_labels": self.background_labels,
            "signal_labels": self.signal_labels,
            "particle_features": self.particle_features,
            "representation_dim": self.representation_dim,
            "hidden_dims": list(self.args.hidden_dims),
            "bottleneck_dim": self.args.bottleneck_dim,
            "activation": self.args.activation,
            "ae_pretrain_epochs": self.args.ae_pretrain_epochs,
            "nae_epochs": self.args.nae_epochs,
            "steps_per_epoch": self.args.steps_per_epoch,
            "val_steps": self.args.val_steps,
            "eval_steps": self.args.eval_steps,
            "batch_size": self.args.batch_size,
            "world_size": self.world_size,
            "per_rank_batch_size": self.per_rank_batch_size,
            "learning_rate": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
            "langevin_steps": self.args.langevin_steps,
            "langevin_step_size": self.args.langevin_step_size,
            "langevin_noise_scale": self.args.langevin_noise_scale,
            "negative_init_noise": self.args.negative_init_noise,
            "negative_energy_weight": self.args.negative_energy_weight,
            "best_pretrain_val_energy": None,
            "best_nae_abs_energy_difference": None,
            "best_nae_epoch": None,
        }

        def save_summary() -> None:
            if not self.is_main:
                return
            temporary = summary_path.with_suffix(".json.tmp")
            with temporary.open("w") as file:
                json.dump(summary, file, indent=2)
            os.replace(temporary, summary_path)

        save_summary()
        train_iterator = iter(self.train_loader)

        # A persistent iterator is required for the infinite stream.  Wrap it in
        # a tiny adapter because _run_epoch expects a DataLoader-like iterable.
        class IteratorAdapter:
            def __iter__(self_nonlocal):
                return train_iterator

        train_source = IteratorAdapter()

        for epoch in range(1, total_epochs + 1):
            phase = (
                "ae_pretrain"
                if epoch <= self.args.ae_pretrain_epochs
                else "nae"
            )
            if self.is_main:
                print(f"\nEpoch [{epoch}/{total_epochs}] — {phase}")

            train_metrics = self._run_epoch(
                train_source,
                self.args.steps_per_epoch,
                optimizer,
                phase,
                epoch,
                scheduler=scheduler,
            )

            val_metrics = self._run_epoch(
                self.val_loader,
                self.args.val_steps,
                optimizer=None,
                phase=phase,
                epoch=epoch,
            )

            for metric in ("loss", "positive_energy", "negative_energy", "energy_difference"):
                history[f"train_{metric}"].append(train_metrics[metric])
                history[f"val_{metric}"].append(val_metrics[metric])

            if self.is_main:
                torch.save(self.nae_core.state_dict(), last_path)
                if phase == "ae_pretrain" and val_metrics["positive_energy"] < best_pretrain_val:
                    best_pretrain_val = val_metrics["positive_energy"]
                    torch.save(self.nae_core.state_dict(), best_pretrain_path)
                elif phase == "nae":
                    abs_difference = abs(val_metrics["energy_difference"])
                    if abs_difference < best_nae_abs_difference:
                        best_nae_abs_difference = abs_difference
                        best_nae_epoch = epoch
                        torch.save(self.nae_core.state_dict(), best_nae_path)

            if self.args.roc_eval_every > 0 and (
                epoch % self.args.roc_eval_every == 0 or epoch == total_epochs
            ):
                train_auc, val_auc = self._evaluate_roc(epoch)
                if self.is_main:
                    history["roc_epochs"].append(epoch)
                    history["train_auc"].append(train_auc)
                    history["val_auc"].append(val_auc)

            if self.is_main:
                self._plot_progress(history, self.args.ae_pretrain_epochs)
                summary.update(
                    {
                        "current_epoch": epoch,
                        "phase": phase,
                        "latest_train_metrics": train_metrics,
                        "latest_val_metrics": val_metrics,
                        "best_pretrain_val_energy": (
                            None if not np.isfinite(best_pretrain_val) else best_pretrain_val
                        ),
                        "best_nae_abs_energy_difference": (
                            None
                            if not np.isfinite(best_nae_abs_difference)
                            else best_nae_abs_difference
                        ),
                        "best_nae_epoch": best_nae_epoch,
                        "current_learning_rate": optimizer.param_groups[0]["lr"],
                        "history": history,
                    }
                )
                save_summary()
                print(f"Train metrics: {train_metrics}")
                print(f"Validation metrics: {val_metrics}")

            if self.distributed:
                dist.barrier()

        if self.args.nae_epochs == 0 and self.is_main:
            torch.save(self.nae_core.state_dict(), best_nae_path)

        if self.distributed:
            dist.barrier()

        # Evaluate the paper-selected NAE checkpoint once more.
        if best_nae_path.is_file():
            self.nae_core.load_state_dict(read_state_dict(best_nae_path, self.device))
        final_train_auc, final_val_auc = self._evaluate_roc(total_epochs)

        if self.is_main:
            summary.update(
                {
                    "status": "completed",
                    "final_train_auc": final_train_auc,
                    "final_val_auc": final_val_auc,
                    "best_nae_epoch": best_nae_epoch,
                }
            )
            save_summary()
            print(f"Final train-background AUC: {final_train_auc:.6f}")
            print(f"Final validation-background AUC: {final_val_auc:.6f}")

    def close(self) -> None:
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a normalized autoencoder on online LeJEPA representations."
    )
    parser.add_argument("--backbone-dir", type=str, required=True)
    parser.add_argument("--backbone-checkpoint", type=str, default=None)
    parser.add_argument("--dataset-root", type=str, default=None)
    parser.add_argument("--signal-labels", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--hidden-dims", type=parse_int_list, default=(64, 32, 16, 8))
    parser.add_argument("--bottleneck-dim", type=int, default=3)
    parser.add_argument(
        "--activation",
        choices=["relu", "leaky_relu", "silu", "gelu", "tanh"],
        default="relu",
    )

    parser.add_argument("--ae-pretrain-epochs", type=int, default=100)
    parser.add_argument("--nae-epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--val-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--roc-eval-every", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--final-lr-ratio", type=float, default=1.0)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)

    parser.add_argument("--langevin-steps", type=int, default=20)
    parser.add_argument("--langevin-step-size", type=float, default=1e-2)
    parser.add_argument("--langevin-noise-scale", type=float, default=1e-2)
    parser.add_argument("--negative-init-noise", type=float, default=1e-2)
    parser.add_argument("--negative-energy-weight", type=float, default=1.0)
    parser.add_argument("--langevin-clip", type=float, default=None)

    parser.add_argument("--max-num-particles", type=int, default=128)
    parser.add_argument("--max-train-events", type=int, default=None)
    parser.add_argument("--max-val-events", type=int, default=None)
    parser.add_argument("--max-test-signal-events", type=int, default=None)
    parser.add_argument("--shuffle-active-shards", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
    )
    parser.add_argument(
        "--backbone-precision",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trainer = TrainNAE(args)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
