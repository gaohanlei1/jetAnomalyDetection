"""
Train a minimal ParticleTransformer representation model with LeJEPA-style SSL.

This script is intentionally simpler than the previous masked-reconstruction
training script:

- It reads per-event node features directly from pandas DataFrames.
- It does not construct graph edges.
- It trains only on background jets using multi-view pt-drop augmentation.
- It validates on held-out background jets with the same SSL objective.
- It plots total / invariant / SIGReg losses.
- It plots the full-jet representation space for background validation jets
  and signal jets without any crop/drop augmentation.

Expected node feature order by default:

    [eta, phi, pt, d0/d0Err, dz/dzErr, mass, charge]

Example command:

python -u scripts/run_train_lejepa_part.py \
    --background "data/processed/qcd-vs-wjet-pt-200to400/QCD_scaled_scaled.pkl" \
    --signal "data/processed/qcd-vs-wjet-pt-200to400/WJet_scaled_scaled.pkl" \
    --embed-dim 128 \
    --representation-dim 128 \
    --num-layers 4 \
    --num-heads 8 \
    --batch-size 128 \
    --epochs 100 \
    --learning-rate 5e-4 \
    --weight-decay 5e-2 \
    --precision bf16 \
    --output-dir "plots/run-lejepa-part"
"""

import argparse
import json
import logging
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Add parent directory to import local project modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import constants as c
from helpers import helpers_main
from models.part import (
    LeJEPALossConfig,
    LeJEPAParticleTransformerRepresentation,
    ParticleTransformerConfig,
    PtDropAugmentationConfig,
)
from visualize.plot_latent_space import reduce_to_2d, plot_latent_space

config = helpers_main.load_config()
bg_file = os.path.join(config["data"]["processed_data_dir"], config["data"]["background_file"])
sg_file = os.path.join(config["data"]["processed_data_dir"], config["data"]["signal_file"])
DEVICE = torch.device(helpers_main.get_device())


def parse_node_features(feature_string: str) -> List[str]:
    """
    Parse a comma-separated node feature list.

    Example:
        "eta,phi,pt,d0/d0Err,dz/dzErr,mass,charge"
    """

    features = [item.strip() for item in feature_string.split(",") if item.strip()]
    if len(features) == 0:
        raise ValueError("At least one node feature must be provided.")
    return features


def row_to_node_tensor(
    row: pd.Series,
    node_feature_names: Sequence[str],
    min_nodes: int,
) -> torch.Tensor:
    """
    Convert one DataFrame row into a node tensor.

    Input row columns are expected to contain array-like per-node values.
    The output tensor has shape:

        (N, F)

    No edge index is constructed.
    """

    arrays = []
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

    # Pairwise physics features assume pt is positive. If pt is present,
    # remove non-positive pt nodes before padding/collation.
    if "pt" in node_feature_names:
        pt_index = node_feature_names.index("pt")
        valid_mask = valid_mask & (stacked[:, pt_index] > 0)

    stacked = stacked[valid_mask]

    if stacked.shape[0] < min_nodes:
        raise ValueError(f"Too few valid nodes after cleaning: {stacked.shape[0]}.")

    return torch.tensor(stacked, dtype=torch.float32)


def dataframe_to_node_tensors(
    df: pd.DataFrame,
    node_feature_names: Sequence[str],
    label: int,
    min_nodes: int,
    max_events: Optional[int] = None,
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Convert a DataFrame into a list of variable-length node tensors.

    Returns:
        node_tensors:
            List of tensors, each with shape (N_i, F).

        labels:
            List of integer labels. These are used only for bookkeeping and
            latent-space plotting; SSL training itself only uses background.
    """

    if max_events is not None:
        df = df.head(max_events)

    node_tensors: List[torch.Tensor] = []
    labels: List[int] = []

    for i in tqdm(range(len(df)), desc="Loading node tensors"):
        try:
            x = row_to_node_tensor(
                row=df.iloc[i],
                node_feature_names=node_feature_names,
                min_nodes=min_nodes,
            )
            node_tensors.append(x)
            labels.append(label)
        except Exception as exc:
            logging.info(f"Skipping event {i} due to error: {exc}")

    return node_tensors, labels


class JetNodeDataset(Dataset):
    """
    Dataset wrapping variable-length jet node tensors.

    Each item is:
        x: (N_i, F)
        y: scalar label
    """

    def __init__(self, node_tensors: Sequence[torch.Tensor], labels: Sequence[int]):
        if len(node_tensors) != len(labels):
            raise ValueError("node_tensors and labels must have the same length.")
        self.node_tensors = list(node_tensors)
        self.labels = list(labels)

    def __len__(self) -> int:
        return len(self.node_tensors)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.node_tensors[index], self.labels[index]


def collate_node_tensors(batch: Sequence[Tuple[torch.Tensor, int]]) -> Dict[str, torch.Tensor]:
    """
    Pad variable-length node tensors into a dense batch.

    Returns:
        x:
            (B, N_max, F)

        padding_mask:
            (B, N_max), bool. True means padded node.

        y:
            (B,)
    """

    xs, labels = zip(*batch)
    batch_size = len(xs)
    max_nodes = max(x.size(0) for x in xs)
    feature_dim = xs[0].size(1)

    padded = torch.zeros(batch_size, max_nodes, feature_dim, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, max_nodes, dtype=torch.bool)

    for i, x in enumerate(xs):
        num_nodes = x.size(0)
        padded[i, :num_nodes] = x
        padding_mask[i, :num_nodes] = False

    y = torch.tensor(labels, dtype=torch.long)

    return {
        "x": padded,
        "padding_mask": padding_mask,
        "y": y,
    }


def compute_feature_stats(node_tensors: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute node-feature mean/std over a list of variable-length tensors.

    This is provided for diagnostics. Feature normalization is disabled by
    default because the pairwise physics bias expects physical eta/phi/pt/mass.
    """

    all_nodes = torch.cat(list(node_tensors), dim=0)
    mean = all_nodes.mean(dim=0)
    std = all_nodes.std(dim=0)
    std[std == 0] = 1.0
    return mean, std


def apply_feature_normalization(
    node_tensors: Sequence[torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Normalize node tensors using provided feature statistics.

    Use with caution: normalizing eta/phi/pt/mass changes the physical meaning
    of the pairwise attention bias.
    """

    return [(x - mean) / std for x in node_tensors]


def make_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    final_lr_ratio: float,
) -> LambdaLR:
    """
    Linear warmup followed by cosine decay.

    Final learning rate:
        initial_lr * final_lr_ratio
    """

    total_steps = max(1, int(total_steps))
    warmup_steps = max(0, int(warmup_steps))
    final_lr_ratio = float(final_lr_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if total_steps <= warmup_steps:
            return 1.0

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_lr_ratio + (1.0 - final_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def precision_to_dtype(precision: str) -> torch.dtype:
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "fp32":
        return torch.float32
    raise ValueError(f"Unknown precision: {precision}")


def autocast_enabled_for_precision(precision: str) -> bool:
    return precision in {"bf16", "fp16"}


class TrainLeJEPAParticleTransformer:
    """
    Driver class for LeJEPA-style SSL pretraining.
    """

    TRAIN_SPLIT = 0.8

    def __init__(self):
        self.args = parser.parse_args()

        self.bg_file = self.args.background
        self.sg_file = self.args.signal
        self.bg_name = helpers_main.trim_name(self.bg_file)
        self.sg_name = helpers_main.trim_name(self.sg_file)

        self.node_feature_names = parse_node_features(self.args.node_features)
        self.pt_index = self.node_feature_names.index("pt")

        self.output_dir = self.args.output_dir
        self.feature_plot_dir = os.path.join(self.output_dir, "features")
        self.latent_plot_dir = os.path.join(self.output_dir, "latent_space")
        self.augmentation_plot_dir = os.path.join(self.output_dir, "augmentation_views")

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.feature_plot_dir, exist_ok=True)
        os.makedirs(self.latent_plot_dir, exist_ok=True)
        os.makedirs(self.augmentation_plot_dir, exist_ok=True)

        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.args.seed)

        self.session_name = os.path.join(
            self.output_dir,
            f"train_lejepa_part_{self.bg_name}_{self.sg_name}_{helpers_main.curr_time()}.log",
        )
        helpers_main.log_config(self.session_name)

    def load(self) -> None:
        """
        Load background and signal DataFrames.

        No pt slicing is performed here. The input files are assumed to have
        already been sliced/preprocessed upstream.
        """

        logging.info(f"Loading background from {self.bg_file}")
        logging.info(f"Loading signal from {self.sg_file}")

        self.bg_data = pd.read_pickle(self.bg_file)
        self.sg_data = pd.read_pickle(self.sg_file)

        if self.args.max_background_events is not None:
            self.bg_data = self.bg_data.head(self.args.max_background_events)
        if self.args.max_signal_events is not None:
            self.sg_data = self.sg_data.head(self.args.max_signal_events)

        logging.info(f"Background rows: {len(self.bg_data)}")
        logging.info(f"Signal rows: {len(self.sg_data)}")
        logging.info(f"Background columns: {self.bg_data.columns.tolist()}")
        logging.info(f"Signal columns: {self.sg_data.columns.tolist()}")

        print(f"Background rows: {len(self.bg_data)}")
        print(f"Signal rows: {len(self.sg_data)}")
        print(f"Node features: {self.node_feature_names}")

    def build_node_datasets(self) -> None:
        """
        Convert DataFrames into variable-length node tensor datasets.
        """

        print("Loading background node tensors...")
        bg_nodes, bg_labels = dataframe_to_node_tensors(
            df=self.bg_data,
            node_feature_names=self.node_feature_names,
            label=0,
            min_nodes=self.args.min_nodes,
            max_events=None,
        )

        print("Loading signal node tensors...")
        sg_nodes, sg_labels = dataframe_to_node_tensors(
            df=self.sg_data,
            node_feature_names=self.node_feature_names,
            label=1,
            min_nodes=self.args.min_nodes,
            max_events=None,
        )

        if len(bg_nodes) == 0:
            raise ValueError("No valid background events were loaded.")
        if len(sg_nodes) == 0:
            raise ValueError("No valid signal events were loaded.")

        train_size = int(self.TRAIN_SPLIT * len(bg_nodes))
        train_size = max(1, min(train_size, len(bg_nodes) - 1))
        # shuffle bg before splitting
        bg_indices = list(range(len(bg_nodes)))
        random.shuffle(bg_indices)
        bg_nodes = [bg_nodes[i] for i in bg_indices]
        bg_labels = [bg_labels[i] for i in bg_indices]
        
        self.bg_train_nodes = bg_nodes[:train_size]
        self.bg_train_labels = bg_labels[:train_size]
        self.bg_val_nodes = bg_nodes[train_size:]
        self.bg_val_labels = bg_labels[train_size:]
        self.sg_nodes = sg_nodes
        self.sg_labels = sg_labels

        if self.args.normalize_features:
            logging.warning(
                "Feature normalization is enabled. This changes eta/phi/pt/mass "
                "before pairwise physics bias computation. Use only if intended."
            )
            self.feature_mean, self.feature_std = compute_feature_stats(self.bg_train_nodes)
            self.bg_train_nodes = apply_feature_normalization(
                self.bg_train_nodes,
                self.feature_mean,
                self.feature_std,
            )
            self.bg_val_nodes = apply_feature_normalization(
                self.bg_val_nodes,
                self.feature_mean,
                self.feature_std,
            )
            self.sg_nodes = apply_feature_normalization(
                self.sg_nodes,
                self.feature_mean,
                self.feature_std,
            )
        else:
            self.feature_mean, self.feature_std = compute_feature_stats(self.bg_train_nodes)

        self.bg_train_dataset = JetNodeDataset(self.bg_train_nodes, self.bg_train_labels)
        self.bg_val_dataset = JetNodeDataset(self.bg_val_nodes, self.bg_val_labels)
        self.sg_dataset = JetNodeDataset(self.sg_nodes, self.sg_labels)

        logging.info(f"Background train events: {len(self.bg_train_dataset)}")
        logging.info(f"Background val events: {len(self.bg_val_dataset)}")
        logging.info(f"Signal events: {len(self.sg_dataset)}")
        logging.info(f"Feature mean: {self.feature_mean}")
        logging.info(f"Feature std: {self.feature_std}")

        print(f"Background train events: {len(self.bg_train_dataset)}")
        print(f"Background val events: {len(self.bg_val_dataset)}")
        print(f"Signal events: {len(self.sg_dataset)}")
        print(f"Example node tensor shape: {self.bg_train_nodes[0].shape}")

    def plot_features(self) -> None:
        """
        Plot background training feature distributions for sanity checks.
        """

        os.makedirs(self.feature_plot_dir, exist_ok=True)
        all_features = torch.cat(self.bg_train_nodes, dim=0).numpy()

        for i, name in enumerate(self.node_feature_names):
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(
                all_features[:, i],
                bins=50,
                density=True,
                alpha=0.8,
                edgecolor="black",
            )
            ax.set_title(f"Feature {i}: {name}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.grid(False)
            fig.tight_layout()

            safe_name = name.replace("/", "_")
            fig.savefig(
                os.path.join(self.feature_plot_dir, f"feature_{i}_{safe_name}.png")
            )
            plt.close(fig)

    def plot_augmentation_samples(self) -> None:
        """
        Plot original background jets and their augmented views before training.

        Each saved figure corresponds to one randomly selected background event.
        The subplots show:
            - original full jet
            - all global pt-drop views
            - all local pt-drop views

        Plot convention:
            x-axis: phi
            y-axis: eta
            color: pt
        """

        if not hasattr(self, "model"):
            raise RuntimeError("Model must be built before plotting augmentation samples.")

        if len(self.bg_train_nodes) == 0:
            logging.warning("No background training nodes available for augmentation plots.")
            return

        os.makedirs(self.augmentation_plot_dir, exist_ok=True)

        num_samples = min(self.args.num_augmentation_plot_samples, len(self.bg_train_nodes))
        if num_samples <= 0:
            return

        sample_indices = random.sample(range(len(self.bg_train_nodes)), k=num_samples)

        for plot_idx, sample_idx in enumerate(sample_indices):
            x_single = self.bg_train_nodes[sample_idx]
            padding_mask_single = torch.zeros(
                1,
                x_single.size(0),
                dtype=torch.bool,
            )
            x_batch = x_single.unsqueeze(0)

            views, view_padding_masks, view_types = self.model.augmentation(
                x=x_batch,
                padding_mask=padding_mask_single,
            )

            panels = [(x_batch[0], padding_mask_single[0], "original")]
            for view_i, (view_x, view_mask, view_type) in enumerate(
                zip(views, view_padding_masks, view_types),
                start=1,
            ):
                panels.append((view_x[0], view_mask[0], f"{view_i}: {view_type}"))

            output_path = os.path.join(
                self.augmentation_plot_dir,
                f"augmentation_sample_{plot_idx + 1:02d}_event_{sample_idx:06d}.png",
            )
            self._plot_single_augmentation_panel(
                panels=panels,
                output_path=output_path,
                title=f"Background event {sample_idx}: original and pt-drop views",
            )

    def _plot_single_augmentation_panel(
        self,
        panels: Sequence[Tuple[torch.Tensor, torch.Tensor, str]],
        output_path: str,
        title: str,
    ) -> None:
        """
        Plot one event's original jet and augmented views as eta-phi subplots.
        """

        eta_index = self.node_feature_names.index("eta")
        phi_index = self.node_feature_names.index("phi")
        pt_index = self.node_feature_names.index("pt")

        num_panels = len(panels)
        num_cols = min(3, num_panels)
        num_rows = int(np.ceil(num_panels / num_cols))

        valid_pts = []
        valid_etas = []
        valid_phis = []

        for x_panel, mask_panel, _ in panels:
            valid = ~mask_panel.bool()
            if valid.any():
                valid_pts.append(x_panel[valid, pt_index].detach().cpu().numpy())
                valid_etas.append(x_panel[valid, eta_index].detach().cpu().numpy())
                valid_phis.append(x_panel[valid, phi_index].detach().cpu().numpy())

        if len(valid_pts) == 0:
            logging.warning(f"Skipping augmentation plot with no valid nodes: {output_path}")
            return

        all_pt = np.concatenate(valid_pts)
        all_eta = np.concatenate(valid_etas)
        all_phi = np.concatenate(valid_phis)

        pt_min = float(np.nanmin(all_pt))
        pt_max = float(np.nanmax(all_pt))
        if not np.isfinite(pt_min) or not np.isfinite(pt_max) or pt_min == pt_max:
            pt_min, pt_max = 0.0, 1.0

        eta_min = float(np.nanmin(all_eta))
        eta_max = float(np.nanmax(all_eta))
        phi_min = float(np.nanmin(all_phi))
        phi_max = float(np.nanmax(all_phi))

        eta_pad = max(0.05, 0.05 * (eta_max - eta_min + 1e-8))
        phi_pad = max(0.05, 0.05 * (phi_max - phi_min + 1e-8))

        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(5 * num_cols, 4.5 * num_rows),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        last_scatter = None

        for ax, (x_panel, mask_panel, panel_title) in zip(axes_flat, panels):
            valid = ~mask_panel.bool()
            x_np = x_panel.detach().cpu().numpy()
            valid_np = valid.detach().cpu().numpy()

            eta = x_np[valid_np, eta_index]
            phi = x_np[valid_np, phi_index]
            pt = x_np[valid_np, pt_index]

            last_scatter = ax.scatter(
                phi,
                eta,
                c=pt,
                cmap="viridis",
                s=16,
                alpha=0.8,
                vmin=pt_min,
                vmax=pt_max,
            )
            ax.set_title(f"{panel_title} ({len(pt)} nodes)")
            ax.set_xlabel("Phi")
            ax.set_ylabel("Eta")
            ax.set_xlim(phi_min - phi_pad, phi_max + phi_pad)
            ax.set_ylim(eta_min - eta_pad, eta_max + eta_pad)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(False)

        for ax in axes_flat[num_panels:]:
            ax.axis("off")

        if last_scatter is not None:
            fig.colorbar(last_scatter, ax=axes_flat[:num_panels].tolist(), label="pt")

        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)

    def make_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_loader = DataLoader(
            self.bg_train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=collate_node_tensors,
        )

        bg_val_loader = DataLoader(
            self.bg_val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=collate_node_tensors,
        )

        signal_loader = DataLoader(
            self.sg_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=collate_node_tensors,
        )

        return train_loader, bg_val_loader, signal_loader

    def build_model(self) -> None:
        model_config = ParticleTransformerConfig(
            input_dim=len(self.node_feature_names),
            embed_dim=self.args.embed_dim,
            num_heads=self.args.num_heads,
            num_layers=self.args.num_layers,
            ffn_mult=self.args.ffn_mult,
            dropout=self.args.dropout,
            representation_dim=self.args.representation_dim,
            use_pairwise_bias=not self.args.no_pairwise_bias,
            pairwise_hidden_dim=self.args.pairwise_hidden_dim,
            pairwise_num_features=self.args.pairwise_num_features,
            compute_dtype=precision_to_dtype(self.args.precision),
            use_internal_autocast=False,
            eps=self.args.eps,
        )

        augmentation_config = PtDropAugmentationConfig(
            num_global_views=self.args.num_global_views,
            num_local_views=self.args.num_local_views,
            global_drop_pt_frac_range=(
                self.args.global_drop_pt_frac_min,
                self.args.global_drop_pt_frac_max,
            ),
            local_drop_pt_frac_range=(
                self.args.local_drop_pt_frac_min,
                self.args.local_drop_pt_frac_max,
            ),
            min_nodes=self.args.min_nodes,
            pt_index=self.pt_index,
            eps=self.args.eps,
            pt_drop_power=self.args.pt_drop_power,
            zero_dropped_features=not self.args.keep_dropped_features,
        )

        loss_config = LeJEPALossConfig(
            invariant_weight=self.args.invariant_weight,
            sigreg_weight=self.args.sigreg_weight,
            epps_pulley_num_points=self.args.epps_pulley_num_points,
            num_slices=self.args.num_slices,
            normalize_representations_for_invariant=self.args.normalize_invariant_representations,
            normalize_representations_for_sigreg=self.args.normalize_sigreg_representations,
        )

        self.model = LeJEPAParticleTransformerRepresentation(
            model_config=model_config,
            augmentation_config=augmentation_config,
            loss_config=loss_config,
        ).to(DEVICE)

        logging.info(f"Model summary:\n{self.model}")
        print(f"Model summary:\n{self.model}")

        self.num_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logging.info(f"Number of trainable parameters: {self.num_params}")
        print(f"Number of trainable parameters: {self.num_params}")

    def plot_progress(
        self,
        train_history: Dict[str, List[float]],
        val_history: Dict[str, List[float]],
        epoch_end_steps: List[int],
        best_val_loss: float,
    ) -> None:
        """
        Plot train/validation curves for total, invariant, and SIGReg losses.
        """

        if len(train_history["total_loss"]) == 0:
            return

        loss_keys = ["total_loss", "invariant_loss", "sigreg_loss"]
        titles = ["Total Loss", "Invariant Loss", "SIGReg Loss"]

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        step_axis = np.arange(1, len(train_history["total_loss"]) + 1)
        epoch_end_steps_np = np.asarray(epoch_end_steps)

        for ax, key, title in zip(axes, loss_keys, titles):
            train_values = np.asarray(train_history[key], dtype=np.float64)
            val_values = np.asarray(val_history[key], dtype=np.float64)

            ax.plot(step_axis, train_values, label="Train", alpha=0.75)

            if len(val_values) > 0:
                repeat_count = int(np.ceil(len(train_values) / len(val_values)))
                repeated_val = np.repeat(val_values, repeat_count)[: len(train_values)]
                ax.plot(step_axis, repeated_val, label="Validation", alpha=0.75)

            if key == "total_loss" and np.isfinite(best_val_loss):
                ax.axhline(
                    y=best_val_loss,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.25,
                    label=f"Best Val: {best_val_loss:.4g}",
                )

            if len(epoch_end_steps_np) > 0:
                max_labels = 12
                stride = max(1, int(np.ceil(len(epoch_end_steps_np) / max_labels)))
                for step_idx in epoch_end_steps_np[::stride]:
                    ax.axvline(
                        step_idx,
                        color="gray",
                        ls="--",
                        lw=0.6,
                        alpha=0.25,
                    )

            ax.set_ylabel(title)
            ax.set_yscale("log")
            ax.legend()
            ax.grid(False)

        axes[-1].set_xlabel("Step Number")

        if len(epoch_end_steps_np) > 0:
            epoch_ids = np.arange(1, len(epoch_end_steps_np) + 1)
            max_labels = 12
            stride = max(1, int(np.ceil(len(epoch_end_steps_np) / max_labels)))
            top_ax = axes[0].secondary_xaxis("top")
            top_ax.set_xticks(epoch_end_steps_np[::stride])
            top_ax.set_xticklabels(epoch_ids[::stride])
            top_ax.set_xlabel("Epoch")

        fig.suptitle("LeJEPA SSL Loss Curves")
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, "loss.png"))
        plt.close(fig)

    @torch.no_grad()
    def collect_representations(self, loader: DataLoader) -> np.ndarray:
        """
        Compute full-jet representations without augmentation/crop/drop.
        """

        self.model.eval()
        latents: List[np.ndarray] = []
        dtype = precision_to_dtype(self.args.precision)
        use_autocast = autocast_enabled_for_precision(self.args.precision)

        for batch in loader:
            x = batch["x"].to(DEVICE, non_blocking=True)
            padding_mask = batch["padding_mask"].to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=dtype,
                enabled=use_autocast,
            ):
                z = self.model(
                    x,
                    padding_mask=padding_mask,
                    normalize_output=self.args.normalize_output_representations,
                )

            latents.append(z.detach().float().cpu().numpy())

        return np.concatenate(latents, axis=0)

    def plot_latent_space_for_epoch(
        self,
        bg_val_loader: DataLoader,
        signal_loader: DataLoader,
        epoch: int,
    ) -> None:
        """
        Plot background validation and signal full-jet representations.
        """

        bg_latents = self.collect_representations(bg_val_loader)
        sg_latents = self.collect_representations(signal_loader)

        if self.args.max_latent_plot_points is not None:
            max_points = self.args.max_latent_plot_points
            if len(bg_latents) > max_points:
                bg_indices = np.random.choice(len(bg_latents), max_points, replace=False)
                bg_latents = bg_latents[bg_indices]
            if len(sg_latents) > max_points:
                sg_indices = np.random.choice(len(sg_latents), max_points, replace=False)
                sg_latents = sg_latents[sg_indices]

        bg_2d, sg_2d, x_label, y_label = reduce_to_2d(bg_latents, sg_latents)

        output_path = os.path.join(
            self.latent_plot_dir,
            f"latent_epoch_{epoch:04d}.png",
        )

        plot_latent_space(
            bg_2d,
            sg_2d,
            background_label="QCD (Val)",
            signal_label="WJet",
            output_path=output_path,
            x_label=x_label,
            y_label=y_label,
        )

    def train(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        train_loader, bg_val_loader, signal_loader = self.make_dataloaders()
        self.build_model()
        self.plot_augmentation_samples()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        total_steps = self.args.epochs * max(1, len(train_loader))
        warmup_steps = self.args.warmup_steps
        if warmup_steps is None:
            warmup_steps = self.args.warmup_epochs * max(1, len(train_loader))

        scheduler = make_warmup_cosine_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            final_lr_ratio=self.args.final_lr_ratio,
        )

        dtype = precision_to_dtype(self.args.precision)
        use_autocast = autocast_enabled_for_precision(self.args.precision)

        train_history = {
            "total_loss": [],
            "invariant_loss": [],
            "sigreg_loss": [],
        }
        val_history = {
            "total_loss": [],
            "invariant_loss": [],
            "sigreg_loss": [],
        }
        epoch_end_steps: List[int] = []

        best_val_loss = float("inf")
        best_model_path = os.path.join(self.output_dir, "best_model.pth")
        timer = helpers_main.LeTimer()

        for epoch in range(1, self.args.epochs + 1):
            logging.info(f"\nEpoch [{epoch}/{self.args.epochs}]")
            logging.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.8g}")

            self.model.train()
            epoch_train = {
                "total_loss": [],
                "invariant_loss": [],
                "sigreg_loss": [],
            }

            pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{self.args.epochs}")
            for batch in pbar:
                x = batch["x"].to(DEVICE, non_blocking=True)
                padding_mask = batch["padding_mask"].to(DEVICE, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=dtype,
                    enabled=use_autocast,
                ):
                    output = self.model.forward_pretrain(
                        x,
                        padding_mask=padding_mask,
                        normalize_output=self.args.normalize_output_representations,
                    )
                    loss = output["total_loss"]

                loss.backward()

                if self.args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.args.grad_clip_norm,
                    )

                optimizer.step()
                scheduler.step()

                step_losses = {
                    "total_loss": float(output["total_loss"].detach().cpu()),
                    "invariant_loss": float(output["invariant_loss"].detach().cpu()),
                    "sigreg_loss": float(output["sigreg_loss"].detach().cpu()),
                }

                for key in train_history:
                    train_history[key].append(step_losses[key])
                    epoch_train[key].append(step_losses[key])

                pbar.set_postfix(
                    {
                        "total": f"{step_losses['total_loss']:.4g}",
                        "inv": f"{step_losses['invariant_loss']:.4g}",
                        "sig": f"{step_losses['sigreg_loss']:.4g}",
                    }
                )

            mean_train = {
                key: float(np.nanmean(values))
                for key, values in epoch_train.items()
            }

            self.model.eval()
            epoch_val = {
                "total_loss": [],
                "invariant_loss": [],
                "sigreg_loss": [],
            }

            with torch.no_grad():
                pbar = tqdm(bg_val_loader, desc=f"Val Epoch {epoch}/{self.args.epochs}")
                for batch in pbar:
                    x = batch["x"].to(DEVICE, non_blocking=True)
                    padding_mask = batch["padding_mask"].to(DEVICE, non_blocking=True)

                    with torch.autocast(
                        device_type=DEVICE.type,
                        dtype=dtype,
                        enabled=use_autocast,
                    ):
                        output = self.model.forward_pretrain(
                            x,
                            padding_mask=padding_mask,
                            normalize_output=self.args.normalize_output_representations,
                        )

                    step_losses = {
                        "total_loss": float(output["total_loss"].detach().cpu()),
                        "invariant_loss": float(output["invariant_loss"].detach().cpu()),
                        "sigreg_loss": float(output["sigreg_loss"].detach().cpu()),
                    }

                    for key in epoch_val:
                        epoch_val[key].append(step_losses[key])

                    pbar.set_postfix(
                        {
                            "total": f"{step_losses['total_loss']:.4g}",
                            "inv": f"{step_losses['invariant_loss']:.4g}",
                            "sig": f"{step_losses['sigreg_loss']:.4g}",
                        }
                    )

            mean_val = {
                key: float(np.nanmean(values))
                for key, values in epoch_val.items()
            }

            for key in val_history:
                val_history[key].append(mean_val[key])

            epoch_end_steps.append(len(train_history["total_loss"]))

            if mean_val["total_loss"] < best_val_loss:
                best_val_loss = mean_val["total_loss"]
                torch.save(self.model, best_model_path)
                logging.info(f"Saved new best model to {best_model_path}")
                print(f"Saved new best model to {best_model_path}")

            self.plot_progress(
                train_history=train_history,
                val_history=val_history,
                epoch_end_steps=epoch_end_steps,
                best_val_loss=best_val_loss,
            )

            if (
                self.args.latent_plot_every > 0
                and (epoch % self.args.latent_plot_every == 0 or epoch == self.args.epochs)
            ):
                self.plot_latent_space_for_epoch(
                    bg_val_loader=bg_val_loader,
                    signal_loader=signal_loader,
                    epoch=epoch,
                )

            logging.info(f"Train losses: {mean_train}")
            logging.info(f"Validation losses: {mean_val}")
            logging.info(timer.time_taken())
            print(f"Epoch {epoch} train losses: {mean_train}")
            print(f"Epoch {epoch} val losses: {mean_val}")

        # Final full-jet representation arrays for downstream inspection.
        bg_val_latents = self.collect_representations(bg_val_loader)
        sg_latents = self.collect_representations(signal_loader)

        np.save(os.path.join(self.output_dir, "background_val_latents.npy"), bg_val_latents)
        np.save(os.path.join(self.output_dir, "signal_latents.npy"), sg_latents)

        summary = {
            "background": self.bg_file,
            "signal": self.sg_file,
            "node_features": self.node_feature_names,
            "batch_size": self.args.batch_size,
            "embed_dim": self.args.embed_dim,
            "representation_dim": self.args.representation_dim,
            "num_layers": self.args.num_layers,
            "num_heads": self.args.num_heads,
            "ffn_mult": self.args.ffn_mult,
            "dropout": self.args.dropout,
            "learning_rate": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
            "precision": self.args.precision,
            "warmup_steps": warmup_steps,
            "final_lr_ratio": self.args.final_lr_ratio,
            "num_global_views": self.args.num_global_views,
            "num_local_views": self.args.num_local_views,
            "global_drop_pt_frac_range": [
                self.args.global_drop_pt_frac_min,
                self.args.global_drop_pt_frac_max,
            ],
            "local_drop_pt_frac_range": [
                self.args.local_drop_pt_frac_min,
                self.args.local_drop_pt_frac_max,
            ],
            "min_nodes": self.args.min_nodes,
            "pt_drop_power": self.args.pt_drop_power,
            "invariant_weight": self.args.invariant_weight,
            "sigreg_weight": self.args.sigreg_weight,
            "epps_pulley_num_points": self.args.epps_pulley_num_points,
            "num_slices": self.args.num_slices,
            "normalize_features": self.args.normalize_features,
            "seed": self.args.seed,
            "device": str(DEVICE),
            "background_train_events": len(self.bg_train_dataset),
            "background_val_events": len(self.bg_val_dataset),
            "signal_events": len(self.sg_dataset),
            "best_val_loss": float(best_val_loss),
            "final_val_loss": float(val_history["total_loss"][-1]),
            "num_trainable_parameters": int(self.num_params),
        }

        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        history_path = os.path.join(self.output_dir, "loss_history.json")
        with open(history_path, "w") as f:
            json.dump(
                {
                    "train": train_history,
                    "val": val_history,
                    "epoch_end_steps": epoch_end_steps,
                },
                f,
                indent=2,
            )

        logging.info(f"Saved run summary to {summary_path}")
        logging.info(f"Saved loss history to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train LeJEPA ParticleTransformer Representation",
        description="Train a minimal ParticleTransformer representation model with LeJEPA-style SSL.",
    )

    # Data.
    parser.add_argument(
        "--background",
        "-b",
        type=str,
        default=bg_file,
        help="Path to processed .pkl background dataset. Defaults to background_file in config.yaml.",
    )
    parser.add_argument(
        "--signal",
        "-s",
        type=str,
        default=sg_file,
        help="Path to processed .pkl signal dataset. Defaults to signal_file in config.yaml.",
    )
    parser.add_argument(
        "--node-features",
        type=str,
        default="eta,phi,pt,d0/d0Err,dz/dzErr,mass,charge",
        help="Comma-separated node feature list. Default: eta,phi,pt,d0/d0Err,dz/dzErr,mass,charge.",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=4,
        help="Minimum number of valid nodes per event and per augmented view. Default: 4.",
    )
    parser.add_argument(
        "--max-background-events",
        type=int,
        default=None,
        help="Optional background row limit for smoke tests.",
    )
    parser.add_argument(
        "--max-signal-events",
        type=int,
        default=None,
        help="Optional signal row limit for smoke tests.",
    )
    parser.add_argument(
        "--normalize-features",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Normalize node features using background-training statistics. "
            "Default: False because pairwise physics bias expects physical eta/phi/pt/mass."
        ),
    )

    # Model.
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Transformer embedding dimension. Default: 128.",
    )
    parser.add_argument(
        "--representation-dim",
        type=int,
        default=128,
        help="Output representation dimension. Default: 128.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of Transformer encoder layers. Default: 4.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads. Default: 8.",
    )
    parser.add_argument(
        "--ffn-mult",
        type=int,
        default=4,
        help="SwiGLU FFN expansion multiplier. Default: 4.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout in embedding, attention, FFN, and projection head. Default: 0.1.",
    )
    parser.add_argument(
        "--no-pairwise-bias",
        action="store_true",
        help="Disable learned pairwise physics attention bias.",
    )
    parser.add_argument(
        "--pairwise-hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension of pairwise attention-bias MLP. Default: 64.",
    )
    parser.add_argument(
        "--pairwise-num-features",
        type=int,
        default=4,
        help="Number of pairwise physics features. Default: 4.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon. Default: 1e-8.",
    )

    # Augmentation.
    parser.add_argument(
        "--num-global-views",
        type=int,
        default=2,
        help="Number of global pt-drop views. Default: 2.",
    )
    parser.add_argument(
        "--num-local-views",
        type=int,
        default=6,
        help="Number of local pt-drop views. Default: 6.",
    )
    parser.add_argument(
        "--global-drop-pt-frac-min",
        type=float,
        default=0.0,
        help="Minimum cumulative pt fraction to drop for global views. Default: 0.0.",
    )
    parser.add_argument(
        "--global-drop-pt-frac-max",
        type=float,
        default=0.50,
        help="Maximum cumulative pt fraction to drop for global views. Default: 0.50.",
    )
    parser.add_argument(
        "--local-drop-pt-frac-min",
        type=float,
        default=0.50,
        help="Minimum cumulative pt fraction to drop for local views. Default: 0.50.",
    )
    parser.add_argument(
        "--local-drop-pt-frac-max",
        type=float,
        default=0.95,
        help="Maximum cumulative pt fraction to drop for local views. Default: 0.95.",
    )
    parser.add_argument(
        "--pt-drop-power",
        type=float,
        default=1.0,
        help="Low-pt-biased drop power. Larger values protect high-pt nodes more strongly. Default: 1.0.",
    )
    parser.add_argument(
        "--keep-dropped-features",
        action="store_true",
        help="Keep dropped node features instead of zeroing them. Dropped nodes are still masked.",
    )

    # Loss.
    parser.add_argument(
        "--invariant-weight",
        type=float,
        default=1.0,
        help="Weight for invariant loss. Default: 1.0.",
    )
    parser.add_argument(
        "--sigreg-weight",
        type=float,
        default=0.02,
        help="Weight for SIGReg loss, matching LeJEPA lambda by default. Default: 0.02.",
    )
    parser.add_argument(
        "--epps-pulley-num-points",
        type=int,
        default=17,
        help="Number of Epps-Pulley test points for SIGReg. Default: 17.",
    )
    parser.add_argument(
        "--num-slices",
        type=int,
        default=1024,
        help="Number of random slices for multivariate SIGReg. Default: 1024.",
    )
    parser.add_argument(
        "--normalize-invariant-representations",
        action="store_true",
        help="L2-normalize representations before invariant loss.",
    )
    parser.add_argument(
        "--normalize-sigreg-representations",
        action="store_true",
        help="L2-normalize representations before SIGReg.",
    )
    parser.add_argument(
        "--normalize-output-representations",
        action="store_true",
        help="L2-normalize representations returned by the encoder.",
    )

    # Optimization.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config["model"]["batch_size"],
        help="Training batch size. Defaults to config model.batch_size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config["training"].get("epochs", 100),
        help="Number of training epochs. Defaults to config training.epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="AdamW learning rate. LeJEPA recommended starting point: 5e-4.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-2,
        help="AdamW weight decay. LeJEPA ViT recommendation: 5e-2.",
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Mixed precision mode. Default: bf16.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        help="Linear warmup epochs. Used if --warmup-steps is not set. Default: 10.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Optional exact number of warmup steps. Overrides --warmup-epochs.",
    )
    parser.add_argument(
        "--final-lr-ratio",
        type=float,
        default=1e-3,
        help="Final LR ratio for cosine decay. Default: 1e-3, so final_lr = initial_lr / 1000.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Optional gradient clipping max norm.",
    )

    # Runtime / output.
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots/run-lejepa-part",
        help="Directory for plots, checkpoints, losses, and summary.json.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers. Default: 0.",
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="Use pinned host memory in DataLoader. Default: True on CUDA, else False.",
    )
    parser.add_argument(
        "--latent-plot-every",
        type=int,
        default=1,
        help="Plot full-jet latent space every N epochs. Use 0 to disable. Default: 1.",
    )
    parser.add_argument(
        "--num-augmentation-plot-samples",
        type=int,
        default=3,
        help="Number of random background events to visualize with their augmented views before training. Default: 3.",
    )
    parser.add_argument(
        "--max-latent-plot-points",
        type=int,
        default=5000,
        help="Maximum points per class in latent-space plots. Use no value by editing to None. Default: 5000.",
    )

    trainer = TrainLeJEPAParticleTransformer()
    trainer.load()
    trainer.build_node_datasets()
    trainer.plot_features()
    trainer.train()