"""
Train a ParticleTransformer representation model with LeJEPA + triplet SSL.

This script is intentionally simpler than the previous masked-reconstruction
training script:

- It reads per-event node features directly from pandas DataFrames.
- It does not construct graph edges.
- It trains only on background jets using multi-view pt-drop augmentation plus corrupted negative views.
- It validates on held-out background jets with the same SSL objective.
- It plots total / invariant / SIGReg / triplet losses.
- It plots the full-jet representation space for background validation jets
  and signal jets without any crop/drop augmentation.

Expected node feature order by default:

    [
        eta, phi, pt, d0/d0Err, dz/dzErr, charge, mass, log_pt,
        pdgId_-211, pdgId_-13, pdgId_-11, pdgId_11,
        pdgId_13, pdgId_22, pdgId_130, pdgId_211,
    ]

Example command:

python -u scripts/run_train_lejepa_trip_part.py \
  --background "data/processed/qcd-vs-wjet-pt-200to400/QCD_scaled_scaled.pkl" \
  --signal "data/processed/qcd-vs-wjet-pt-200to400/WJet_scaled_scaled.pkl" \
  --embed-dim 128 \
  --representation-dim 128 \
  --num-layers 8 \
  --num-heads 8 \
  --batch-size 128 \
  --epochs 50 \
  --learning-rate 5e-4 \
  --weight-decay 5e-2 \
  --precision bf16 \
  --triplet-weight 0.1 \
  --triplet-margin 1.0 \
  --num-global-views 2 \
  --num-local-views 3 \
  --num-negative-views 4 \
  --output-dir "plots/run-lejepa-trip-part"
"""

import argparse
import json
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
from sklearn.metrics import roc_auc_score
from visualize.plot_metrics import plot_anomaly_score, plot_roc_curve

from helpers import helpers_main
from models.part import (
    CorruptedNegativeAugmentationConfig,
    LeJEPALossConfig,
    LeJEPATripletParticleTransformerRepresentation,
    ParticleTransformerConfig,
    MultiViewAugmentationConfig,
    TripletLossConfig,
)
from torch.profiler import (
    profile,
    ProfilerActivity,
    record_function,
    schedule,
)
from contextlib import nullcontext
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
            print(f"Skipping event {i} due to error: {exc}")

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


class TrainLeJEPATripletParticleTransformer:
    """
    Driver class for LeJEPA + corrupted-negative triplet SSL pretraining.
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

        print(f"Loading background from {self.bg_file}")
        print(f"Loading signal from {self.sg_file}")

        self.bg_data = pd.read_pickle(self.bg_file)
        self.sg_data = pd.read_pickle(self.sg_file)

        if self.args.max_background_events is not None:
            self.bg_data = self.bg_data.head(self.args.max_background_events)
        if self.args.max_signal_events is not None:
            self.sg_data = self.sg_data.head(self.args.max_signal_events)

        print(f"Background rows: {len(self.bg_data)}")
        print(f"Signal rows: {len(self.sg_data)}")
        print(f"Background columns: {self.bg_data.columns.tolist()}")
        print(f"Signal columns: {self.sg_data.columns.tolist()}")

        print(f"Background rows: {len(self.bg_data)}")
        print(f"Signal rows: {len(self.sg_data)}")
        print(f"Node feature names: {self.node_feature_names}")

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
            print(
                "Warning: Feature normalization is enabled. This changes eta/phi/pt/mass "
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

        print(f"Loaded background train events: {len(self.bg_train_dataset)}")
        print(f"Loaded background val events: {len(self.bg_val_dataset)}")
        print(f"Loaded signal events: {len(self.sg_dataset)}")
        print(f"Feature mean: {self.feature_mean}")
        print(f"Feature std: {self.feature_std}")
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

    def plot_augmentation_samples(self, train_loader: DataLoader) -> None:
        """
        Plot original background jets and their augmented views before training.

        Augmentations are generated from the actual training DataLoader batches,
        so visualization uses the same batch size, shuffling, collation, and
        padding behavior as training. This is important for batch_mix, which
        requires multiple events in the same batch to provide donor jets.

        Each saved figure corresponds to one event selected from a real training
        batch. The subplots show:
            - original full jet
            - all global pt-drop views
            - all local pt-drop views
            - all negative views

        Plot convention:
            x-axis: phi
            y-axis: eta
            color: pt
        """

        if not hasattr(self, "model"):
            raise RuntimeError("Model must be built before plotting augmentation samples.")

        os.makedirs(self.augmentation_plot_dir, exist_ok=True)

        num_samples = min(
            self.args.num_augmentation_plot_samples,
            len(self.bg_train_dataset),
        )
        if num_samples <= 0:
            return

        self.model.eval()
        num_plotted = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(train_loader):
                x_batch = batch["x"].to(DEVICE, non_blocking=True)
                padding_mask_batch = batch["padding_mask"].to(DEVICE, non_blocking=True)

                views, view_padding_masks, view_types = self.model.augmentation(
                    x=x_batch,
                    padding_mask=padding_mask_batch,
                    return_types=True,
                )

                negative_views, negative_padding_masks, negative_types = (
                    self.model.negative_augmentation(
                        x=x_batch,
                        padding_mask=padding_mask_batch,
                        return_types=True,
                    )
                )

                remaining = num_samples - num_plotted
                if remaining <= 0:
                    break

                rows_this_batch = min(remaining, x_batch.size(0))
                selected_rows = random.sample(
                    range(x_batch.size(0)),
                    k=rows_this_batch,
                )

                for row_idx in selected_rows:
                    panels = [
                        (
                            x_batch[row_idx],
                            padding_mask_batch[row_idx],
                            "original",
                        )
                    ]

                    for view_i, (view_x, view_mask, view_type) in enumerate(
                        zip(views, view_padding_masks, view_types),
                        start=1,
                    ):
                        panels.append(
                            (
                                view_x[row_idx],
                                view_mask[row_idx],
                                f"{view_i}: {view_type}",
                            )
                        )

                    for neg_i, (neg_x, neg_mask, neg_types_for_view) in enumerate(
                        zip(negative_views, negative_padding_masks, negative_types),
                        start=1,
                    ):
                        event_neg_type = neg_types_for_view[row_idx]
                        panels.append(
                            (
                                neg_x[row_idx],
                                neg_mask[row_idx],
                                f"neg {neg_i}: {event_neg_type}",
                            )
                        )

                    output_path = os.path.join(
                        self.augmentation_plot_dir,
                        f"augmentation_sample_{num_plotted + 1:02d}_"
                        f"batch_{batch_idx:04d}_row_{row_idx:03d}.png",
                    )

                    self._plot_single_augmentation_panel(
                        panels=panels,
                        output_path=output_path,
                        title=(
                            f"Training batch {batch_idx}, row {row_idx}: "
                            "original and augmented views"
                        ),
                    )

                    num_plotted += 1
                    if num_plotted >= num_samples:
                        break

                if num_plotted >= num_samples:
                    break

        if num_plotted < num_samples:
            print(
                f"Warning: Requested {num_samples} augmentation plots but only produced {num_plotted}."
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
            print(f"Warning: Skipping augmentation plot with no valid nodes: {output_path}")
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

        shared_eta_min = eta_min - eta_pad
        shared_eta_max = eta_max + eta_pad
        shared_phi_min = phi_min - phi_pad
        shared_phi_max = phi_max + phi_pad

        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(5 * num_cols, 5 * num_rows),
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
            ax.set_xlim(shared_phi_min, shared_phi_max)
            ax.set_ylim(shared_eta_min, shared_eta_max)
            ax.set_box_aspect(1)
            ax.grid(False)

        for ax in axes_flat[num_panels:]:
            ax.axis("off")

        fig.suptitle(title)
        fig.tight_layout(rect=[0.0, 0.0, 0.92, 0.96])

        if last_scatter is not None:
            colorbar_ax = fig.add_axes([0.94, 0.10, 0.018, 0.80])
            fig.colorbar(last_scatter, cax=colorbar_ax, label="pt")

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
            persistent_workers=self.args.num_workers > 0,
        )

        bg_val_loader = DataLoader(
            self.bg_val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=collate_node_tensors,
            persistent_workers=self.args.num_workers > 0,
        )

        signal_loader = DataLoader(
            self.sg_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=collate_node_tensors,
            persistent_workers=self.args.num_workers > 0,
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

        augmentation_config = MultiViewAugmentationConfig(
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

        negative_augmentation_config = CorruptedNegativeAugmentationConfig(
            num_negative_views=self.args.num_negative_views,

            batch_mix_prob=self.args.batch_mix_prob,
            pt_resample_prob=self.args.pt_resample_prob,
            node_eta_phi_rotation_prob=self.args.node_eta_phi_rotation_prob,
            eta_phi_shuffle_prob=self.args.eta_phi_shuffle_prob,
            identity_shuffle_prob=self.args.identity_shuffle_prob,

            min_nodes=self.args.min_nodes,
            eps=self.args.eps,

            eta_index=self.node_feature_names.index("eta"),
            phi_index=self.node_feature_names.index("phi"),
            pt_index=self.node_feature_names.index("pt"),
            d0_index=self.node_feature_names.index("d0/d0Err"),
            dz_index=self.node_feature_names.index("dz/dzErr"),
            charge_index=self.node_feature_names.index("charge"),
            mass_index=self.node_feature_names.index("mass"),
            log_pt_index=self.node_feature_names.index("log_pt"),
            pdg_start_index=self.node_feature_names.index("pdgId_-211"),
            pdg_end_index=self.node_feature_names.index("pdgId_211") + 1,

            corrupt_node_frac=self.args.corrupt_node_frac,
            batch_mix_anchor_frac_min=self.args.batch_mix_anchor_frac_min,
            batch_mix_anchor_frac_max=self.args.batch_mix_anchor_frac_max,
            renormalize_pt_sum=self.args.renormalize_negative_pt_sum,
            renormalize_log_pt_stats=self.args.renormalize_negative_log_pt_stats,
        )

        loss_config = LeJEPALossConfig(
            invariant_weight=self.args.invariant_weight,
            sigreg_weight=self.args.sigreg_weight,
            epps_pulley_num_points=self.args.epps_pulley_num_points,
            num_slices=self.args.num_slices,
            normalize_representations_for_invariant=self.args.normalize_invariant_representations,
            normalize_representations_for_sigreg=self.args.normalize_sigreg_representations,
        )

        triplet_loss_config = TripletLossConfig(
            triplet_weight=self.args.triplet_weight,
            triplet_margin=self.args.triplet_margin,
            normalize_representations_for_triplet=self.args.normalize_triplet_representations,
            use_global_views_as_positives=not self.args.use_all_views_as_triplet_positives,
        )

        self.model = LeJEPATripletParticleTransformerRepresentation(
            model_config=model_config,
            augmentation_config=augmentation_config,
            negative_augmentation_config=negative_augmentation_config,
            loss_config=loss_config,
            triplet_loss_config=triplet_loss_config,
        ).to(DEVICE)

        print(f"Model summary:\n{self.model}")
        print(f"Model summary:\n{self.model}")

        self.num_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Number of trainable parameters: {self.num_params}")
        print(f"Number of trainable parameters: {self.num_params}")

    def plot_progress(
        self,
        train_history: Dict[str, List[float]],
        val_history: Dict[str, List[float]],
        epoch_end_steps: List[int],
        best_val_loss: float,
        auc_history: Dict[str, List[float]],
        mahalanobis_eval_steps: List[int],
    ) -> None:
        """
        Plot train/validation curves for total, invariant, SIGReg, triplet
        losses/distances, plus Mahalanobis anomaly-detection AUC.

        AUC values are plotted at the actual training steps where Mahalanobis
        evaluation was performed.
        """

        if len(train_history["total_loss"]) == 0:
            return

        loss_keys = [
            "total_loss",
            "invariant_loss",
            "sigreg_loss",
            "triplet_loss",
            "triplet_pos_distance",
            "triplet_neg_distance",
        ]
        titles = [
            "Total Loss",
            "Invariant Loss",
            "SIGReg Loss",
            "Triplet Loss",
            "Triplet Positive Distance",
            "Triplet Negative Distance",
        ]

        num_subplots = len(loss_keys) + 1

        fig, axes = plt.subplots(
            num_subplots,
            1,
            figsize=(10, 3.8 * num_subplots),
            sharex=True,
        )

        step_axis = np.arange(1, len(train_history["total_loss"]) + 1)
        epoch_end_steps_np = np.asarray(epoch_end_steps)

        # Plot loss curves
        for ax, key, title in zip(axes[:-1], loss_keys, titles):
            train_values = np.asarray(train_history[key], dtype=np.float64)
            val_values = np.asarray(val_history[key], dtype=np.float64)

            ax.plot(
                step_axis,
                train_values,
                label="Train",
                alpha=0.75,
            )

            if len(val_values) > 0:
                repeat_count = int(
                    np.ceil(len(train_values) / len(val_values))
                )
                repeated_val = np.repeat(
                    val_values,
                    repeat_count,
                )[: len(train_values)]

                ax.plot(
                    step_axis,
                    repeated_val,
                    label="Validation",
                    alpha=0.75,
                )

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
                stride = max(
                    1,
                    int(np.ceil(len(epoch_end_steps_np) / max_labels)),
                )

                for step_idx in epoch_end_steps_np[::stride]:
                    ax.axvline(
                        step_idx,
                        color="gray",
                        ls="--",
                        lw=0.6,
                        alpha=0.25,
                    )

            ax.set_ylabel(title)

            if np.all(train_values > 0):
                ax.set_yscale("log")

            ax.legend()
            ax.grid(False)

        # Mahalanobis AUC subplot
        auc_ax = axes[-1]

        eval_steps = np.asarray(
            mahalanobis_eval_steps,
            dtype=np.int64,
        )
        auc_bgtrain = np.asarray(
            auc_history["auc_bgtrain_vs_signal"],
            dtype=np.float64,
        )
        auc_bgval = np.asarray(
            auc_history["auc_bgval_vs_signal"],
            dtype=np.float64,
        )
        best_auc_bgval = np.nanmax(auc_bgval) if len(auc_bgval) > 0 else np.nan

        if len(eval_steps) > 0:
            auc_ax.step(
                eval_steps,
                auc_bgtrain,
                label="QCD Train vs WJet",
                alpha=0.75,
            )

            auc_ax.step(
                eval_steps,
                auc_bgval,
                label="QCD Val vs WJet",
                alpha=0.75,
            )
            if np.isfinite(best_auc_bgval):
                auc_ax.axhline(
                    y=best_auc_bgval,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.25,
                    label=f"Best Val: {best_auc_bgval:.4f}",
                )

        # Draw vertical lines for epoch boundaries
        if len(epoch_end_steps_np) > 0:
            max_labels = 12
            stride = max(
                1,
                int(np.ceil(len(epoch_end_steps_np) / max_labels)),
            )

            for step_idx in epoch_end_steps_np[::stride]:
                auc_ax.axvline(
                    step_idx,
                    color="gray",
                    ls="--",
                    lw=0.6,
                    alpha=0.25,
                )

        auc_ax.set_ylabel("ROC AUC")
        auc_ax.set_ylim(0.0, 1.0)
        auc_ax.legend()
        auc_ax.grid(False)

        axes[-1].set_xlabel("Step Number")

        # Add secondary x-axis for epoch numbers
        if len(epoch_end_steps_np) > 0:
            epoch_ids = np.arange(1, len(epoch_end_steps_np) + 1)
            max_labels = 12
            stride = max(
                1,
                int(np.ceil(len(epoch_end_steps_np) / max_labels)),
            )

            top_ax = axes[0].secondary_xaxis("top")
            top_ax.set_xticks(epoch_end_steps_np[::stride])
            top_ax.set_xticklabels(epoch_ids[::stride])
            top_ax.set_xlabel("Epoch")

        fig.suptitle("LeJEPA + Triplet SSL Training Progress")
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

    def collect_evaluation_latents(
        self,
        bg_train_loader: DataLoader,
        bg_val_loader: DataLoader,
        signal_loader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Collect all latent arrays needed by both latent-space plotting and
        Mahalanobis/ROC evaluation exactly once.
        """

        print("Collecting background train latents...")
        bg_train_latents = self.collect_representations(bg_train_loader)

        print("Collecting background validation latents...")
        bg_val_latents = self.collect_representations(bg_val_loader)

        print("Collecting signal latents...")
        signal_latents = self.collect_representations(signal_loader)

        print(f"Background train latents: {bg_train_latents.shape}")
        print(f"Background val latents: {bg_val_latents.shape}")
        print(f"Signal latents: {signal_latents.shape}")

        return bg_train_latents, bg_val_latents, signal_latents

    def fit_mahalanobis_background(
        self,
        bg_train_latents: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit a Gaussian background model from background train latents.

        Returns:
            mean: (D,)
            precision: (D, D)
        """

        if bg_train_latents.ndim != 2:
            raise ValueError(
                f"Expected bg_train_latents shape (N, D), got {bg_train_latents.shape}."
            )

        mean = bg_train_latents.mean(axis=0)
        centered = bg_train_latents - mean
        cov = np.cov(centered, rowvar=False)

        if cov.ndim == 0:
            cov = np.asarray([[float(cov)]], dtype=np.float64)

        cov = np.asarray(cov, dtype=np.float64)
        cov = cov + self.args.mahalanobis_cov_eps * np.eye(cov.shape[0], dtype=np.float64)
        precision = np.linalg.pinv(cov)

        return mean.astype(np.float64), precision.astype(np.float64)

    @staticmethod
    def mahalanobis_scores(
        latents: np.ndarray,
        mean: np.ndarray,
        precision: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-event Mahalanobis distance scores.

        No batch averaging is performed. Output shape is (N,).
        """

        latents = np.asarray(latents, dtype=np.float64)
        centered = latents - mean
        # scores = np.einsum("nd,dd,nd->n", centered, precision, centered)
        scores = np.einsum("ni,ij,nj->n", centered, precision, centered)
        return scores.astype(np.float64)

    @staticmethod
    def compute_auc(background_scores: np.ndarray, signal_scores: np.ndarray) -> float:
        y_true = np.concatenate(
            [
                np.zeros(len(background_scores), dtype=np.int64),
                np.ones(len(signal_scores), dtype=np.int64),
            ]
        )
        y_score = np.concatenate([background_scores, signal_scores])
        return float(roc_auc_score(y_true, y_score))

    def evaluate_mahalanobis_for_epoch(
        self,
        bg_train_latents: np.ndarray,
        bg_val_latents: np.ndarray,
        signal_latents: np.ndarray,
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Evaluate Mahalanobis anomaly scores from already-collected latents.

        The background-only Gaussian model is fit on background train latents.
        ROC/AUC is reported for both:
            - background train vs signal
            - background validation vs signal
        
        Returns:
            auc_bgtrain_vs_signal: float
            auc_bgval_vs_signal: float
        """

        mahal_dir = os.path.join(self.output_dir, "mahalanobis_eval", f"epoch_{epoch:04d}")
        os.makedirs(mahal_dir, exist_ok=True)

        np.save(os.path.join(mahal_dir, "background_train_latents.npy"), bg_train_latents)
        np.save(os.path.join(mahal_dir, "background_val_latents.npy"), bg_val_latents)
        np.save(os.path.join(mahal_dir, "signal_latents.npy"), signal_latents)

        mean, precision = self.fit_mahalanobis_background(bg_train_latents)

        background_train_scores = self.mahalanobis_scores(
            bg_train_latents,
            mean,
            precision,
        )
        background_val_scores = self.mahalanobis_scores(
            bg_val_latents,
            mean,
            precision,
        )
        signal_scores = self.mahalanobis_scores(
            signal_latents,
            mean,
            precision,
        )

        auc_bgtrain_vs_signal = self.compute_auc(background_train_scores, signal_scores)
        auc_bgval_vs_signal = self.compute_auc(background_val_scores, signal_scores)

        np.save(
            os.path.join(mahal_dir, "background_train_mahalanobis_scores.npy"),
            background_train_scores,
        )
        np.save(
            os.path.join(mahal_dir, "background_val_mahalanobis_scores.npy"),
            background_val_scores,
        )
        np.save(
            os.path.join(mahal_dir, "signal_mahalanobis_scores.npy"),
            signal_scores,
        )

        metrics = {
            "epoch": int(epoch),
            "auc_bgtrain_vs_signal": float(auc_bgtrain_vs_signal),
            "auc_bgval_vs_signal": float(auc_bgval_vs_signal),
            "background_train_score_mean": float(np.mean(background_train_scores)),
            "background_val_score_mean": float(np.mean(background_val_scores)),
            "signal_score_mean": float(np.mean(signal_scores)),
            "background_train_score_median": float(np.median(background_train_scores)),
            "background_val_score_median": float(np.median(background_val_scores)),
            "signal_score_median": float(np.median(signal_scores)),
            "mahalanobis_cov_eps": float(self.args.mahalanobis_cov_eps),
            "background_train_latent_shape": list(bg_train_latents.shape),
            "background_val_latent_shape": list(bg_val_latents.shape),
            "signal_latent_shape": list(signal_latents.shape),
        }

        metrics_path = os.path.join(mahal_dir, "mahalanobis_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        plot_anomaly_score(
            background_val_scores,
            signal_scores,
            background_label="QCD (Val)",
            signal_label="WJet",
            save_path=os.path.join(mahal_dir, "bgval-vs-signal-mahalanobis-score.png"),
        )

        plot_anomaly_score(
            background_train_scores,
            signal_scores,
            background_label="QCD (Train)",
            signal_label="WJet",
            save_path=os.path.join(mahal_dir, "bgtrain-vs-signal-mahalanobis-score.png"),
        )

        plot_roc_curve(
            background_val_scores,
            signal_scores,
            background_label="QCD (Val)",
            signal_label="WJet",
            savepath=os.path.join(mahal_dir, "roc-bgval-vs-signal-mahalanobis.png"),
            examples=False,
            loss_fn=torch.nn.MSELoss(reduction="mean"),
        )

        plot_roc_curve(
            background_train_scores,
            signal_scores,
            background_label="QCD (Train)",
            signal_label="WJet",
            savepath=os.path.join(mahal_dir, "roc-bgtrain-vs-signal-mahalanobis.png"),
            examples=False,
            loss_fn=torch.nn.MSELoss(reduction="mean"),
        )

        latest_dir = os.path.join(self.output_dir, "mahalanobis_eval", "latest")
        os.makedirs(latest_dir, exist_ok=True)

        latest_metrics_path = os.path.join(latest_dir, "mahalanobis_metrics.json")
        with open(latest_metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Mahalanobis metrics saved to {metrics_path}")
        print(f"Mahalanobis AUC, QCD train vs WJet: {auc_bgtrain_vs_signal:.6f}")
        print(f"Mahalanobis AUC, QCD val vs WJet: {auc_bgval_vs_signal:.6f}")
        
        return auc_bgtrain_vs_signal, auc_bgval_vs_signal
        
    def plot_latent_space_for_epoch(
        self,
        bg_val_latents: np.ndarray,
        signal_latents: np.ndarray,
        epoch: int,
    ) -> None:
        """
        Plot background validation and signal full-jet representations from
        already-collected latents.
        """

        bg_plot_latents = bg_val_latents
        sg_plot_latents = signal_latents

        if self.args.max_latent_plot_points is not None:
            max_points = self.args.max_latent_plot_points
            if len(bg_plot_latents) > max_points:
                bg_indices = np.random.choice(len(bg_plot_latents), max_points, replace=False)
                bg_plot_latents = bg_plot_latents[bg_indices]
            if len(sg_plot_latents) > max_points:
                sg_indices = np.random.choice(len(sg_plot_latents), max_points, replace=False)
                sg_plot_latents = sg_plot_latents[sg_indices]

        bg_2d, sg_2d, x_label, y_label = reduce_to_2d(bg_plot_latents, sg_plot_latents)

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
        self.plot_augmentation_samples(train_loader)

        summary_path = os.path.join(self.output_dir, "summary.json")

        summary = {
            # Run status.
            "status": "initialized",
            "current_epoch": 0,
            "completed_epochs": 0,

            # Data.
            "background": self.bg_file,
            "signal": self.sg_file,
            "node_features": self.node_feature_names,
            "background_train_events": len(self.bg_train_dataset),
            "background_val_events": len(self.bg_val_dataset),
            "signal_events": len(self.sg_dataset),

            # Model.
            "batch_size": self.args.batch_size,
            "embed_dim": self.args.embed_dim,
            "representation_dim": self.args.representation_dim,
            "num_layers": self.args.num_layers,
            "num_heads": self.args.num_heads,
            "ffn_mult": self.args.ffn_mult,
            "dropout": self.args.dropout,
            "num_trainable_parameters": int(self.num_params),

            # Optimization.
            "epochs": self.args.epochs,
            "learning_rate": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
            "precision": self.args.precision,
            "final_lr_ratio": self.args.final_lr_ratio,

            # Positive-view augmentation.
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

            # LeJEPA loss.
            "invariant_weight": self.args.invariant_weight,
            "sigreg_weight": self.args.sigreg_weight,
            "epps_pulley_num_points": self.args.epps_pulley_num_points,
            "num_slices": self.args.num_slices,

            # Triplet loss.
            "triplet_weight": self.args.triplet_weight,
            "triplet_margin": self.args.triplet_margin,
            "normalize_triplet_representations":
                self.args.normalize_triplet_representations,
            "use_all_views_as_triplet_positives":
                self.args.use_all_views_as_triplet_positives,

            # Negative augmentation.
            "num_negative_views": self.args.num_negative_views,
            "batch_mix_prob": self.args.batch_mix_prob,
            "pt_resample_prob": self.args.pt_resample_prob,
            "node_eta_phi_rotation_prob":
                self.args.node_eta_phi_rotation_prob,
            "eta_phi_shuffle_prob": self.args.eta_phi_shuffle_prob,
            "identity_shuffle_prob": self.args.identity_shuffle_prob,
            "corrupt_node_frac": self.args.corrupt_node_frac,
            "batch_mix_anchor_frac_min": self.args.batch_mix_anchor_frac_min,
            "batch_mix_anchor_frac_max": self.args.batch_mix_anchor_frac_max,
            "renormalize_negative_pt_sum":
                self.args.renormalize_negative_pt_sum,
            "renormalize_negative_log_pt_stats":
                self.args.renormalize_negative_log_pt_stats,

            # Evaluation.
            "mahalanobis_eval": not self.args.no_mahalanobis_eval,
            "mahalanobis_cov_eps": self.args.mahalanobis_cov_eps,

            # Misc.
            "normalize_features": self.args.normalize_features,
            "seed": self.args.seed,
            "device": str(DEVICE),

            # Fields populated during training.
            "warmup_steps": None,
            "current_learning_rate": None,
            "best_val_loss": None,
            "final_val_loss": None,
            "latest_train_losses": None,
            "latest_val_losses": None,
        }

        def update_summary(**updates) -> None:
            summary.update(updates)

            temp_path = summary_path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(summary, f, indent=2)

            os.replace(temp_path, summary_path)

        # Write a partial summary immediately, before optimization starts.
        update_summary()
        print(f"Initialized run summary at {summary_path}")
        
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
        
        update_summary(
            status="training",
            warmup_steps=int(warmup_steps),
            total_training_steps=int(total_steps),
            current_learning_rate=float(optimizer.param_groups[0]["lr"]),
        )

        dtype = precision_to_dtype(self.args.precision)
        use_autocast = autocast_enabled_for_precision(self.args.precision)

        train_history = {
            "total_loss": [],
            "invariant_loss": [],
            "sigreg_loss": [],
            "triplet_loss": [],
            "triplet_pos_distance": [],
            "triplet_neg_distance": [],
        }
        val_history = {
            "total_loss": [],
            "invariant_loss": [],
            "sigreg_loss": [],
            "triplet_loss": [],
            "triplet_pos_distance": [],
            "triplet_neg_distance": [],
        }
        auc_history = {
            "auc_bgtrain_vs_signal": [],
            "auc_bgval_vs_signal": [],
        }
        epoch_end_steps: List[int] = []
        mahalanobis_eval_steps: List[int] = []

        best_val_loss = float("inf")
        best_model_path = os.path.join(self.output_dir, "best_model.pth")

        profiler_schedule = schedule(
            wait=2,
            warmup=2,
            active=5,
            repeat=1,
        )
        for epoch in range(1, self.args.epochs + 1):
            print(f"\nEpoch [{epoch}/{self.args.epochs}]")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.8g}")

            self.model.train()

            epoch_train = {
                "total_loss": [],
                "invariant_loss": [],
                "sigreg_loss": [],
                "triplet_loss": [],
                "triplet_pos_distance": [],
                "triplet_neg_distance": [],
            }

            pbar = tqdm(
                train_loader,
                desc=f"Train Epoch {epoch}/{self.args.epochs}",
            )

            # Profile only the first epoch.
            should_profile = self.args.profile and epoch == 1

            profiler_context = (
                profile(
                    activities=[
                        ProfilerActivity.CPU,
                        ProfilerActivity.CUDA,
                    ],
                    schedule=profiler_schedule,
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=False,
                )
                if should_profile
                else nullcontext()
            )

            with profiler_context as prof:
                for step, batch in enumerate(pbar):
                    x = batch["x"].to(
                        DEVICE,
                        non_blocking=True,
                    )
                    padding_mask = batch["padding_mask"].to(
                        DEVICE,
                        non_blocking=True,
                    )

                    optimizer.zero_grad(set_to_none=True)

                    with record_function("training_forward"):
                        with torch.autocast(
                            device_type=DEVICE.type,
                            dtype=dtype,
                            enabled=use_autocast,
                        ):
                            output = self.model.forward_pretrain(
                                x,
                                padding_mask=padding_mask,
                                normalize_output=(
                                    self.args.normalize_output_representations
                                ),
                            )
                            loss = output["total_loss"]

                    with record_function("training_backward"):
                        loss.backward()

                    if self.args.grad_clip_norm is not None:
                        with record_function("gradient_clipping"):
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                max_norm=self.args.grad_clip_norm,
                            )

                    with record_function("optimizer_step"):
                        optimizer.step()

                    with record_function("scheduler_step"):
                        scheduler.step()

                    with record_function("metrics_to_cpu"):
                        loss_values = torch.stack(
                            [
                                output["total_loss"].detach(),
                                output["invariant_loss"].detach(),
                                output["sigreg_loss"].detach(),
                                output["triplet_loss"].detach(),
                                output["triplet_pos_distance"].detach(),
                                output["triplet_neg_distance"].detach(),
                            ]
                        ).float().cpu().tolist()

                    step_losses = {
                        "total_loss": loss_values[0],
                        "invariant_loss": loss_values[1],
                        "sigreg_loss": loss_values[2],
                        "triplet_loss": loss_values[3],
                        "triplet_pos_distance": loss_values[4],
                        "triplet_neg_distance": loss_values[5],
                    }

                    for key in train_history:
                        train_history[key].append(step_losses[key])
                        epoch_train[key].append(step_losses[key])

                    with record_function("progress_bar_update"):
                        if step % 10 == 0:
                            pbar.set_postfix(
                                {
                                    "total": f"{step_losses['total_loss']:.4g}",
                                    "inv": f"{step_losses['invariant_loss']:.4g}",
                                    "sig": f"{step_losses['sigreg_loss']:.4g}",
                                    "tri": f"{step_losses['triplet_loss']:.4g}",
                                    "d+": f"{step_losses['triplet_pos_distance']:.4g}",
                                    "d-": f"{step_losses['triplet_neg_distance']:.4g}",
                                }
                            )
                    # Advance profiler state once per training iteration.
                    if should_profile:
                        prof.step()

            if should_profile:
                print(
                    prof.key_averages().table(
                        sort_by="cuda_time_total",
                        row_limit=50,
                    )
                )

                profile_path = os.path.join(
                    self.output_dir,
                    "l40s_profile_trace.json",
                )
                prof.export_chrome_trace(profile_path)

                print(f"Saved profiler trace to {profile_path}")
            
            mean_train = {
                key: float(np.nanmean(values))
                for key, values in epoch_train.items()
            }

            self.model.eval()
            epoch_val = {
                "total_loss": [],
                "invariant_loss": [],
                "sigreg_loss": [],
                "triplet_loss": [],
                "triplet_pos_distance": [],
                "triplet_neg_distance": [],
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
                        "triplet_loss": float(output["triplet_loss"].detach().cpu()),
                        "triplet_pos_distance": float(output["triplet_pos_distance"].detach().cpu()),
                        "triplet_neg_distance": float(output["triplet_neg_distance"].detach().cpu()),
                    }

                    for key in epoch_val:
                        epoch_val[key].append(step_losses[key])

                    pbar.set_postfix(
                        {
                            "total": f"{step_losses['total_loss']:.4g}",
                            "inv": f"{step_losses['invariant_loss']:.4g}",
                            "sig": f"{step_losses['sigreg_loss']:.4g}",
                            "tri": f"{step_losses['triplet_loss']:.4g}",
                            "d+": f"{step_losses['triplet_pos_distance']:.4g}",
                            "d-": f"{step_losses['triplet_neg_distance']:.4g}",
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
                print(f"Saved new best model to {best_model_path}")

            if ( # Plot latent space and evaluate Mahalanobis only at specified intervals or at the final epoch.
                self.args.latent_plot_every > 0
                and (epoch % self.args.latent_plot_every == 0 or epoch == self.args.epochs)
            ):
                bg_train_latents, bg_val_latents, signal_latents = self.collect_evaluation_latents(
                    bg_train_loader=train_loader,
                    bg_val_loader=bg_val_loader,
                    signal_loader=signal_loader,
                )

                self.plot_latent_space_for_epoch(
                    bg_val_latents=bg_val_latents,
                    signal_latents=signal_latents,
                    epoch=epoch,
                )

                if not self.args.no_mahalanobis_eval:
                    (
                        auc_bgtrain_vs_signal,
                        auc_bgval_vs_signal,
                    ) = self.evaluate_mahalanobis_for_epoch(
                        bg_train_latents=bg_train_latents,
                        bg_val_latents=bg_val_latents,
                        signal_latents=signal_latents,
                        epoch=epoch,
                    )

                    auc_history["auc_bgtrain_vs_signal"].append(
                        float(auc_bgtrain_vs_signal)
                    )
                    auc_history["auc_bgval_vs_signal"].append(
                        float(auc_bgval_vs_signal)
                    )

                    mahalanobis_eval_steps.append(
                        len(train_history["total_loss"])
                    )

            self.plot_progress(
                train_history=train_history,
                val_history=val_history,
                epoch_end_steps=epoch_end_steps,
                best_val_loss=best_val_loss,
                auc_history=auc_history,
                mahalanobis_eval_steps=mahalanobis_eval_steps,
            )
            
            print(f"Epoch {epoch} train losses: {mean_train}")
            print(f"Epoch {epoch} val losses: {mean_val}")
            
            update_summary(
                status="training",
                current_epoch=int(epoch),
                completed_epochs=int(epoch),
                current_learning_rate=float(optimizer.param_groups[0]["lr"]),
                best_val_loss=float(best_val_loss),
                final_val_loss=float(mean_val["total_loss"]),
                latest_train_losses={
                    key: float(value)
                    for key, value in mean_train.items()
                },
                latest_val_losses={
                    key: float(value)
                    for key, value in mean_val.items()
                },
            )

            print(f"Updated run summary at {summary_path}")

        # Final full-jet representation arrays for downstream inspection.
        # Reuse the last evaluation latents when the final epoch already ran
        # latent-space/Mahalanobis evaluation. Otherwise collect them once here.
        if (
            self.args.latent_plot_every > 0
            and (self.args.epochs % self.args.latent_plot_every == 0)
        ):
            bg_train_latents, bg_val_latents, sg_latents = self.collect_evaluation_latents(
                bg_train_loader=train_loader,
                bg_val_loader=bg_val_loader,
                signal_loader=signal_loader,
            )
            np.save(os.path.join(self.output_dir, "background_train_latents.npy"), bg_train_latents)
        else:
            bg_val_latents = self.collect_representations(bg_val_loader)
            sg_latents = self.collect_representations(signal_loader)

        np.save(os.path.join(self.output_dir, "background_val_latents.npy"), bg_val_latents)
        np.save(os.path.join(self.output_dir, "signal_latents.npy"), sg_latents)

        update_summary(
            status="completed",
            current_epoch=int(self.args.epochs),
            completed_epochs=int(self.args.epochs),
            current_learning_rate=float(optimizer.param_groups[0]["lr"]),
            best_val_loss=float(best_val_loss),
            final_val_loss=float(val_history["total_loss"][-1]),
            latest_train_losses={
                key: float(value)
                for key, value in mean_train.items()
            },
            latest_val_losses={
                key: float(value)
                for key, value in mean_val.items()
            },
        )

        history_path = os.path.join(self.output_dir, "loss_history.json")
        with open(history_path, "w") as f:
            json.dump(
                {
                    "train": train_history,
                    "val": val_history,
                    "auc": auc_history,
                    "epoch_end_steps": epoch_end_steps,
                    "mahalanobis_eval_steps": mahalanobis_eval_steps,
                },
                f,
                indent=2,
            )

        print(f"Saved run summary to {summary_path}")
        print(f"Saved loss history to {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train LeJEPA Triplet ParticleTransformer Representation",
        description="Train a ParticleTransformer representation model with LeJEPA + corrupted-negative triplet SSL.",
    )
    # Profile
    parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Profile a short training window in the first epoch.",
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
        default=(
            "eta,phi,pt,d0/d0Err,dz/dzErr,charge,mass,log_pt,"
            "pdgId_-211,pdgId_-13,pdgId_-11,pdgId_11,"
            "pdgId_13,pdgId_22,pdgId_130,pdgId_211"
        ),
        help=(
            "Comma-separated node feature list. Default: eta,phi,pt,d0/d0Err,dz/dzErr,"
            "charge,mass,log_pt,pdgId_-211,pdgId_-13,pdgId_-11,pdgId_11,"
            "pdgId_13,pdgId_22,pdgId_130,pdgId_211."
        ),
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
        default=8,
        help="Number of Transformer encoder layers. Default: 8.",
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

    # Triplet / corrupted-negative objective.
    parser.add_argument(
        "--triplet-weight",
        type=float,
        default=0.1,
        help="Weight for corrupted-negative triplet loss. Default: 0.1.",
    )
    parser.add_argument(
        "--triplet-margin",
        type=float,
        default=1.0,
        help="Margin in ReLU(d_pos - d_neg + margin). Default: 1.0.",
    )
    parser.add_argument(
        "--normalize-triplet-representations",
        action="store_true",
        help="L2-normalize representations before triplet distance computation.",
    )
    parser.add_argument(
        "--use-all-views-as-triplet-positives",
        action="store_true",
        help="Use all global/local LeJEPA views as triplet positives instead of only global views.",
    )
    parser.add_argument(
        "--num-negative-views",
        type=int,
        default=4,
        help="Number of corrupted negative views generated per batch. Default: 4.",
    )
    parser.add_argument(
        "--batch-mix-prob",
        type=float,
        default=0.45,
        help="Probability of sampling batch_mix for a negative view. Default: 0.45.",
    )
    parser.add_argument(
        "--pt-resample-prob",
        type=float,
        default=0.25,
        help="Probability of sampling pt_resample for a negative view. Default: 0.25.",
    )
    parser.add_argument(
        "--node-eta-phi-rotation-prob",
        type=float,
        default=0.20,
        help="Probability of sampling independent node-level eta-phi rotation. Default: 0.20.",
    )
    parser.add_argument(
        "--eta-phi-shuffle-prob",
        type=float,
        default=0.05,
        help="Probability of sampling eta_phi_shuffle for a negative view. Default: 0.05.",
    )
    parser.add_argument(
        "--identity-shuffle-prob",
        type=float,
        default=0.05,
        help="Probability of sampling identity_shuffle for a negative view. Default: 0.05.",
    )
    parser.add_argument(
        "--corrupt-node-frac",
        type=float,
        default=1.0,
        help="Fraction of valid nodes corrupted in within-event negative modes. Default: 1.0.",
    )
    parser.add_argument(
        "--batch-mix-anchor-frac-min",
        type=float,
        default=0.1,
        help="Minimum fraction of anchor-event nodes kept in batch_mix negatives. Default: 0.1.",
    )
    parser.add_argument(
        "--batch-mix-anchor-frac-max",
        type=float,
        default=0.9,
        help="Maximum fraction of anchor-event nodes kept in batch_mix negatives. Default: 0.9.",
    )
    parser.add_argument(
        "--renormalize-negative-pt-sum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Renormalize corrupted negative pt sums to the original event pt sum. Default: True.",
    )
    parser.add_argument(
        "--renormalize-negative-log-pt-stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match corrupted negative log_pt mean/std to the original event. Default: True.",
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
        default="plots/run-lejepa-trip-part",
        help="Directory for plots, checkpoints, losses, and summary.json.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers. Default: 4.",
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
        "--no-mahalanobis-eval",
        action="store_true",
        help="Disable Mahalanobis anomaly-score and ROC evaluation during latent plotting epochs.",
    )
    parser.add_argument(
        "--mahalanobis-cov-eps",
        type=float,
        default=1e-4,
        help="Diagonal regularization added to the background latent covariance. Default: 1e-4.",
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

    trainer = TrainLeJEPATripletParticleTransformer()
    trainer.load()
    trainer.build_node_datasets()
    trainer.plot_features()
    trainer.train()