"""Train one LeJEPA semi-supervised triplet ParticleTransformer on JetClass or CMS.

Each dataset module exposes PARTICLE_FEATURES and BATCH_NORMALIZED_FEATURES.
CMS ROOT shards are already shuffled across production families and pT ranges,
then split randomly inside each jet type and streamed with the same label-
stratified active-pool design as JetClass. Cross-dataset checkpoint loading
preserves schema-independent weights
while reinitializing the node embedding and feature-normalization state.

Example commands:

Train on CMS dataset: (use --dataset [cms | jetclass])
python -u \
    scripts/run_train_lejepa_part.py \
    --dataset cms \
    --dataset-root "/HEP/export/home/hgao50/jet-anomaly-data/ak8-v4" \
    --model semi-sup-triplet \
    --background-labels "label_QCD,label_Hbb,label_Zqq,label_Tbqq" \
    --signal-labels "label_Wqq" \
    --embed-dim 32 \
    --representation-dim 32 \
    --dropout 0.01 \
    --num-layers 4 \
    --num-heads 8 \
    --batch-size 256 \
    --steps-per-epoch 4000 \
    --val-steps 3000 \
    --eval-steps 3000 \
    --epochs 12 \
    --learning-rate 1e-3 \
    --weight-decay 5e-2 \
    --precision bf16 \
    --num-global-views 2 \
    --num-local-views 4 \
    --num-negative-views 4 \
    --batch-mix-prob 0.4 \
    --pt-resample-prob 0.25 \
    --node-deta-dphi-rotation-prob 0.1 \
    --deta-dphi-shuffle-prob 0.1 \
    --identity-shuffle-prob 0.15 \
    --global-drop-pt-frac-min 0.0 \
    --global-drop-pt-frac-max 0.3 \
    --local-drop-pt-frac-min 0.3 \
    --local-drop-pt-frac-max 0.75 \
    --batch-mix-anchor-frac-min 0.4 \
    --batch-mix-anchor-frac-max 0.6 \
    --anomaly-score mahalanobis \
    --pairwise-hidden-dim 32 \
    --triplet-weight 0.1 \
    --triplet-margin 0.2 \
    --classification-weight 0.1 \
    --num-workers 3 \
    --prefetch-factor 2 \
    --shuffle-active-shards 4 \
    --output-dir "plots/cms-wqq"

Finetune a JetClass model on CMS dataset: (specify --checkpoint and --checkpoint-summary)
python -u \
    scripts/run_train_lejepa_part.py \
    --dataset cms \
    --dataset-root /HEP/export/home/hgao50/jet-anomaly-data/ak8-v4 \
    --model semi-sup-triplet \
    --background-labels label_QCD,label_Hbb \
    --signal-labels label_Wqq \
    --checkpoint plots/old-jetclass-run/best_model.pth \
    --checkpoint-summary plots/old-jetclass-run/summary.json \
    --output-dir plots/finetune-cms

Fine-tune a CMS MC checkpoint on real data with a fresh 50/5/45 split:
python -u scripts/run_train_lejepa_part.py \
    --dataset cms \
    --dataset-root /path/to/shuffled/real-data \
    --background-labels label_Real \
    --cms-val-fraction 0.05 \
    --cms-test-fraction 0.45 \
    --classification-weight 0 \
    --checkpoint /path/to/cms-mc-run/best_model.pth \
    --checkpoint-summary /path/to/cms-mc-run/summary.json \
    --output-dir /path/to/new/real-data-finetune-run

Use DDP training: (4 GPU example)
Replace 
    python -u 
with
    torchrun --standalone --nproc-per-node=4
"""

import argparse
import json
import math
import os
import random
import sys
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

# Add parent directory to import local project modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.cms_streaming import (
    BATCH_NORMALIZED_FEATURES as CMS_BATCH_NORMALIZED_FEATURES,
    PARTICLE_FEATURES as CMS_PARTICLE_FEATURES,
    CMS_FEATURE_SOURCES,
    CMSIterableDataset,
    CMS_LABELS,
    CMS_TO_JETCLASS_FEATURE_MAP,
    cms_split_manifest,
    collate_cms_tensors,
    discover_cms_files,
    split_cms_files,
    validate_cms_labels,
)
from datasets.jetclass_streaming import (
    BATCH_NORMALIZED_FEATURES as JETCLASS_BATCH_NORMALIZED_FEATURES,
    PARTICLE_FEATURES as JETCLASS_PARTICLE_FEATURES,
    JETCLASS_LABELS,
    JetClassIterableDataset,
    collate_jetclass_tensors,
    validate_requested_labels as validate_jetclass_labels,
)
from models.part_jetclass import (
    CorruptedNegativeAugmentationConfig,
    LeJEPALossConfig,
    SemiSupervisedLossConfig,
    ParticleTransformerConfig,
    MultiViewAugmentationConfig,
    TripletLossConfig,
    LeJEPASemiSupervisedTripletParticleTransformerRepresentation,
)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

FEATURE_SCHEMA_VERSION = "dataset-module-macros-v3"
ONLY_MODEL_NAME = "semi-sup-triplet"
FOUR_VECTOR_FEATURE_PREFIX = [
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
]


def validate_four_vector_prefix(feature_names: Sequence[str], *, source: str) -> None:
    prefix = list(feature_names[:4])
    if prefix != FOUR_VECTOR_FEATURE_PREFIX:
        raise ValueError(
            f"{source} must begin with the ordered four-momentum features "
            f"{FOUR_VECTOR_FEATURE_PREFIX}, found {prefix}."
        )


class PretrainForwardAdapter(torch.nn.Module):
    """Expose forward_pretrain through nn.Module.forward for DDP."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self,
        x_particles: torch.Tensor,
        y: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.model.forward_pretrain(
            x_particles,
            y,
            padding_mask=padding_mask,
        )


def parse_csv_list(value: str) -> List[str]:
    """Parse a comma-separated CLI list while preserving item order."""
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated item.")
    return items


def unique_preserving_order(items: Sequence[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _torch_load_checkpoint(path: str, device: torch.device):
    """Load checkpoints on old and new PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _strip_state_dict_prefixes(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = dict(state_dict)
    for prefix in ("module.", "model."):
        if cleaned and all(key.startswith(prefix) for key in cleaned):
            cleaned = {key.removeprefix(prefix): value for key, value in cleaned.items()}
    return cleaned

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
    """Training driver for the single supported semi-supervised triplet model."""
    
    def _prepare_labels_for_model(self, y: torch.Tensor) -> torch.Tensor:
        """Project the dataset one-hot axis to the selected background order."""

        indices = [
            self.dataset_label_axis.index(label)
            for label in self.background_labels
        ]
        return y[:, indices]
    
    def _seed_rank_rng(self) -> None:
        """Create independent stochastic streams on each DDP rank."""

        rank_seed = (
            self.base_seed
            + self.rank
        )

        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                rank_seed
            )

    def __init__(self):
        self.args = parser.parse_args()

        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.distributed = self.world_size > 1

        if self.distributed:
            if not torch.cuda.is_available():
                raise RuntimeError("DDP multi-GPU training requires CUDA.")
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            self.device = torch.device("cuda", self.local_rank)
            global DEVICE
            DEVICE = self.device
        else:
            self.device = DEVICE
        self.is_main_process = self.rank == 0

        if self.args.batch_size % self.world_size != 0:
            raise ValueError(
                "--batch-size must be divisible by "
                f"world_size={self.world_size}, got {self.args.batch_size}."
            )
        self.per_rank_batch_size = self.args.batch_size // self.world_size

        self.model_name = self.args.model
        if self.model_name != ONLY_MODEL_NAME:
            raise ValueError(
                f"Only --model {ONLY_MODEL_NAME!r} is supported, got "
                f"{self.model_name!r}."
            )
        self.ssl_metric_keys = [
            "total_loss",
            "invariant_loss",
            "sigreg_loss",
            "triplet_loss",
            "triplet_pos_distance",
            "triplet_neg_distance",
            "classification_loss",
        ]

        self.dataset_name = self.args.dataset
        self.dataset_root = self.args.dataset_root
        if self.args.background_labels is None:
            default_backgrounds = {
                "jetclass": "label_QCD,label_Hbb,label_Hcc",
                "cms": "label_QCD,label_Hbb",
            }
            self.background_labels = parse_csv_list(default_backgrounds[self.dataset_name])
        else:
            self.background_labels = parse_csv_list(self.args.background_labels)
        if self.args.signal_labels is None:
            self.signal_labels = []
        else:
            self.signal_labels = parse_csv_list(self.args.signal_labels)

        # A one-class categorical objective is mathematically vacuous: with a
        # single logit, cross-entropy is identically zero. Require an explicit
        # opt-out so a real-data fine-tuning command cannot silently inherit the
        # multi-class default classification weight.
        if self.args.classification_weight is None:
            if len(self.background_labels) == 1:
                raise ValueError(
                    "Single-label training requires explicitly passing "
                    "--classification-weight 0. The classification objective "
                    "is undefined as a useful learning signal with one class."
                )
            self.args.classification_weight = 0.1
        elif (
            len(self.background_labels) == 1
            and not math.isclose(
                float(self.args.classification_weight),
                0.0,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise ValueError(
                "Single-label training requires --classification-weight 0; "
                f"got {self.args.classification_weight}."
            )

        self.background_display_name = "+".join(
            label.removeprefix("label_") for label in self.background_labels
        )
        self.signal_display_name = "+".join(
            label.removeprefix("label_") for label in self.signal_labels
        )
        if self.dataset_name == "jetclass":
            self.particle_feature_names = list(JETCLASS_PARTICLE_FEATURES)
            self.batch_normalized_feature_names = list(
                JETCLASS_BATCH_NORMALIZED_FEATURES
            )
        elif self.dataset_name == "cms":
            self.particle_feature_names = list(CMS_PARTICLE_FEATURES)
            self.batch_normalized_feature_names = list(
                CMS_BATCH_NORMALIZED_FEATURES
            )
        else:
            raise ValueError(f"Unsupported dataset {self.dataset_name!r}.")

        if not self.particle_feature_names:
            raise ValueError(
                f"{self.dataset_name}.PARTICLE_FEATURES must not be empty."
            )
        missing_normalized = sorted(
            set(self.batch_normalized_feature_names)
            - set(self.particle_feature_names)
        )
        if missing_normalized:
            raise ValueError(
                f"{self.dataset_name}.BATCH_NORMALIZED_FEATURES contains "
                f"features absent from PARTICLE_FEATURES: {missing_normalized}."
            )
        validate_four_vector_prefix(
            self.particle_feature_names,
            source=f"{self.dataset_name} dataset feature schema",
        )
        self.pt_index = self.particle_feature_names.index("part_pt")

        all_requested_labels = unique_preserving_order(
            self.background_labels + self.signal_labels
        )
        if self.dataset_name == "jetclass":
            self.dataset_label_axis = list(JETCLASS_LABELS)
            validate_jetclass_labels(all_requested_labels)
            self.collate_fn = collate_jetclass_tensors
            self.train_dir = os.path.join(self.dataset_root, "train_100M")
            self.val_dir = os.path.join(self.dataset_root, "val_5M")
            self.test_dir = os.path.join(self.dataset_root, "test_20M")
            for split_dir in (self.train_dir, self.val_dir, self.test_dir):
                if not os.path.isdir(split_dir):
                    raise FileNotFoundError(
                        f"Missing JetClass split directory: {split_dir}"
                    )
            self.cms_splits = None
            self.cms_manifest = None
        elif self.dataset_name == "cms":
            self.dataset_label_axis = list(CMS_LABELS)
            validate_cms_labels(all_requested_labels)
            self.collate_fn = collate_cms_tensors
            discovered = discover_cms_files(
                self.dataset_root,
                all_requested_labels,
            )
            self.cms_splits = split_cms_files(
                discovered,
                val_fraction=self.args.cms_val_fraction,
                test_fraction=self.args.cms_test_fraction,
                seed=self.args.cms_split_seed,
            )
            self.cms_manifest = cms_split_manifest(self.cms_splits)
        else:
            raise ValueError(f"Unsupported dataset {self.dataset_name!r}.")

        self.output_dir = self.args.output_dir
        if self.args.checkpoint is not None and not self.args.resume_training_state:
            checkpoint_parent = Path(self.args.checkpoint).expanduser().resolve().parent
            output_path = Path(self.output_dir).expanduser().resolve()
            if checkpoint_parent == output_path:
                raise ValueError(
                    "Weight-only fine-tuning must use a new --output-dir. The "
                    "checkpoint run directory is treated as immutable, including "
                    "its cms_split_manifest.json."
                )
        self.feature_plot_dir = os.path.join(self.output_dir, "features")
        self.augmentation_plot_dir = os.path.join(
            self.output_dir, "augmentation_views"
        )
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.feature_plot_dir, exist_ok=True)
        os.makedirs(self.augmentation_plot_dir, exist_ok=True)

        if self.is_main_process and self.cms_manifest is not None:
            # This manifest is always built from the current dataset root,
            # requested labels, split fractions, and split seed. Checkpoint
            # metadata and any manifest beside the checkpoint are intentionally
            # ignored for a new fine-tuning run.
            manifest_path = os.path.join(self.output_dir, "cms_split_manifest.json")
            with open(manifest_path + ".tmp", "w") as handle:
                json.dump(self.cms_manifest, handle, indent=2)
            os.replace(manifest_path + ".tmp", manifest_path)

        self.base_seed = int(self.args.seed)
        random.seed(self.base_seed)
        np.random.seed(self.base_seed)
        torch.manual_seed(self.base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.base_seed)

        self.checkpoint_payload = None
        self.checkpoint_metadata: Dict[str, object] = {}
        self.checkpoint_load_info: Dict[str, object] = {
            "path": self.args.checkpoint,
            "loaded": False,
            "feature_schema_changed": False,
            "normalization_schema_changed": False,
            "node_embedding_reset": False,
            "feature_normalization_state_reset": False,
            "classification_head_reset": False,
        }

    def load(self) -> None:
        """Construct lazy streaming datasets for the selected backend."""
        if self.is_main_process:
            print(f"Dataset: {self.dataset_name}")
            print(f"Dataset root: {self.dataset_root}")
            print(f"Background labels: {self.background_labels}")
            print(f"Signal labels: {self.signal_labels}")
            print(f"Particle features: {self.particle_feature_names}")
            print(
                "Batch-normalized particle features: "
                f"{self.batch_normalized_feature_names}"
            )

        if self.dataset_name == "jetclass":
            dataset_kwargs = {
                "particle_features": self.particle_feature_names,
                "max_num_particles": self.args.max_num_particles,
                "shuffle_active_shards": self.args.shuffle_active_shards,
                "shuffle_files": True,
                "seed": self.args.seed,
                "rank": self.rank,
                "world_size": self.world_size,
            }
            self.bg_train_dataset = JetClassIterableDataset(
                split_dir=self.train_dir,
                labels_to_load=self.background_labels,
                max_events=self.args.max_train_events,
                infinite=True,
                **dataset_kwargs,
            )
            self.bg_train_eval_dataset = JetClassIterableDataset(
                split_dir=self.train_dir,
                labels_to_load=self.background_labels,
                max_events=self.args.max_val_events,
                infinite=True,
                **dataset_kwargs,
            )
            self.bg_val_dataset = JetClassIterableDataset(
                split_dir=self.val_dir,
                labels_to_load=self.background_labels,
                max_events=self.args.max_val_events,
                infinite=True,
                **dataset_kwargs,
            )
            self.bg_test_dataset = JetClassIterableDataset(
                split_dir=self.test_dir,
                labels_to_load=self.background_labels,
                max_events=self.args.max_test_background_events,
                infinite=True,
                **dataset_kwargs,
            )
            self.sg_dataset = (
                JetClassIterableDataset(
                    split_dir=self.test_dir,
                    labels_to_load=self.signal_labels,
                    max_events=self.args.max_test_signal_events,
                    **dataset_kwargs,
                )
                if self.signal_labels
                else None
            )
        else:
            assert self.cms_splits is not None

            def make_cms_dataset(
                split_name: str,
                labels: Sequence[str],
                max_events: Optional[int],
                seed_offset: int,
            ) -> CMSIterableDataset:
                return CMSIterableDataset(
                    files_by_label=self.cms_splits[split_name],
                    labels_to_load=labels,
                    label_axis=self.dataset_label_axis,
                    particle_features=self.particle_feature_names,
                    max_num_particles=self.args.max_num_particles,
                    min_nodes=self.args.min_nodes,
                    lowerpt=self.args.cms_pt_min,
                    upperpt=self.args.cms_pt_max,
                    max_events=max_events,
                    shuffle_files=True,
                    shuffle_active_shards=self.args.shuffle_active_shards,
                    infinite=True,
                    seed=self.args.seed + seed_offset,
                    rank=self.rank,
                    world_size=self.world_size,
                )

            self.bg_train_dataset = make_cms_dataset(
                "train", self.background_labels, self.args.max_train_events, 0
            )
            self.bg_train_eval_dataset = make_cms_dataset(
                "train", self.background_labels, self.args.max_val_events, 101
            )
            self.bg_val_dataset = make_cms_dataset(
                "val", self.background_labels, self.args.max_val_events, 202
            )
            self.bg_test_dataset = make_cms_dataset(
                "test",
                self.background_labels,
                self.args.max_test_background_events,
                303,
            )
            self.sg_dataset = (
                make_cms_dataset(
                    "test", self.signal_labels, self.args.max_test_signal_events, 404
                )
                if self.signal_labels
                else None
            )

        datasets = [
            dataset
            for dataset in (
                self.bg_train_dataset,
                self.bg_train_eval_dataset,
                self.bg_val_dataset,
                self.bg_test_dataset,
                self.sg_dataset,
            )
            if dataset is not None
        ]
        feature_lists = [list(dataset.feature_names) for dataset in datasets]
        normalized_lists = [
            list(dataset.batch_normalized_feature_names) for dataset in datasets
        ]
        if any(names != feature_lists[0] for names in feature_lists[1:]):
            raise RuntimeError(
                f"Dataset objects disagree on feature order: {feature_lists}"
            )
        if any(names != normalized_lists[0] for names in normalized_lists[1:]):
            raise RuntimeError(
                "Dataset objects disagree on batch-normalized features: "
                f"{normalized_lists}"
            )
        self.particle_feature_names = feature_lists[0]
        self.batch_normalized_feature_names = normalized_lists[0]
        validate_four_vector_prefix(
            self.particle_feature_names,
            source=f"{self.dataset_name} dataset object",
        )
        self.pt_index = self.particle_feature_names.index("part_pt")

    def build_node_datasets(self) -> None:
        """Report lazy dataset metadata without materializing any events."""

        if not self.is_main_process:
            return
        print(
            f"Lazy {self.dataset_name} datasets ready; no full split has been "
            "loaded into RAM."
        )
        print(f"Background train ROOT shards: {len(self.bg_train_dataset.filepaths)}")
        print(f"Background val ROOT shards: {len(self.bg_val_dataset.filepaths)}")
        print(f"Background test ROOT shards: {len(self.bg_test_dataset.filepaths)}")
        if self.sg_dataset is None:
            print("Signal test stream: disabled")
        else:
            print(f"Signal test ROOT shards: {len(self.sg_dataset.filepaths)}")
        print(
            "Per-event particle shape: "
            f"({self.args.max_num_particles}, {len(self.particle_feature_names)})"
        )
        if self.dataset_name == "cms":
            print(
                "CMS active shards per worker: requested="
                f"{self.args.shuffle_active_shards}, effective="
                f"{self.bg_train_dataset.effective_active_shards}; "
                "at least one active shard per requested jet type."
            )

    def plot_features(self) -> None:
        """Plot particle features from a bounded streaming training sample."""
        if not self.is_main_process:
            return

        os.makedirs(self.feature_plot_dir, exist_ok=True)
        sampled_rows: List[np.ndarray] = []
        sampled_events = 0

        for event_x, _ in self.bg_train_dataset:
            valid = ~event_x.eq(0).all(dim=-1)
            if valid.any():
                sampled_rows.append(event_x[valid].numpy())

            sampled_events += 1
            if sampled_events >= self.args.feature_plot_events:
                break

        if not sampled_rows:
            print("Warning: no valid particles found for feature plots.")
            return

        all_features = np.concatenate(sampled_rows, axis=0)

        for i, name in enumerate(self.particle_feature_names):
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
            fig.savefig(
                os.path.join(self.feature_plot_dir, f"feature_{i}_{name}.png")
            )
            plt.close(fig)

    def make_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader]:
        train_kwargs = {
            "batch_size": self.per_rank_batch_size,
            "num_workers": self.args.num_workers,
            "pin_memory": self.args.pin_memory,
            "collate_fn": self.collate_fn,
            "persistent_workers": self.args.num_workers > 0,
            "drop_last": True,
        }

        # Match diagnostics: evaluation uses the same worker count and
        # prefetch factor recorded for training. Persistent workers are disabled
        # so each bounded evaluation starts from a freshly initialized,
        # deterministic dataset iterator.
        eval_kwargs = {
            "batch_size": self.per_rank_batch_size,
            "num_workers": self.args.num_workers,
            "pin_memory": self.args.pin_memory,
            "collate_fn": self.collate_fn,
            "persistent_workers": False,
            "drop_last": True,
        }

        if self.args.num_workers > 0:
            train_kwargs["prefetch_factor"] = self.args.prefetch_factor
            eval_kwargs["prefetch_factor"] = self.args.prefetch_factor

        # IterableDataset owns sample order; DataLoader shuffle is invalid here.
        train_loader = DataLoader(self.bg_train_dataset, **train_kwargs)
        # train_eval_loader is used for collecting train set representations only
        # it has the same data as train_loader
        train_eval_loader = DataLoader(self.bg_train_eval_dataset, **eval_kwargs)
        bg_val_loader = DataLoader(self.bg_val_dataset, **eval_kwargs)
        bg_test_loader = DataLoader(self.bg_test_dataset, **eval_kwargs)
        signal_loader = (
            DataLoader(self.sg_dataset, **eval_kwargs)
            if self.sg_dataset is not None
            else None
        )

        return train_loader, train_eval_loader, bg_val_loader, bg_test_loader, signal_loader

    def _extract_ssl_metrics(
        self,
        output: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """
        Extract the active model's scalar training metrics.

        The metric set depends on --model.
        """

        metrics: Dict[str, float] = {}

        for key in self.ssl_metric_keys:
            if key not in output:
                raise KeyError(
                    f"Expected metric {key!r} in model output. "
                    f"Available keys: {list(output.keys())}"
                )

            metrics[key] = float(
                self._distributed_mean(output[key]).cpu()
            )

        return metrics

    def _distributed_mean(
        self,
        value: torch.Tensor,
    ) -> torch.Tensor:
        value = value.detach().float()

        if not self.distributed:
            return value

        value = value.clone()
        dist.all_reduce(
            value,
            op=dist.ReduceOp.SUM,
        )
        value /= self.world_size

        return value

    def _progress_postfix(
        self,
        metrics: Dict[str, float],
    ) -> Dict[str, str]:
        """Compact tqdm postfix for the single supported objective."""
        return {
            "total": f"{metrics['total_loss']:.4g}",
            "inv": f"{metrics['invariant_loss']:.4g}",
            "sig": f"{metrics['sigreg_loss']:.4g}",
            "tri": f"{metrics['triplet_loss']:.4g}",
            "d+": f"{metrics['triplet_pos_distance']:.4g}",
            "d-": f"{metrics['triplet_neg_distance']:.4g}",
            "cls": f"{metrics['classification_loss']:.4g}",
        }
        
    def _all_gather_event_tensor(
        self,
        tensor: torch.Tensor,
        event_dim: int,
    ) -> torch.Tensor:
        if not self.distributed:
            return tensor

        tensor = tensor.to(
            DEVICE,
            non_blocking=True,
        )
        tensor = tensor.movedim(
            event_dim,
            0,
        ).contiguous()

        local_size = torch.tensor(
            [tensor.size(0)],
            device=DEVICE,
            dtype=torch.long,
        )

        sizes = [
            torch.zeros_like(local_size)
            for _ in range(self.world_size)
        ]
        dist.all_gather(
            sizes,
            local_size,
        )

        sizes_int = [
            int(size.item())
            for size in sizes
        ]
        max_size = max(sizes_int)

        if tensor.size(0) < max_size:
            pad_shape = (
                max_size - tensor.size(0),
                *tensor.shape[1:],
            )
            padding = torch.zeros(
                pad_shape,
                device=tensor.device,
                dtype=tensor.dtype,
            )
            tensor = torch.cat(
                [tensor, padding],
                dim=0,
            )

        gathered = [
            torch.empty_like(tensor)
            for _ in range(self.world_size)
        ]
        dist.all_gather(
            gathered,
            tensor,
        )

        merged = torch.cat(
            [
                rank_tensor[:rank_size]
                for rank_tensor, rank_size
                in zip(gathered, sizes_int)
            ],
            dim=0,
        )

        return merged.movedim(
            0,
            event_dim,
        ).cpu()
        
    def _read_checkpoint(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        checkpoint_path = Path(self.args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
        payload = _torch_load_checkpoint(str(checkpoint_path), DEVICE)
        self.checkpoint_payload = payload

        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
        elif isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
        elif isinstance(payload, dict) and payload and all(
            isinstance(value, torch.Tensor) for value in payload.values()
        ):
            state_dict = payload
        else:
            raise TypeError(
                "Unsupported checkpoint format. Expected a raw state_dict or a "
                "dictionary containing model_state_dict/state_dict."
            )
        state_dict = _strip_state_dict_prefixes(state_dict)

        metadata: Dict[str, object] = {}
        if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
            metadata.update(payload["metadata"])

        if self.args.checkpoint_summary is not None:
            summary_path = Path(self.args.checkpoint_summary).expanduser().resolve()
        else:
            summary_path = checkpoint_path.parent / "summary.json"
        if summary_path.is_file():
            with open(summary_path) as handle:
                summary_metadata = json.load(handle)
            summary_metadata.update(metadata)
            metadata = summary_metadata
        self.checkpoint_metadata = metadata
        return state_dict, metadata

    @staticmethod
    def _classification_state_keys(model: torch.nn.Module) -> List[str]:
        return [
            key for key in model.state_dict()
            if key.startswith("classification_head.")
        ]

    @staticmethod
    def _node_embedding_state_keys(model: torch.nn.Module) -> List[str]:
        return [
            key for key in model.state_dict()
            if key.startswith("node_embedding.")
        ]

    @staticmethod
    def _feature_normalization_state_keys(model: torch.nn.Module) -> List[str]:
        names = {
            "_feature_running_mean",
            "_feature_running_var",
            "_feature_num_batches_tracked",
        }
        return [key for key in model.state_dict() if key in names]

    def _load_checkpoint_into_model(self, core_model: torch.nn.Module) -> None:
        if self.args.checkpoint is None:
            return

        state_dict, metadata = self._read_checkpoint()
        source_dataset = metadata.get("dataset")
        if source_dataset is None and metadata:
            source_dataset = "jetclass"
        source_features = metadata.get("particle_features")
        source_model = metadata.get("model")
        source_backgrounds = metadata.get("background_labels")
        source_normalized_features = metadata.get(
            "batch_standardized_particle_features"
        )

        if source_model is not None and str(source_model) != ONLY_MODEL_NAME:
            raise ValueError(
                f"Checkpoint model variant {source_model!r} is not the supported "
                f"model {ONLY_MODEL_NAME!r}."
            )

        if source_features is not None:
            source_features = list(source_features)
            validate_four_vector_prefix(
                source_features, source="checkpoint feature schema"
            )
        elif source_dataset is not None and str(source_dataset) != self.dataset_name:
            raise ValueError(
                "Cross-dataset checkpoint loading requires particle_features "
                "metadata. Place summary.json beside the checkpoint or pass "
                "--checkpoint-summary."
            )

        feature_schema_changed = (
            source_features is not None
            and source_features != self.particle_feature_names
        )
        normalization_schema_changed = (
            source_normalized_features is not None
            and list(source_normalized_features)
            != self.batch_normalized_feature_names
        )
        dataset_changed = (
            source_dataset is not None and str(source_dataset) != self.dataset_name
        )
        class_order_changed = (
            source_backgrounds is not None
            and list(source_backgrounds) != self.background_labels
        )
        reset_classification_head = (
            dataset_changed
            or class_order_changed
            or self.args.reset_classification_head
        )
        reset_node_embedding = (
            feature_schema_changed or normalization_schema_changed
        )

        classification_keys = self._classification_state_keys(core_model)
        node_embedding_keys = self._node_embedding_state_keys(core_model)
        feature_norm_keys = self._feature_normalization_state_keys(core_model)
        if not classification_keys:
            raise RuntimeError(
                "Could not identify classification_head parameters in the model."
            )
        if not node_embedding_keys:
            raise RuntimeError(
                "Could not identify node_embedding parameters in the model."
            )

        reset_keys = set()
        if reset_classification_head:
            reset_keys.update(classification_keys)
        if reset_node_embedding:
            reset_keys.update(node_embedding_keys)
            reset_keys.update(feature_norm_keys)

        filtered_state = {
            key: value for key, value in state_dict.items() if key not in reset_keys
        }
        current_state = core_model.state_dict()
        shape_mismatches = [
            (key, tuple(value.shape), tuple(current_state[key].shape))
            for key, value in filtered_state.items()
            if key in current_state and current_state[key].shape != value.shape
        ]
        if shape_mismatches:
            raise ValueError(
                "Checkpoint parameter shapes do not match after applying the "
                f"requested resets: {shape_mismatches[:10]}"
            )

        load_result = core_model.load_state_dict(filtered_state, strict=False)
        disallowed_missing = sorted(
            set(load_result.missing_keys) - reset_keys
        )
        if disallowed_missing or load_result.unexpected_keys:
            raise RuntimeError(
                "Checkpoint load was not architecture-compatible. "
                f"missing={disallowed_missing}, "
                f"unexpected={load_result.unexpected_keys}."
            )

        if self.args.resume_training_state:
            if reset_classification_head or reset_node_embedding:
                raise ValueError(
                    "--resume-training-state cannot be used when the "
                    "classification head or node embedding is reinitialized. "
                    "Use weight-only fine-tuning instead."
                )
            if not isinstance(self.checkpoint_payload, dict) or not all(
                key in self.checkpoint_payload
                for key in ("optimizer_state_dict", "scheduler_state_dict", "epoch")
            ):
                raise ValueError(
                    "--resume-training-state requires a full training checkpoint "
                    "created by this script."
                )
            if dataset_changed or class_order_changed:
                raise ValueError(
                    "Full optimizer/scheduler resume requires the same dataset "
                    "and background class order."
                )

        self.checkpoint_load_info = {
            "path": str(Path(self.args.checkpoint).expanduser().resolve()),
            "loaded": True,
            "source_dataset": source_dataset,
            "source_background_labels": source_backgrounds,
            "source_particle_features": source_features,
            "source_batch_standardized_particle_features": (
                list(source_normalized_features)
                if source_normalized_features is not None
                else None
            ),
            "feature_schema_changed": feature_schema_changed,
            "normalization_schema_changed": normalization_schema_changed,
            "node_embedding_reset": reset_node_embedding,
            "feature_normalization_state_reset": reset_node_embedding,
            "classification_head_reset": reset_classification_head,
            "resume_training_state": bool(self.args.resume_training_state),
        }
        if self.is_main_process:
            print(f"Loaded checkpoint: {self.checkpoint_load_info['path']}")
            if reset_node_embedding:
                print(
                    "Checkpoint feature or normalization schema differs from the "
                    "current dataset; node_embedding and running feature-"
                    "normalization state remain "
                    "at their new random/default initialization."
                )
            if reset_classification_head:
                print(
                    "Classification head remains at its new random initialization."
                )

    def _checkpoint_metadata_payload(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset_name,
            "dataset_root": self.dataset_root,
            "model": ONLY_MODEL_NAME,
            "background_labels": list(self.background_labels),
            "signal_labels": list(self.signal_labels),
            "dataset_label_axis": list(self.dataset_label_axis),
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "particle_features": list(self.particle_feature_names),
            "batch_standardized_particle_features": list(
                self.batch_normalized_feature_names
            ),
            "full_event_latent_space": "representation",
        }

    def build_model(self) -> None:
        model_config = ParticleTransformerConfig(
            input_dim=len(self.particle_feature_names),
            input_feature_names=tuple(self.particle_feature_names),
            standardized_feature_names=tuple(
                self.batch_normalized_feature_names
            ),
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
            px_index=self.particle_feature_names.index("part_px"),
            py_index=self.particle_feature_names.index("part_py"),
            pz_index=self.particle_feature_names.index("part_pz"),
            energy_index=self.particle_feature_names.index("part_energy"),
            pt_index=self.particle_feature_names.index("part_pt"),
            log_pt_fraction_index=self.particle_feature_names.index(
                "log_pt_fraction"
            ),
            eps=self.args.eps,
            pt_drop_power=self.args.pt_drop_power,
            zero_dropped_features=not self.args.keep_dropped_features,
        )

        negative_augmentation_config = CorruptedNegativeAugmentationConfig(
            num_negative_views=self.args.num_negative_views,
            batch_mix_prob=self.args.batch_mix_prob,
            pt_resample_prob=self.args.pt_resample_prob,
            node_deta_dphi_rotation_prob=(
                self.args.node_deta_dphi_rotation_prob
            ),
            deta_dphi_shuffle_prob=self.args.deta_dphi_shuffle_prob,
            identity_shuffle_prob=self.args.identity_shuffle_prob,
            min_nodes=self.args.min_nodes,
            eps=self.args.eps,
            deta_index=self.particle_feature_names.index("part_deta"),
            dphi_index=self.particle_feature_names.index("part_dphi"),
            pt_index=self.particle_feature_names.index("part_pt"),
            log_pt_fraction_index=self.particle_feature_names.index(
                "log_pt_fraction"
            ),
            charge_index=self.particle_feature_names.index("part_charge"),
            identity_start_index=self.particle_feature_names.index(
                "part_isChargedHadron"
            ),
            identity_end_index=self.particle_feature_names.index(
                "part_isMuon"
            ) + 1,
            corrupt_node_frac=self.args.corrupt_node_frac,
            batch_mix_anchor_frac_min=self.args.batch_mix_anchor_frac_min,
            batch_mix_anchor_frac_max=self.args.batch_mix_anchor_frac_max,
            renormalize_pt_sum=self.args.renormalize_negative_pt_sum,
        )

        loss_config = LeJEPALossConfig(
            invariant_weight=self.args.invariant_weight,
            sigreg_weight=self.args.sigreg_weight,
            epps_pulley_num_points=self.args.epps_pulley_num_points,
            num_slices=self.args.num_slices,
        )
        triplet_loss_config = TripletLossConfig(
            triplet_weight=self.args.triplet_weight,
            triplet_margin=self.args.triplet_margin,
            use_global_views_as_positives=(
                not self.args.use_all_views_as_triplet_positives
            ),
        )
        semi_supervised_loss_config = SemiSupervisedLossConfig(
            classification_weight=self.args.classification_weight,
            num_classes=len(self.background_labels),
        )

        core_model = LeJEPASemiSupervisedTripletParticleTransformerRepresentation(
            model_config=model_config,
            augmentation_config=augmentation_config,
            negative_augmentation_config=negative_augmentation_config,
            loss_config=loss_config,
            triplet_loss_config=triplet_loss_config,
            semi_supervised_config=semi_supervised_loss_config,
        ).to(DEVICE)

        self._load_checkpoint_into_model(core_model)

        self.num_params = sum(
            parameter.numel()
            for parameter in core_model.parameters()
            if parameter.requires_grad
        )
        if self.is_main_process:
            print(f"Selected SSL model: {ONLY_MODEL_NAME}")
            print(f"Number of trainable parameters: {self.num_params:,}")

        self.model_core = core_model
        adapter = PretrainForwardAdapter(core_model).to(DEVICE)
        if self.distributed:
            self.model = DDP(
                adapter,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        else:
            self.model = adapter

    def plot_progress(
        self,
        train_history: Dict[str, List[float]],
        val_history: Dict[str, List[float]],
        epoch_end_steps: List[int],
        best_val_loss: float,
        auc_history: Dict[str, Dict[str, Dict[str, List[float]]]],
        roc_eval_steps: List[int],
    ) -> None:
        """Plot loss curves and one pairwise ROC-AUC panel per signal type."""

        if len(train_history["total_loss"]) == 0:
            return

        title_map = {
            "total_loss": "Total Loss",
            "invariant_loss": "Invariant Loss",
            "sigreg_loss": "SIGReg Loss",
            "classification_loss": "Classification Loss",
            "triplet_loss": "Triplet Loss",
            "triplet_pos_distance": "Triplet Positive Distance",
            "triplet_neg_distance": "Triplet Negative Distance",
        }
        loss_keys = list(self.ssl_metric_keys)
        signal_panels = [
            signal
            for signal in self.signal_labels
            if signal in auc_history
        ]
        num_subplots = len(loss_keys) + len(signal_panels)

        fig, axes = plt.subplots(
            num_subplots,
            1,
            figsize=(10, 3.8 * num_subplots),
            sharex=True,
            layout="constrained",
        )
        axes = np.atleast_1d(axes)

        step_axis = np.arange(1, len(train_history["total_loss"]) + 1)
        epoch_end_steps_np = np.asarray(epoch_end_steps)

        for ax, key in zip(axes[: len(loss_keys)], loss_keys):
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
                stride = max(1, int(np.ceil(len(epoch_end_steps_np) / 12)))
                for step_idx in epoch_end_steps_np[::stride]:
                    ax.axvline(step_idx, color="gray", ls="--", lw=0.6, alpha=0.25)

            ax.set_ylabel(title_map[key])
            if np.all(train_values > 0):
                ax.set_yscale("log")
            ax.legend()
            ax.grid(False)

        eval_steps = np.asarray(roc_eval_steps, dtype=np.int64)
        default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

        for panel_index, signal_label in enumerate(signal_panels):
            ax = axes[len(loss_keys) + panel_index]
            signal_name = signal_label.removeprefix("label_")
            pair_history = auc_history[signal_label]

            for background_index, background_label in enumerate(self.background_labels):
                # Background classes are shown against the configured test signal.
                #     continue
                if background_label not in pair_history:
                    continue
                background_name = background_label.removeprefix("label_")
                pair_values = pair_history[background_label]
                train_heldout_auc = np.asarray(
                    pair_values.get("train_heldout", []), dtype=np.float64
                )
                val_auc = np.asarray(pair_values.get("val", []), dtype=np.float64)
                color = (
                    default_colors[background_index % len(default_colors)]
                    if default_colors
                    else None
                )

                if len(train_heldout_auc) > 0:
                    train_heldout_last = float(train_heldout_auc[-1])
                    train_heldout_steps = np.concatenate(
                        ([0], eval_steps[: len(train_heldout_auc)])
                    )  # Show the first epoch value over the 0-1 epoch interval.
                    train_heldout_auc = np.concatenate(
                        ([train_heldout_auc[0]], train_heldout_auc)
                    )
                    ax.step(
                        train_heldout_steps,
                        train_heldout_auc,
                        where="pre",
                        linestyle="-",
                        color=color,
                        alpha=0.85,
                        label=(
                            f"{background_name} Held-out train vs {signal_name} "
                            f"(Last: {train_heldout_last:.4f})"
                        ),
                    )
                if len(val_auc) > 0:
                    val_last = float(val_auc[-1])
                    val_steps = np.concatenate(([0], eval_steps[: len(val_auc)]))
                    val_auc = np.concatenate(([val_auc[0]], val_auc))
                    ax.step(
                        val_steps,
                        val_auc,
                        where="pre",
                        linestyle="--",
                        color=color,
                        alpha=0.85,
                        label=(
                            f"{background_name} Val vs {signal_name} "
                            f"(Last: {val_last:.4f})"
                        ),
                    )

            ax.axhline(
                0.5,
                color="black",
                linestyle="--",
                linewidth=1,
                alpha=0.45,
            )
            if len(epoch_end_steps_np) > 0:
                stride = max(1, int(np.ceil(len(epoch_end_steps_np) / 12)))
                for step_idx in epoch_end_steps_np[::stride]:
                    ax.axvline(step_idx, color="gray", ls="--", lw=0.6, alpha=0.25)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("ROC AUC")
            ax.set_title(f"Pairwise ROC AUC with {signal_name} as visualization signal")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=8, ncol=2)
            ax.grid(False)

        axes[-1].set_xlabel("Step Number")

        if len(epoch_end_steps_np) > 0:
            epoch_ids = np.arange(1, len(epoch_end_steps_np) + 1)
            stride = max(1, int(np.ceil(len(epoch_end_steps_np) / 12)))
            top_ax = axes[0].secondary_xaxis("top")
            top_ax.set_xticks(epoch_end_steps_np[::stride])
            top_ax.set_xticklabels(epoch_ids[::stride])
            top_ax.set_xlabel("Epoch")

        fig.suptitle(
            "LeJEPA + Triplet + Semi-Supervised Classification Training Progress",
            fontsize=15
        )
        fig.savefig(
            os.path.join(self.output_dir, "loss.png"),
            bbox_inches="tight",
        )
        plt.close(fig)

    @torch.no_grad()
    def collect_representations(
        self,
        loader: DataLoader,
        return_labels: bool = False,
    ):
        """
        Compute full-jet LeJEPA representations without augmentation/crop/drop.

        The full-event CLS state is always passed through ``representation_head``
        so ROC evaluation uses the same representation space optimized by the
        LeJEPA, SIGReg, and triplet objectives.

        When ``return_labels=True``, also return the original dataset one-hot
        labels gathered in exactly the same event order as the latent tensor.
        """

        self.model.eval()
        latents: List[torch.Tensor] = []
        labels: List[torch.Tensor] = []
        dtype = precision_to_dtype(self.args.precision)
        use_autocast = autocast_enabled_for_precision(self.args.precision)

        loader_iter = iter(loader)

        for batch_idx in tqdm(
            range(self.args.eval_steps),
            desc="Collecting representations",
            leave=False,
            disable=not self.is_main_process,
        ):
            batch = next(loader_iter)
            x_particles = batch["x_particles"].to(DEVICE, non_blocking=True)
            padding_mask = batch["padding_mask"].to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=dtype,
                enabled=use_autocast,
            ):
                cls = self.model_core(
                    x_particles,
                    padding_mask=padding_mask,
                )
                z = self.model_core.representation_head(cls)

            latents.append(z.detach().float().cpu())
            if return_labels:
                labels.append(batch["y"].detach().cpu())

        latents = torch.cat(latents, dim=0)
        latents = self._all_gather_event_tensor(
            latents,
            event_dim=0,
        )

        if not return_labels:
            return latents

        labels_tensor = torch.cat(labels, dim=0)
        labels_tensor = self._all_gather_event_tensor(
            labels_tensor,
            event_dim=0,
        )
        return latents, labels_tensor

    @torch.no_grad()
    def collect_view_representations(
        self,
        dataloader,
        which_view: Literal["view", "negative", "all"] = "view",
        return_labels: bool = False,
    ):
        """
        Collect augmented view representations for a full dataset.

        Unlike collect_representations(), this function calls:

            self.model.forward_pretrain(...)

        so that the model's existing MultiViewAugmentation is reused exactly.

        Output:
            z_views:
                Tensor of shape:

                    (V, N_events, D)

                where V is the total number of positive views.
            
            z_negatives:
                Tensor of shape:

                    (K, N_events, D)

                where K is the total number of negative views.

        Important:
            If which_view is set to "view", the z_negatives is returned as None.
            If which_view is set to "negative", the z_views is returned as None.
        """
        
        assert which_view in {"view", "negative", "all"}, (
            f"Invalid which_view: {which_view}. Choose from 'view', 'negative', or 'all'."
        )

        self.model.eval()

        collected_z_views = []
        collected_z_negatives = []
        collected_labels = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(
                    dataloader,
                    total=self.args.eval_steps,
                    desc="Collecting positive-view representations",
                    leave=False,
                    disable=not self.is_main_process,
                )
            ):
                if batch_idx >= self.args.eval_steps:
                    break
                x_particles = batch["x_particles"].to(DEVICE, non_blocking=True)
                raw_y = batch["y"]
                if return_labels:
                    collected_labels.append(raw_y.detach().cpu())
                y = raw_y.to(DEVICE, non_blocking=True)
                y = self._prepare_labels_for_model(y)
                padding_mask = batch["padding_mask"].to(DEVICE, non_blocking=True)

                dtype = precision_to_dtype(self.args.precision)
                use_autocast = autocast_enabled_for_precision(self.args.precision)
        
                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=dtype,
                    enabled=use_autocast,
                ):
                    output = self.model( # collect view representations
                        x_particles,
                        y,
                        padding_mask=padding_mask,
                    )
                
                if which_view == "view" or which_view == "all":
                    # Shape:
                    #     (V, B, D)
                    z_views = output["z_views"]

                    collected_z_views.append(
                        z_views.detach().float().cpu()
                    )
                if which_view == "negative" or which_view == "all":
                    # Shape:
                    #     (K, B, D)
                    z_negatives = output["z_negatives"]

                    collected_z_negatives.append(
                        z_negatives.detach().float().cpu()
                    )

        if which_view == "view" or which_view == "all":
            # Concatenate over event/batch dimension:
            #     list of (V, B_i, D)
            # ->  (V, sum_i B_i, D)
            z_views = torch.cat(
                collected_z_views,
                dim=1,
            )
            z_views = self._all_gather_event_tensor(
                z_views,
                event_dim=1,
            )
        if which_view == "negative" or which_view == "all":
            # Concatenate over event/batch dimension:
            #     list of (K, B_i, D)
            # ->  (K, sum_i B_i, D)
            z_negatives = torch.cat(
                collected_z_negatives,
                dim=1,
            )
            z_negatives = self._all_gather_event_tensor(
                z_negatives,
                event_dim=1,
            )

        gathered_labels = None
        if return_labels:
            gathered_labels = torch.cat(collected_labels, dim=0)
            gathered_labels = self._all_gather_event_tensor(
                gathered_labels,
                event_dim=0,
            )

        if which_view == "view":
            result = (z_views, None)
        elif which_view == "negative":
            result = (None, z_negatives)
        else:  # which_view == "all"
            result = (z_views, z_negatives)

        if return_labels:
            return (*result, gathered_labels)
        return result

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
    def local_global_consistency_scores(
        z_views: torch.Tensor,
        num_global_views: int,
    ) -> torch.Tensor:
        """
        Compute one local-global consistency anomaly score per event.

        Input:
            z_views:
                Tensor of shape:

                    (V, B, D)

                where:
                    V = total number of positive views
                    B = batch size / number of events
                    D = representation dimension

                View ordering must match MultiViewAugmentation:

                    first G views:
                        global views

                    remaining V - G views:
                        local views

            num_global_views:
                Number of global views G.

        Output:
            scores:
                Tensor of shape:

                    (B,)

                One anomaly score per event.

        Definition:
            For event b, first construct the global anchor:

                anchor_b
                    = mean_g z_global[g, b]

            Then compute the mean squared representation distance from every
            local view to that event-specific global anchor:

                score_b
                    = mean_l mean_d
                        (z_local[l, b, d] - anchor_b[d])^2

        Interpretation:
            Larger score means the event's local representations are less
            consistent with its global representation.

            The encoder is trained only on QCD positive-view consistency, so
            an unseen anomaly may receive a larger score if its local and global
            structure do not satisfy the learned QCD consistency relation.
        """

        if z_views.ndim != 3:
            raise ValueError(
                "Expected z_views shape (V, B, D), got "
                f"{tuple(z_views.shape)}."
            )

        num_views = z_views.size(0)

        if not (1 <= num_global_views < num_views):
            raise ValueError(
                "local_global_consistency_scores requires at least one global "
                "and one local view. Got "
                f"num_global_views={num_global_views}, "
                f"num_views={num_views}."
            )

        # Shape:
        #     global_views: (G, B, D)
        #     local_views:  (L, B, D)
        global_views = z_views[:num_global_views]
        local_views = z_views[num_global_views:]

        # Event-specific global anchor:
        #
        #     anchor_b = mean over global views
        #
        # Shape:
        #     (B, D)
        anchor = global_views.mean(dim=0)

        # Per-local-view MSE to the event-specific global anchor.
        #
        # Shape after mean over representation dimension:
        #     (L, B)
        local_anchor_mse = (
            local_views
            - anchor.unsqueeze(0)
        ).square().mean(dim=-1)

        # Mean over local views.
        #
        # Shape:
        #     (B,)
        scores = local_anchor_mse.mean(dim=0)

        return scores

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

    @torch.no_grad()
    def compute_mahalanobis_anomaly_scores(
        self,
        background_train_loader,
        background_val_loader,
        signal_loader,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Fit one Gaussian per background class and compute pairwise AUCs.

        This follows the diagnostics protocol exactly for the background train
        sample:

            1. Collect the bounded train-evaluation representation sample.
            2. Split every background class independently into 50% fit and 50%
               held-out subsets with seed ``base_seed + 404``.
            3. Fit one class-specific Gaussian using only that class's fit half.
            4. Report held-out-train and validation AUCs against the same
               test-split signal sample.

        No event used to fit a Gaussian contributes to the reported
        held-out-train AUC.
        """

        bg_train_latents, bg_train_labels = self.collect_representations(
            background_train_loader,
            return_labels=True,
        )
        bg_val_latents, bg_val_labels = self.collect_representations(
            background_val_loader,
            return_labels=True,
        )
        signal_latents, signal_labels = self.collect_representations(
            signal_loader,
            return_labels=True,
        )

        # Every rank must participate in the representation gathers above, but
        # only rank 0 performs NumPy fitting and ROC bookkeeping.
        if not self.is_main_process:
            return {}

        bg_train_latents_np = bg_train_latents.numpy()
        bg_val_latents_np = bg_val_latents.numpy()
        signal_latents_np = signal_latents.numpy()
        bg_train_ids = self._dataset_label_ids(bg_train_labels.numpy())
        bg_val_ids = self._dataset_label_ids(bg_val_labels.numpy())
        signal_ids = self._dataset_label_ids(signal_labels.numpy())

        fit_indices, heldout_indices = self._stratified_fit_heldout_indices(
            bg_train_ids,
            labels=self.background_labels,
            fit_fraction=0.5,
            seed=self.base_seed + 404,
        )

        results: Dict[str, Dict[str, Dict[str, float]]] = {
            signal_label: {} for signal_label in self.signal_labels
        }

        for background_label in self.background_labels:
            background_index = self.dataset_label_axis.index(background_label)
            class_train_indices = np.flatnonzero(
                bg_train_ids == background_index
            )
            class_fit_indices = fit_indices[
                np.isin(fit_indices, class_train_indices)
            ]
            class_heldout_indices = heldout_indices[
                np.isin(heldout_indices, class_train_indices)
            ]
            val_mask = bg_val_ids == background_index

            num_fit = int(len(class_fit_indices))
            num_heldout = int(len(class_heldout_indices))
            num_val = int(np.count_nonzero(val_mask))
            if num_fit < 2 or num_heldout < 1 or num_val < 1:
                warnings.warn(
                    "Skipping an insufficient class-specific Mahalanobis sample: "
                    f"background={background_label}, fit={num_fit}, "
                    f"heldout={num_heldout}, val={num_val}."
                )
                continue

            mean, precision = self.fit_mahalanobis_background(
                bg_train_latents_np[class_fit_indices]
            )
            background_train_heldout_scores = self.mahalanobis_scores(
                bg_train_latents_np[class_heldout_indices],
                mean,
                precision,
            )
            background_val_scores = self.mahalanobis_scores(
                bg_val_latents_np[val_mask],
                mean,
                precision,
            )
            all_signal_scores = self.mahalanobis_scores(
                signal_latents_np,
                mean,
                precision,
            )

            for signal_label in self.signal_labels:
                signal_index = self.dataset_label_axis.index(signal_label)
                signal_mask = signal_ids == signal_index
                if not np.any(signal_mask):
                    warnings.warn(
                        "No test events were collected for signal label "
                        f"{signal_label!r}."
                    )
                    continue

                signal_scores = all_signal_scores[signal_mask]
                results[signal_label][background_label] = {
                    "train_heldout": self.compute_auc(
                        background_train_heldout_scores,
                        signal_scores,
                    ),
                    "val": self.compute_auc(
                        background_val_scores,
                        signal_scores,
                    ),
                }

        return results

    @torch.no_grad()
    def compute_local_global_anomaly_scores(
        self,
        background_train_loader,
        background_val_loader,
        signal_loader,
    ):
        """Compute local-global scores while retaining original dataset labels."""

        bg_train_latents, _, bg_train_labels = self.collect_view_representations(
            background_train_loader,
            which_view="view",
            return_labels=True,
        )
        bg_val_latents, _, bg_val_labels = self.collect_view_representations(
            background_val_loader,
            which_view="view",
            return_labels=True,
        )
        signal_latents, _, signal_labels = self.collect_view_representations(
            signal_loader,
            which_view="view",
            return_labels=True,
        )

        return (
            self.local_global_consistency_scores(
                bg_train_latents, self.args.num_global_views
            ),
            bg_train_labels,
            self.local_global_consistency_scores(
                bg_val_latents, self.args.num_global_views
            ),
            bg_val_labels,
            self.local_global_consistency_scores(
                signal_latents, self.args.num_global_views
            ),
            signal_labels,
        )

    @staticmethod
    def _dataset_label_ids(labels: np.ndarray) -> np.ndarray:
        if labels.ndim == 1:
            return labels.astype(np.int64, copy=False)
        if labels.ndim != 2:
            raise ValueError(f"Expected class ids or one-hot labels, got {labels.shape}.")
        return np.argmax(labels, axis=1).astype(np.int64, copy=False)

    def _stratified_fit_heldout_indices(
        self,
        class_ids: np.ndarray,
        *,
        labels: Sequence[str],
        fit_fraction: float,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return deterministic class-stratified fit and held-out indices.

        The implementation matches ``diagnose_lejepa_latents.py``: every
        requested class contributes independently to both subsets, with at
        least two events retained on each side.
        """

        if not 0.0 < fit_fraction < 1.0:
            raise ValueError("fit_fraction must lie strictly between 0 and 1.")

        class_ids = np.asarray(class_ids, dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        fit_parts: List[np.ndarray] = []
        heldout_parts: List[np.ndarray] = []

        for label in labels:
            label_index = self.dataset_label_axis.index(label)
            indices = np.flatnonzero(class_ids == label_index)
            if len(indices) < 4:
                raise RuntimeError(
                    f"Only {len(indices)} sampled train-evaluation events found "
                    f"for {label}; at least four are required for a 50/50 "
                    "fit/held-out split."
                )
            rng.shuffle(indices)
            cut = min(
                max(int(round(len(indices) * fit_fraction)), 2),
                len(indices) - 2,
            )
            fit_parts.append(indices[:cut])
            heldout_parts.append(indices[cut:])

        fit_indices = np.concatenate(fit_parts)
        heldout_indices = np.concatenate(heldout_parts)
        rng.shuffle(fit_indices)
        rng.shuffle(heldout_indices)
        return fit_indices, heldout_indices

    def _empty_auc_history(self) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        return {
            signal_label: {
                background_label: {"train_heldout": [], "val": []}
                for background_label in self.background_labels
            }
            for signal_label in self.signal_labels
        }

    def evaluate_anomaly_score_for_epoch(
        self,
        bg_train_loader,
        bg_val_loader,
        sg_loader,
        score_fn,
        score_name,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute pairwise AUCs for the configured anomaly score.

        Mahalanobis fitting is class-specific and is completed directly inside
        ``compute_mahalanobis_anomaly_scores`` because each background class
        defines a different score for the same signal events. Local-global scores
        remain class-independent and are regrouped below by dataset label.
        """

        if sg_loader is None or not self.signal_labels:
            return {}

        if score_name == "mahalanobis":
            return score_fn(bg_train_loader, bg_val_loader, sg_loader)

        (
            background_train_scores,
            background_train_labels,
            background_val_scores,
            background_val_labels,
            signal_scores,
            signal_labels,
        ) = score_fn(bg_train_loader, bg_val_loader, sg_loader)

        if not self.is_main_process:
            return {}

        def scores_to_numpy(scores) -> np.ndarray:
            if isinstance(scores, torch.Tensor):
                return scores.detach().float().cpu().numpy()
            return np.asarray(scores, dtype=np.float64)

        def labels_to_numpy(labels) -> np.ndarray:
            if isinstance(labels, torch.Tensor):
                return labels.detach().cpu().numpy()
            return np.asarray(labels)

        background_train_scores = scores_to_numpy(background_train_scores)
        background_val_scores = scores_to_numpy(background_val_scores)
        signal_scores = scores_to_numpy(signal_scores)
        background_train_ids = self._dataset_label_ids(
            labels_to_numpy(background_train_labels)
        )
        background_val_ids = self._dataset_label_ids(
            labels_to_numpy(background_val_labels)
        )
        signal_ids = self._dataset_label_ids(labels_to_numpy(signal_labels))
        _, train_heldout_indices = self._stratified_fit_heldout_indices(
            background_train_ids,
            labels=self.background_labels,
            fit_fraction=0.5,
            seed=self.base_seed + 404,
        )
        train_heldout_mask = np.zeros(
            len(background_train_ids), dtype=bool
        )
        train_heldout_mask[train_heldout_indices] = True

        results: Dict[str, Dict[str, Dict[str, float]]] = {}
        for signal_label in self.signal_labels:
            signal_index = self.dataset_label_axis.index(signal_label)
            signal_mask = signal_ids == signal_index
            if not np.any(signal_mask):
                warnings.warn(
                    f"No test events were collected for signal label {signal_label!r}."
                )
                continue

            per_background: Dict[str, Dict[str, float]] = {}
            for background_label in self.background_labels:
                background_index = self.dataset_label_axis.index(background_label)
                train_mask = (
                    (background_train_ids == background_index)
                    & train_heldout_mask
                )
                val_mask = background_val_ids == background_index
                if not np.any(train_mask) or not np.any(val_mask):
                    warnings.warn(
                        "Skipping an empty pairwise ROC sample: "
                        f"background={background_label}, signal={signal_label}."
                    )
                    continue
                per_background[background_label] = {
                    "train_heldout": self.compute_auc(
                        background_train_scores[train_mask], signal_scores[signal_mask]
                    ),
                    "val": self.compute_auc(
                        background_val_scores[val_mask], signal_scores[signal_mask]
                    ),
                }
            results[signal_label] = per_background

        return results

    def train(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        train_loader, train_eval_loader, bg_val_loader, bg_test_loader, signal_loader = self.make_dataloaders()
        self.build_model()

        summary_path = os.path.join(self.output_dir, "summary.json")

        # Write a first summary of the run
        summary = {
            "model": ONLY_MODEL_NAME,
            "dataset": self.dataset_name,
            "dataset_label_axis": self.dataset_label_axis,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "source_particle_features": list(self.particle_feature_names),
            "checkpoint": self.checkpoint_load_info,
            "best_model_path": os.path.join(self.output_dir, "best_model.pth"),
            "last_model_path": os.path.join(self.output_dir, "last_model.pth"),
            "best_checkpoint_path": os.path.join(self.output_dir, "best_checkpoint.pt"),
            "last_checkpoint_path": os.path.join(self.output_dir, "last_checkpoint.pt"),
            
            # Run status.
            "status": "initialized",
            "current_epoch": 0,
            "completed_epochs": 0,

            # Data.
            "dataset_root": self.dataset_root,
            "background_labels": self.background_labels,
            "signal_labels": self.signal_labels,
            "particle_features": self.particle_feature_names,
            "particle_feature_order_is_strict": True,
            "batch_standardized_particle_features": list(
                self.batch_normalized_feature_names
            ),
            "full_event_latent_space": "representation",
            "background_train_root_shards": len(self.bg_train_dataset.filepaths),
            "background_val_root_shards": len(self.bg_val_dataset.filepaths),
            "background_test_root_shards": len(self.bg_test_dataset.filepaths),
            "signal_test_root_shards": (
                len(self.sg_dataset.filepaths) if self.sg_dataset is not None else 0
            ),
            "max_train_events": self.args.max_train_events,
            "max_val_events": self.args.max_val_events,
            "max_test_background_events": self.args.max_test_background_events,
            "max_test_signal_events": self.args.max_test_signal_events,

            # Model.
            "batch_size": self.args.batch_size,
            "embed_dim": self.args.embed_dim,
            "representation_dim": self.args.representation_dim,
            "num_layers": self.args.num_layers,
            "num_heads": self.args.num_heads,
            "ffn_mult": self.args.ffn_mult,
            "dropout": self.args.dropout,
            "num_trainable_parameters": int(self.num_params),
            "use_pairwise_bias": not self.args.no_pairwise_bias,
            "pairwise_hidden_dim": self.args.pairwise_hidden_dim,
            "pairwise_num_features": self.args.pairwise_num_features,
            "max_num_particles": self.args.max_num_particles,
            "eps": self.args.eps,
            "keep_dropped_features": self.args.keep_dropped_features,

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

            # Evaluation.
            "anomaly_score": self.args.anomaly_score,
            "evaluation_num_workers": self.args.num_workers,
            "evaluation_prefetch_factor": self.args.prefetch_factor,
            "mahalanobis_fit_fraction": 0.5,
            "mahalanobis_split_seed_offset": 404,
            "mahalanobis_cov_eps": self.args.mahalanobis_cov_eps,
            "pairwise_train_metric": "heldout_train",
            "signal_test_seed_offset": 404,

            "base_seed": self.args.seed,
            "device": str(DEVICE),

            # Fields populated during training.
            "warmup_steps": None,
            "current_learning_rate": None,
            "best_val_loss": None,
            "final_val_loss": None,
            "latest_train_losses": None,
            "latest_val_losses": None,
            
            "distributed": self.distributed,
            "train_infinite_stream": True,
            "world_size": self.world_size,
            "global_batch_size": self.args.batch_size,
            "per_rank_batch_size": self.per_rank_batch_size,
            "steps_per_epoch": self.args.steps_per_epoch,
            "val_steps": self.args.val_steps,
            "eval_steps": self.args.eval_steps,
            "num_workers": self.args.num_workers,
            "prefetch_factor": self.args.prefetch_factor,
            "shuffle_active_shards": self.args.shuffle_active_shards,
        }

        if self.dataset_name == "cms":
            summary.update(
                {
                    "cms_pt_min": self.args.cms_pt_min,
                    "cms_pt_max": self.args.cms_pt_max,
                    "cms_val_fraction": self.args.cms_val_fraction,
                    "cms_test_fraction": self.args.cms_test_fraction,
                    "cms_split_seed": self.args.cms_split_seed,
                    "cms_split_manifest_sha256": self.cms_manifest["sha256"],
                    "cms_split_manifest_policy": (
                        "fresh-current-run-manifest; checkpoint manifest ignored"
                    ),
                    "cms_split_root_shard_counts": {
                        split_name: {
                            label: len(paths)
                            for label, paths in label_map.items()
                        }
                        for split_name, label_map in self.cms_splits.items()
                    },
                    "cms_effective_active_shards": (
                        self.bg_train_dataset.effective_active_shards
                    ),
                    "cms_feature_name_mapping": CMS_TO_JETCLASS_FEATURE_MAP,
                    "cms_feature_sources": (
                        CMS_FEATURE_SOURCES
                    ),
                }
            )

        # Negative augmentation settings for the supported model.
        summary.update(
            {
                # Negative augmentation.
                    "num_negative_views": self.args.num_negative_views,
                    "batch_mix_prob": self.args.batch_mix_prob,
                    "pt_resample_prob": self.args.pt_resample_prob,
                    "node_deta_dphi_rotation_prob":
                        self.args.node_deta_dphi_rotation_prob,
                    "deta_dphi_shuffle_prob": self.args.deta_dphi_shuffle_prob,
                    "identity_shuffle_prob": self.args.identity_shuffle_prob,
                    "corrupt_node_frac": self.args.corrupt_node_frac,
                    "batch_mix_anchor_frac_min": self.args.batch_mix_anchor_frac_min,
                    "batch_mix_anchor_frac_max": self.args.batch_mix_anchor_frac_max,
                "renormalize_negative_pt_sum":
                    self.args.renormalize_negative_pt_sum,
            }
        )
        # Triplet settings for the supported model.
        summary.update(
            {
                "triplet_weight": (
                        self.args.triplet_weight
                    ),
                    "triplet_margin": (
                        self.args.triplet_margin
                    ),
                "use_all_views_as_triplet_positives": (
                    self.args.use_all_views_as_triplet_positives
                ),
            }
        )
        
        summary.update(
            {
                "classification_weight": self.args.classification_weight,
                "num_classification_classes": len(self.background_labels),
            }
        )
            
        

        def update_summary(**updates) -> None:
            if not self.is_main_process:
                return
            summary.update(updates)

            temp_path = summary_path + ".tmp"
            with open(temp_path, "w") as f:
                json.dump(summary, f, indent=2)

            os.replace(temp_path, summary_path)

        # Write a partial summary immediately, before optimization starts.
        update_summary()

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        total_steps = self.args.epochs * self.args.steps_per_epoch
        warmup_steps = self.args.warmup_steps
        if warmup_steps is None:
            warmup_steps = self.args.warmup_epochs * self.args.steps_per_epoch

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

        train_history = {key: [] for key in self.ssl_metric_keys}
        val_history = {key: [] for key in self.ssl_metric_keys}
        auc_history = self._empty_auc_history()
        epoch_end_steps: List[int] = []
        roc_eval_steps: List[int] = []

        best_val_loss = float("inf")
        start_epoch = 1
        global_step = 0
        if self.args.resume_training_state:
            payload = self.checkpoint_payload
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            scheduler.load_state_dict(payload["scheduler_state_dict"])
            start_epoch = int(payload["epoch"]) + 1
            global_step = int(
                payload.get("global_step", int(payload["epoch"]) * self.args.steps_per_epoch)
            )
            best_val_loss = float(payload.get("best_val_loss", float("inf")))
            saved_train_history = payload.get("train_history", {})
            saved_val_history = payload.get("val_history", {})
            for key in self.ssl_metric_keys:
                train_history[key] = list(saved_train_history.get(key, []))
                val_history[key] = list(saved_val_history.get(key, []))
            saved_auc_history = payload.get("auc_history", {})
            for signal_label, per_background in auc_history.items():
                saved_signal = saved_auc_history.get(signal_label, {})
                if not isinstance(saved_signal, dict):
                    continue
                for background_label, split_history in per_background.items():
                    saved_pair = saved_signal.get(background_label, {})
                    if not isinstance(saved_pair, dict):
                        continue
                    split_history["train_heldout"] = list(
                        saved_pair.get(
                            "train_heldout",
                            saved_pair.get("train", []),
                        )
                    )
                    split_history["val"] = list(saved_pair.get("val", []))
            epoch_end_steps = list(payload.get("epoch_end_steps", []))
            roc_eval_steps = list(payload.get("roc_eval_steps", []))
            if start_epoch > self.args.epochs:
                raise ValueError(
                    f"Checkpoint completed epoch {start_epoch - 1}, but --epochs="
                    f"{self.args.epochs}. Set --epochs to a larger total target."
                )
            if self.is_main_process:
                print(
                    f"Resuming optimizer/scheduler at epoch {start_epoch}; "
                    f"global_step={global_step}."
                )

        best_model_path = os.path.join(self.output_dir, "best_model.pth")
        last_model_path = os.path.join(self.output_dir, "last_model.pth")
        best_checkpoint_path = os.path.join(self.output_dir, "best_checkpoint.pt")
        last_checkpoint_path = os.path.join(self.output_dir, "last_checkpoint.pt")

        def save_full_checkpoint(path: str, epoch: int) -> None:
            torch.save(
                {
                    "format_version": 1,
                    "model_state_dict": self.model_core.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "best_val_loss": float(best_val_loss),
                    "train_history": train_history,
                    "val_history": val_history,
                    "auc_history": auc_history,
                    "epoch_end_steps": epoch_end_steps,
                    "roc_eval_steps": roc_eval_steps,
                    "metadata": self._checkpoint_metadata_payload(),
                },
                path,
            )

        update_summary(
            resumed_from_epoch=int(start_epoch - 1),
            current_learning_rate=float(optimizer.param_groups[0]["lr"]),
        )

        profiler_schedule = schedule(
            wait=2,
            warmup=2,
            active=5,
            repeat=1,
        )
        train_iter = iter(train_loader) # infinite stream of training batches
        
        for epoch in range(start_epoch, self.args.epochs + 1):
            if self.is_main_process:
                print(f"\nEpoch [{epoch}/{self.args.epochs}]")

            self.model.train()

            epoch_train = {
                key: []
                for key in self.ssl_metric_keys
            }

            pbar = tqdm(
                range(self.args.steps_per_epoch),
                total=self.args.steps_per_epoch,
                desc=f"Train Epoch {epoch}/{self.args.epochs}",
                disable=not self.is_main_process,
            )

            # Profile only the first epoch.
            should_profile = self.args.profile and epoch == start_epoch

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
                for step in range(self.args.steps_per_epoch):
                    batch = next(train_iter)
                    x_particles = batch["x_particles"].to(
                        DEVICE,
                        non_blocking=True,
                    )
                    y = batch["y"].to(
                        DEVICE,
                        non_blocking=True,
                    )
                    y = self._prepare_labels_for_model(y)
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
                            output = self.model( # train forward
                                x_particles,
                                y,
                                padding_mask=padding_mask,
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
                    global_step += 1

                    with record_function("metrics_to_cpu"):
                        step_losses = self._extract_ssl_metrics(
                            output
                        )

                    for key in train_history:
                        train_history[key].append(step_losses[key])
                        epoch_train[key].append(step_losses[key])

                    with record_function("progress_bar_update"):
                        if (step+1) % 50 == 0 or step == 0:
                            pbar.set_postfix(
                                self._progress_postfix(step_losses),
                                refresh=False,
                            )
                            if step == 0:
                                pbar.update(1)
                            elif step == self.args.steps_per_epoch - 1:
                                pbar.update(49)
                            else:
                                pbar.update(50)
                    # Advance profiler state once per training iteration.
                    if should_profile:
                        prof.step()
                pbar.close()    

            if should_profile and self.is_main_process:
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

            self.model.eval() # val stage
            epoch_val = {
                key: []
                for key in self.ssl_metric_keys
            }

            with torch.no_grad():
                val_iter = iter(bg_val_loader)

                pbar = tqdm(
                    range(self.args.val_steps),
                    total=self.args.val_steps,
                    desc=f"Val Epoch {epoch}/{self.args.epochs}",
                    disable=not self.is_main_process,
                )

                with torch.no_grad():
                    for val_step in range(self.args.val_steps):
                        batch = next(val_iter)
                        x_particles = batch["x_particles"].to(DEVICE, non_blocking=True)
                        y = batch["y"].to(DEVICE, non_blocking=True)
                        y = self._prepare_labels_for_model(y)
                        padding_mask = batch["padding_mask"].to(DEVICE, non_blocking=True)

                        with torch.autocast(
                            device_type=DEVICE.type,
                            dtype=dtype,
                            enabled=use_autocast,
                        ):
                            output = self.model( # validation forward
                                x_particles,
                                y,
                                padding_mask=padding_mask,
                            )

                        step_losses = (
                            self._extract_ssl_metrics(output)
                        )

                        for key in epoch_val:
                            epoch_val[key].append(step_losses[key])
                        
                        if (val_step+1) % 5 == 0 or val_step == 0:
                            pbar.set_postfix(
                                self._progress_postfix(
                                    step_losses
                                ),
                                refresh=False,
                            )
                            if val_step == 0:
                                pbar.update(1)
                            elif val_step == self.args.val_steps - 1:
                                pbar.update(4)
                            else:
                                pbar.update(5)
                    pbar.close()
            
            mean_val = {
                key: float(np.nanmean(values))
                for key, values in epoch_val.items()
            }

            for key in val_history:
                val_history[key].append(mean_val[key])

            epoch_end_steps.append(len(train_history["total_loss"]))

            is_new_best = mean_val["total_loss"] < best_val_loss
            if is_new_best:
                best_val_loss = mean_val["total_loss"]

            if (
                self.signal_labels
                and self.args.roc_eval_every > 0
                and (epoch % self.args.roc_eval_every == 0 or epoch == self.args.epochs)
            ):
                epoch_aucs = self.evaluate_anomaly_score_for_epoch(
                    bg_train_loader=train_eval_loader,
                    bg_val_loader=bg_val_loader,
                    sg_loader=signal_loader,
                    score_name=self.args.anomaly_score,
                    score_fn=(
                        {
                            "mahalanobis": self.compute_mahalanobis_anomaly_scores,
                            "local-global": self.compute_local_global_anomaly_scores,
                        }[self.args.anomaly_score]
                    ),
                )

                if self.is_main_process:
                    for signal_label, per_background in epoch_aucs.items():
                        for background_label, pair_auc in per_background.items():
                            pair_history = auc_history[signal_label][background_label]
                            pair_history["train_heldout"].append(
                                float(pair_auc["train_heldout"])
                            )
                            pair_history["val"].append(float(pair_auc["val"]))
                    roc_eval_steps.append(len(train_history["total_loss"]))

            # Save full checkpoints after epoch-level ROC bookkeeping so a
            # resumed run restores all histories through this epoch.
            if self.is_main_process:
                if is_new_best:
                    torch.save(
                        self.model_core.state_dict(),
                        best_model_path,
                    )
                    save_full_checkpoint(best_checkpoint_path, epoch)
                    print(
                        "Saved new best model state_dict/full checkpoint to "
                        f"{best_model_path} and {best_checkpoint_path}"
                    )

                torch.save(
                    self.model_core.state_dict(),
                    last_model_path,
                )
                save_full_checkpoint(last_checkpoint_path, epoch)
                print(
                    "Saved last model state_dict/full checkpoint to "
                    f"{last_model_path} and {last_checkpoint_path}"
                )

                self.plot_progress(
                    train_history=train_history,
                    val_history=val_history,
                    epoch_end_steps=epoch_end_steps,
                    best_val_loss=best_val_loss,
                    auc_history=auc_history,
                    roc_eval_steps=roc_eval_steps,
                )
                
                print(f"Epoch {epoch} train losses: {mean_train}")
                print(f"Epoch {epoch} val losses: {mean_val}")

            update_summary(
                status="training",
                current_epoch=int(epoch),
                completed_epochs=int(epoch),
                current_learning_rate=float(optimizer.param_groups[0]["lr"]),
                global_step=int(global_step),
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

        print(f"Saved run summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train LeJEPA ParticleTransformer Representation",
        description=(
            "Train the LeJEPA + SIGReg + corrupted-negative triplet + "
            "semi-supervised ParticleTransformer representation model."
        ),
    )
    # SSL model.
    parser.add_argument(
        "--model",
        type=str,
        choices=[ONLY_MODEL_NAME],
        default=ONLY_MODEL_NAME,
        help=(
            "Only semi-sup-triplet"
            "is supported."
        ),
    )
    # Anomaly score function.
    parser.add_argument(
        "--anomaly-score",
        type=str,
        choices=[
            "mahalanobis",
            "local-global",
        ],
        default="mahalanobis",
        help=(
            "Epoch-level anomaly score used for ROC/AUC evaluation. "
            "'mahalanobis' fits a normal latent distribution from background "
            "training representations. "
            "'local-global' generates positive global/local views using the "
            "same MultiViewAugmentation hyperparameters as training and scores "
            "each event by the mean local-to-global-anchor representation MSE. "
            "Default: mahalanobis."
        ),
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
        "--dataset",
        choices=["jetclass", "cms"],
        default="jetclass",
        help=(
            "Dataset backend. JetClass uses official split directories; CMS "
            "uses deterministic random file splits within each jet type."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/HEP/export/home/lwang223/JetClass/JetClass/Pythia",
        help=(
            "Dataset root. For JetClass this contains train_100M, val_5M, and "
            "test_20M. For CMS this contains hbb/qcd/ttbar/wjets/zjets."
        ),
    )
    parser.add_argument(
        "--background-labels",
        type=str,
        default=None,
        help=(
            "Comma-separated canonical labels used as normal/background classes. "
            "Defaults: JetClass=QCD,Hbb,Hcc; CMS=QCD,Hbb."
        ),
    )
    parser.add_argument(
        "--signal-labels",
        type=str,
        default=None,
        help=(
            "Comma-separated canonical labels used only as anomaly signals. "
            "Default: label_Wqq."
        ),
    )
    parser.add_argument(
        "--max-num-particles",
        type=int,
        default=128,
        help="Maximum particles per jet; shorter jets are zero-padded by read_file.",
    )
    parser.add_argument(
        "--max-train-events",
        type=int,
        default=None,
        help=(
            "Optional global event cap per training-stream pass. "
            "With the infinite training stream, the capped subset is "
            "reshuffled and reused across passes."
        )
    )
    parser.add_argument(
        "--max-val-events",
        type=int,
        default=None,
        help="Optional cap on loaded background validation events.",
    )
    parser.add_argument(
        "--max-test-background-events",
        type=int,
        default=None,
        help="Optional cap on loaded background test events.",
    )
    parser.add_argument(
        "--max-test-signal-events",
        type=int,
        default=None,
        help="Optional cap on loaded signal test events.",
    )
    parser.add_argument(
        "--shuffle-active-shards",
        type=int,
        default=3,
        help=(
            "Requested number of preprocessed ROOT shards kept active per "
            "DataLoader worker. Both datasets raise this to at least the number "
            "of requested jet types, guaranteeing one active shard per type. "
            "Higher values improve mixing but use more CPU RAM. Default: 3."
        ),
    )
    parser.add_argument(
        "--feature-plot-events",
        type=int,
        default=10000,
        help="Maximum streamed training events used for feature histograms. Default: 10000.",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=4,
        help="Minimum number of valid nodes per event and per augmented view. Default: 4.",
    )
    parser.add_argument(
        "--cms-pt-min",
        type=float,
        default=None,
        help=(
            "Optional event-level lower cut on the actual CMS jet_pt branch in GeV. "
            "No filename-based pT slicing is performed."
        ),
    )
    parser.add_argument(
        "--cms-pt-max",
        type=float,
        default=None,
        help=(
            "Optional exclusive event-level upper cut on the actual CMS jet_pt "
            "branch in GeV."
        ),
    )
    parser.add_argument(
        "--cms-val-fraction",
        type=float,
        default=0.1,
        help="CMS ROOT-shard validation fraction within every jet type.",
    )
    parser.add_argument(
        "--cms-test-fraction",
        type=float,
        default=0.1,
        help="CMS ROOT-shard test fraction within every jet type.",
    )
    parser.add_argument(
        "--cms-split-seed",
        type=int,
        default=42,
        help="Seed for deterministic CMS per-type train/val/test file splits.",
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
        default=0.05,
        help="Weight for SIGReg loss, matching LeJEPA lambda by default. Default: 0.05.",
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
        "--classification-weight",
        type=float,
        default=None,
        help=(
            "Weight for classification loss. Defaults to 0.1 for multi-class "
            "training. Single-label training requires explicitly passing 0."
        ),
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
        "--node-deta-dphi-rotation-prob",
        type=float,
        default=0.20,
        help="Probability of sampling independent node-level deta-dphi rotation. Default: 0.20.",
    )
    parser.add_argument(
        "--deta-dphi-shuffle-prob",
        type=float,
        default=0.05,
        help="Probability of sampling deta_dphi_shuffle for a negative view. Default: 0.05.",
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
        help="Renormalize corrupted negative sample's pt sums to the original event pt sum. Default: True.",
    )
    
    # Optimization.
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Training batch size. Defaults to 128.",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=4000,
        help=(
            "Number of streamed training batches per epoch. Required for lazy "
            "IterableDataset training because the full dataset length is not materialized. "
            "Default: 4000."
        ),
    )
    parser.add_argument(
        "--val-steps",
        type=int,
        default=500,
        help="Maximum streamed validation batches per epoch. Default: 500.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help=(
            "Maximum streamed batches collected per dataset for latent-space "
            "and anomaly-score evaluation. Default: 500."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of training epochs. Defaults to 80.",
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
        choices=["bf16", "fp32"],
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

    # Checkpoint initialization / resume.
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Optional raw model state_dict or full checkpoint. Cross-dataset "
            "loads automatically discard the old classification head."
        ),
    )
    parser.add_argument(
        "--checkpoint-summary",
        type=str,
        default=None,
        help=(
            "Optional summary.json associated with --checkpoint. By default the "
            "script looks beside the checkpoint."
        ),
    )
    parser.add_argument(
        "--reset-classification-head",
        action="store_true",
        help=(
            "Force a fresh semi-supervised classification head even on the same "
            "dataset. Dataset or class-order changes reset it automatically."
        ),
    )
    parser.add_argument(
        "--resume-training-state",
        action="store_true",
        help=(
            "Resume optimizer, scheduler, epoch, and histories from a full "
            "*_checkpoint.pt file. Requires the same dataset and class order."
        ),
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
        "--prefetch-factor",
        type=int,
        default=2,
        help=(
            "Number of batches prefetched by each DataLoader worker. Used only "
            "when --num-workers > 0. Default: 2."
        ),
    )
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=torch.cuda.is_available(),
        help="Use pinned host memory in DataLoader. Default: True on CUDA, else False.",
    )
    parser.add_argument(
        "--roc-eval-every",
        type=int,
        default=1,
        help="Compute regrouped pairwise ROC AUC every N epochs. Use 0 to disable. Default: 1.",
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

    trainer = TrainLeJEPAParticleTransformer()
    trainer.load()
    trainer.build_node_datasets()
    trainer.plot_features()
    trainer.train()
    if trainer.distributed:
        dist.destroy_process_group()