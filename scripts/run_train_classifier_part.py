"""
Train a supervised ParticleTransformer classifier as an approximate AUC ceiling.

This script mirrors the PART + LeJEPA data pipeline (padded node tensors, no
graph edges) but trains with background/signal labels using BCE loss. The
held-out ROC AUC is a useful reference when comparing against unsupervised
LeJEPA Mahalanobis scores on the same dataset and architecture.

Example command:

python -u scripts/run_train_classifier_part.py \
    --background "data/processed/PT-200to400/scaledby_QCD/QCD_scaled.pkl" \
    --signal "data/processed/PT-200to400/scaledby_QCD/WJet_scaled.pkl" \
    --embed-dim 128 \
    --num-layers 8 \
    --num-heads 8 \
    --batch-size 128 \
    --epochs 50 \
    --learning-rate 5e-4 \
    --weight-decay 5e-2 \
    --precision bf16 \
    --output-dir "plots/run-classifier-part-upper-bound"
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from helpers import helpers_main
from models.part import ParticleTransformerClassifier, ParticleTransformerConfig

config = helpers_main.load_config()
bg_file = os.path.join(config["data"]["processed_data_dir"], config["data"]["background_file"])
sg_file = os.path.join(config["data"]["processed_data_dir"], config["data"]["signal_file"])
DEVICE = torch.device(helpers_main.get_device())

DEFAULT_NODE_FEATURES = (
    "eta,phi,pt,d0/d0Err,dz/dzErr,charge,mass,log_pt,"
    "pdgId_-211,pdgId_-13,pdgId_-11,pdgId_11,"
    "pdgId_13,pdgId_22,pdgId_130,pdgId_211"
)


def parse_node_features(feature_string: str) -> List[str]:
    features = [item.strip() for item in feature_string.split(",") if item.strip()]
    if not features:
        raise ValueError("At least one node feature must be provided.")
    return features


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


def row_to_node_tensor(
    row: pd.Series,
    node_feature_names: Sequence[str],
    min_nodes: int,
) -> torch.Tensor:
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


def split_one_class(
    nodes: Sequence[torch.Tensor],
    labels: Sequence[int],
    val_fraction: float,
    test_fraction: float,
) -> Tuple[
    List[torch.Tensor],
    List[int],
    List[torch.Tensor],
    List[int],
    List[torch.Tensor],
    List[int],
]:
    if len(nodes) < 3:
        raise ValueError(
            "Need at least 3 events per class to create train, validation, and test splits."
        )

    n_total = len(nodes)
    n_test = max(1, int(round(n_total * test_fraction)))
    n_val = max(1, int(round(n_total * val_fraction)))

    if n_test + n_val >= n_total:
        n_test = 1
        n_val = 1

    n_train = n_total - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Invalid split for {n_total} events: train={n_train}, "
            f"val={n_val}, test={n_test}."
        )

    train_nodes = list(nodes[:n_train])
    train_labels = list(labels[:n_train])
    val_nodes = list(nodes[n_train : n_train + n_val])
    val_labels = list(labels[n_train : n_train + n_val])
    test_nodes = list(nodes[n_train + n_val :])
    test_labels = list(labels[n_train + n_val :])
    return train_nodes, train_labels, val_nodes, val_labels, test_nodes, test_labels


def shuffled_copy(
    nodes: Sequence[torch.Tensor],
    labels: Sequence[int],
    rng: random.Random,
) -> Tuple[List[torch.Tensor], List[int]]:
    indices = list(range(len(nodes)))
    rng.shuffle(indices)
    return [nodes[i] for i in indices], [labels[i] for i in indices]


def combine_and_shuffle(
    bg_nodes: Sequence[torch.Tensor],
    bg_labels: Sequence[int],
    sg_nodes: Sequence[torch.Tensor],
    sg_labels: Sequence[int],
    rng: random.Random,
    balance: bool,
) -> Tuple[List[torch.Tensor], List[int]]:
    bg_nodes = list(bg_nodes)
    bg_labels = list(bg_labels)
    sg_nodes = list(sg_nodes)
    sg_labels = list(sg_labels)

    if balance:
        n_per_class = min(len(bg_nodes), len(sg_nodes))
        bg_nodes = bg_nodes[:n_per_class]
        bg_labels = bg_labels[:n_per_class]
        sg_nodes = sg_nodes[:n_per_class]
        sg_labels = sg_labels[:n_per_class]

    combined_nodes = bg_nodes + sg_nodes
    combined_labels = bg_labels + sg_labels
    indices = list(range(len(combined_nodes)))
    rng.shuffle(indices)
    return [combined_nodes[i] for i in indices], [combined_labels[i] for i in indices]


def plot_training_curves(metrics: pd.DataFrame, save_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(metrics["epoch"], metrics["train_loss"], label="Train")
    axes[0].plot(metrics["epoch"], metrics["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Classifier Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(metrics["epoch"], metrics["train_auc"], label="Train")
    axes[1].plot(metrics["epoch"], metrics["val_auc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("ROC AUC")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Classifier AUC")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_score_distribution(
    background_scores: Sequence[float],
    signal_scores: Sequence[float],
    save_path: str,
) -> None:
    plt.figure(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 101)
    plt.hist(
        background_scores,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Background",
        color="tab:blue",
    )
    plt.hist(
        signal_scores,
        bins=bins,
        density=True,
        alpha=0.55,
        label="Signal",
        color="tab:red",
    )
    plt.xlabel("Classifier signal probability")
    plt.ylabel("Density")
    plt.title("Supervised PART Classifier Score Distribution")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc(y_true: Sequence[int], y_score: Sequence[float], save_path: str) -> float:
    auc_score = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Supervised PART Classifier ROC")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return float(auc_score)


class TrainPartClassifierUpperBound:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.bg_file = args.background
        self.sg_file = args.signal
        self.bg_name = helpers_main.trim_name(self.bg_file)
        self.sg_name = helpers_main.trim_name(self.sg_file)
        self.node_feature_names = parse_node_features(args.node_features)
        self.output_dir = args.output_dir
        self.feature_plots_dir = os.path.join(self.output_dir, "features")
        self.rng = random.Random(args.seed)

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

        self.session_name = os.path.join(
            self.output_dir,
            f"train_classifier_part_{self.bg_name}_{self.sg_name}_{helpers_main.curr_time()}.log",
        )
        helpers_main.log_config(self.session_name)

    def load(self) -> None:
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
        logging.info(f"Node features: {self.node_feature_names}")

        print(f"Background rows: {len(self.bg_data)}")
        print(f"Signal rows: {len(self.sg_data)}")
        print(f"Node features: {self.node_feature_names}")

    def build_node_datasets(self) -> None:
        print("Loading background node tensors...")
        bg_nodes, bg_labels = dataframe_to_node_tensors(
            df=self.bg_data,
            node_feature_names=self.node_feature_names,
            label=0,
            min_nodes=self.args.min_nodes,
        )

        print("Loading signal node tensors...")
        sg_nodes, sg_labels = dataframe_to_node_tensors(
            df=self.sg_data,
            node_feature_names=self.node_feature_names,
            label=1,
            min_nodes=self.args.min_nodes,
        )

        if len(bg_nodes) < 3:
            raise ValueError(f"Only {len(bg_nodes)} background events were loaded.")
        if len(sg_nodes) < 3:
            raise ValueError(f"Only {len(sg_nodes)} signal events were loaded.")

        shuffled_bg_nodes, shuffled_bg_labels = shuffled_copy(bg_nodes, bg_labels, self.rng)
        shuffled_sg_nodes, shuffled_sg_labels = shuffled_copy(sg_nodes, sg_labels, self.rng)

        (
            self.bg_train_nodes,
            self.bg_train_labels,
            self.bg_val_nodes,
            self.bg_val_labels,
            self.bg_test_nodes,
            self.bg_test_labels,
        ) = split_one_class(
            shuffled_bg_nodes,
            shuffled_bg_labels,
            self.args.val_fraction,
            self.args.test_fraction,
        )
        (
            self.sg_train_nodes,
            self.sg_train_labels,
            self.sg_val_nodes,
            self.sg_val_labels,
            self.sg_test_nodes,
            self.sg_test_labels,
        ) = split_one_class(
            shuffled_sg_nodes,
            shuffled_sg_labels,
            self.args.val_fraction,
            self.args.test_fraction,
        )

        self.train_nodes, self.train_labels = combine_and_shuffle(
            self.bg_train_nodes,
            self.bg_train_labels,
            self.sg_train_nodes,
            self.sg_train_labels,
            self.rng,
            self.args.balance_train,
        )
        self.val_nodes, self.val_labels = combine_and_shuffle(
            self.bg_val_nodes,
            self.bg_val_labels,
            self.sg_val_nodes,
            self.sg_val_labels,
            self.rng,
            balance=False,
        )
        self.test_nodes, self.test_labels = combine_and_shuffle(
            self.bg_test_nodes,
            self.bg_test_labels,
            self.sg_test_nodes,
            self.sg_test_labels,
            self.rng,
            balance=False,
        )

        self.train_dataset = JetNodeDataset(self.train_nodes, self.train_labels)
        self.val_dataset = JetNodeDataset(self.val_nodes, self.val_labels)
        self.test_dataset = JetNodeDataset(self.test_nodes, self.test_labels)

        print(f"Background events: {len(bg_nodes)}")
        print(f"Signal events: {len(sg_nodes)}")
        print(
            "Train split:",
            f"bg={len(self.bg_train_nodes)}",
            f"signal={len(self.sg_train_nodes)}",
            f"used={len(self.train_nodes)}",
        )
        print(
            "Validation split:",
            f"bg={len(self.bg_val_nodes)}",
            f"signal={len(self.sg_val_nodes)}",
        )
        print(
            "Test split:",
            f"bg={len(self.bg_test_nodes)}",
            f"signal={len(self.sg_test_nodes)}",
        )
        print(f"Example node tensor shape: {self.train_nodes[0].shape}")

    def plot_features(self) -> None:
        os.makedirs(self.feature_plots_dir, exist_ok=True)
        all_features = torch.cat(self.train_nodes, dim=0).numpy()

        for i, name in enumerate(self.node_feature_names):
            plt.figure(figsize=(7, 5))
            plt.hist(
                all_features[:, i],
                bins=50,
                density=True,
                color="tab:blue",
                edgecolor="black",
                alpha=0.75,
            )
            plt.title(f"Feature {i + 1}: {name}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.grid(alpha=0.25)
            plt.tight_layout()

            safe_name = name.replace("/", "_")
            plt.savefig(
                os.path.join(self.feature_plots_dir, f"feature_{i + 1}_{safe_name}.png")
            )
            plt.close()

    def build_model(self) -> ParticleTransformerClassifier:
        model_config = ParticleTransformerConfig(
            input_dim=len(self.node_feature_names),
            embed_dim=self.args.embed_dim,
            num_heads=self.args.num_heads,
            num_layers=self.args.num_layers,
            num_class_layers=self.args.num_class_layers,
            ffn_mult=self.args.ffn_mult,
            dropout=self.args.dropout,
            class_dropout=self.args.class_dropout,
            representation_dim=self.args.representation_dim,
            use_pairwise_bias=not self.args.no_pairwise_bias,
            pairwise_hidden_dim=self.args.pairwise_hidden_dim,
            pairwise_num_features=self.args.pairwise_num_features,
            compute_dtype=precision_to_dtype(self.args.precision),
            use_internal_autocast=False,
            eps=self.args.eps,
        )
        return ParticleTransformerClassifier(model_config).to(DEVICE)

    def evaluate(
        self,
        model: torch.nn.Module,
        dataset: JetNodeDataset,
        criterion: torch.nn.Module,
        description: str,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=collate_node_tensors,
        )
        model.eval()
        total_loss = 0.0
        all_scores: List[float] = []
        all_labels: List[int] = []

        dtype = precision_to_dtype(self.args.precision)
        use_autocast = autocast_enabled_for_precision(self.args.precision)

        with torch.no_grad():
            for batch in tqdm(loader, desc=description):
                x = batch["x"].to(DEVICE, non_blocking=True)
                padding_mask = batch["padding_mask"].to(DEVICE, non_blocking=True)
                labels = batch["y"].float().to(DEVICE)

                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=dtype,
                    enabled=use_autocast,
                ):
                    logits = model(x, padding_mask=padding_mask).view(-1)
                    loss = criterion(logits, labels)
                    scores = torch.sigmoid(logits)

                total_loss += loss.item() * labels.size(0)
                all_scores.extend(scores.detach().float().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())

        labels_np = np.asarray(all_labels, dtype=np.int64)
        scores_np = np.asarray(all_scores, dtype=np.float64)
        preds_np = (scores_np >= 0.5).astype(np.int64)

        mean_loss = total_loss / max(len(dataset), 1)
        accuracy = accuracy_score(labels_np, preds_np)
        auc_score = roc_auc_score(labels_np, scores_np)
        return mean_loss, accuracy, auc_score, labels_np, scores_np

    def train(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = self.build_model()
        num_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        logging.info(f"Model summary:\n{self.model}")
        logging.info(f"Number of trainable parameters: {num_params}")
        print(f"Model summary:\n{self.model}")
        print(f"Number of trainable parameters: {num_params}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        criterion = torch.nn.BCEWithLogitsLoss()

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=collate_node_tensors,
        )

        dtype = precision_to_dtype(self.args.precision)
        use_autocast = autocast_enabled_for_precision(self.args.precision)

        metrics = []
        best_val_auc = -float("inf")
        best_val_loss = float("inf")
        best_state_dict = None
        best_epoch = 0
        best_model_path = os.path.join(self.output_dir, "best_model.pth")
        timer = helpers_main.LeTimer()

        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            train_scores: List[float] = []
            train_labels: List[int] = []

            pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{self.args.epochs}")
            for batch in pbar:
                x = batch["x"].to(DEVICE, non_blocking=True)
                padding_mask = batch["padding_mask"].to(DEVICE, non_blocking=True)
                labels = batch["y"].float().to(DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(
                    device_type=DEVICE.type,
                    dtype=dtype,
                    enabled=use_autocast,
                ):
                    logits = self.model(x, padding_mask=padding_mask).view(-1)
                    loss = criterion(logits, labels)
                    scores = torch.sigmoid(logits)

                loss.backward()

                if self.args.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.args.grad_clip_norm,
                    )

                optimizer.step()

                epoch_loss += loss.item() * labels.size(0)
                train_scores.extend(scores.detach().float().cpu().numpy().tolist())
                train_labels.extend(labels.detach().cpu().numpy().tolist())
                pbar.set_postfix({"loss": f"{loss.item():.5g}"})

            train_loss = epoch_loss / len(self.train_dataset)
            train_labels_np = np.asarray(train_labels, dtype=np.int64)
            train_scores_np = np.asarray(train_scores, dtype=np.float64)
            train_acc = accuracy_score(
                train_labels_np,
                (train_scores_np >= 0.5).astype(np.int64),
            )
            train_auc = roc_auc_score(train_labels_np, train_scores_np)

            val_loss, val_acc, val_auc, _, _ = self.evaluate(
                self.model,
                self.val_dataset,
                criterion,
                description=f"Val Epoch {epoch}/{self.args.epochs}",
            )

            metrics.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "train_auc": train_auc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_auc": val_auc,
                }
            )

            if val_auc > best_val_auc or (
                np.isclose(val_auc, best_val_auc) and val_loss < best_val_loss
            ):
                best_val_auc = val_auc
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
                torch.save(best_state_dict, best_model_path)
                logging.info(f"Saved new best model to {best_model_path}")

            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(
                os.path.join(self.output_dir, "classifier_metrics.csv"),
                index=False,
            )
            plot_training_curves(
                metrics_df,
                os.path.join(self.output_dir, "training_curves.png"),
            )

            logging.info(
                "Epoch %s/%s | train loss %.6f | train AUC %.6f | "
                "val loss %.6f | val AUC %.6f%s",
                epoch,
                self.args.epochs,
                train_loss,
                train_auc,
                val_loss,
                val_auc,
                timer.time_taken(),
            )

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        test_loss, test_acc, test_auc, test_labels, test_scores = self.evaluate(
            self.model,
            self.test_dataset,
            criterion,
            description="Final Test Evaluation",
        )

        background_scores = test_scores[test_labels == 0]
        signal_scores = test_scores[test_labels == 1]

        np.save(
            os.path.join(self.output_dir, "background_test_scores.npy"),
            background_scores,
        )
        np.save(
            os.path.join(self.output_dir, "signal_test_scores.npy"),
            signal_scores,
        )
        np.save(os.path.join(self.output_dir, "test_labels.npy"), test_labels)
        np.save(os.path.join(self.output_dir, "test_scores.npy"), test_scores)

        plot_score_distribution(
            background_scores,
            signal_scores,
            os.path.join(self.output_dir, "classifier_score.png"),
        )
        plot_roc(
            test_labels,
            test_scores,
            os.path.join(self.output_dir, "roc.png"),
        )

        summary = {
            "model_type": "ParticleTransformerClassifier",
            "background": self.bg_file,
            "signal": self.sg_file,
            "node_features": self.node_feature_names,
            "min_nodes": self.args.min_nodes,
            "embed_dim": self.args.embed_dim,
            "representation_dim": self.args.representation_dim,
            "num_layers": self.args.num_layers,
            "num_heads": self.args.num_heads,
            "num_class_layers": self.args.num_class_layers,
            "ffn_mult": self.args.ffn_mult,
            "dropout": self.args.dropout,
            "class_dropout": self.args.class_dropout,
            "use_pairwise_bias": not self.args.no_pairwise_bias,
            "pairwise_hidden_dim": self.args.pairwise_hidden_dim,
            "pairwise_num_features": self.args.pairwise_num_features,
            "precision": self.args.precision,
            "batch_size": self.args.batch_size,
            "epochs": self.args.epochs,
            "learning_rate": self.args.learning_rate,
            "weight_decay": self.args.weight_decay,
            "seed": self.args.seed,
            "device": str(DEVICE),
            "max_background_events": self.args.max_background_events,
            "max_signal_events": self.args.max_signal_events,
            "val_fraction": self.args.val_fraction,
            "test_fraction": self.args.test_fraction,
            "balance_train": self.args.balance_train,
            "auc": float(test_auc),
            "accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_val_auc),
            "best_val_loss": float(best_val_loss),
            "background_events": len(self.bg_train_nodes)
            + len(self.bg_val_nodes)
            + len(self.bg_test_nodes),
            "signal_events": len(self.sg_train_nodes)
            + len(self.sg_val_nodes)
            + len(self.sg_test_nodes),
            "background_train_events": len(self.bg_train_nodes),
            "signal_train_events": len(self.sg_train_nodes),
            "training_events_used": len(self.train_nodes),
            "background_val_events": len(self.bg_val_nodes),
            "signal_val_events": len(self.sg_val_nodes),
            "background_test_events": len(self.bg_test_nodes),
            "signal_test_events": len(self.sg_test_nodes),
            "num_trainable_parameters": int(num_params),
        }

        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logging.info(f"Saved run summary to {summary_path}")
        logging.info(f"Final supervised PART classifier AUC: {test_auc}")
        print(f"Final supervised PART classifier AUC: {test_auc:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Train PART Classifier Upper Bound",
        description=(
            "Train a supervised ParticleTransformer classifier on background "
            "and signal labels, then report held-out ROC AUC as an approximate "
            "ceiling for the PART + LeJEPA pipeline."
        ),
    )

    parser.add_argument(
        "--background",
        "-b",
        type=str,
        default=bg_file,
        help="Path to processed .pkl background dataset.",
    )
    parser.add_argument(
        "--signal",
        "-s",
        type=str,
        default=sg_file,
        help="Path to processed .pkl signal dataset.",
    )
    parser.add_argument(
        "--node-features",
        type=str,
        default=DEFAULT_NODE_FEATURES,
        help="Comma-separated node feature list matching the LeJEPA PART runs.",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=4,
        help="Minimum number of valid nodes per event. Default: 4.",
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
        "--embed-dim",
        type=int,
        default=128,
        help="Transformer embedding dimension. Default: 128.",
    )
    parser.add_argument(
        "--representation-dim",
        type=int,
        default=128,
        help="Unused representation head dimension inherited from backbone config. Default: 128.",
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
        "--num-class-layers",
        type=int,
        default=2,
        help="Number of CLS pooling layers. Default: 2.",
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
        help="Dropout in embedding, attention, and FFN. Default: 0.1.",
    )
    parser.add_argument(
        "--class-dropout",
        type=float,
        default=0.0,
        help="Dropout in CLS pooling layers. Default: 0.0.",
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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config["model"]["batch_size"],
        help="Training batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config["training"].get("epochs", 50),
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="AdamW learning rate. Default: 5e-4.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=5e-2,
        help="AdamW weight decay. Default: 5e-2.",
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
        help="Mixed precision mode. Default: bf16.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=None,
        help="Optional gradient clipping max norm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/run-classifier-part-upper-bound",
        help="Directory for plots, checkpoints, metrics, and summary.json.",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.1,
        help="Fraction of each class held out for validation.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.1,
        help="Fraction of each class held out for final test AUC.",
    )
    parser.add_argument(
        "--balance-train",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Downsample the larger class in the training split.",
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
        help="Use pinned host memory in DataLoader.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.val_fraction <= 0 or args.test_fraction <= 0:
        raise ValueError("--val-fraction and --test-fraction must be positive.")
    if args.val_fraction + args.test_fraction >= 1:
        raise ValueError(
            "--val-fraction + --test-fraction must be less than 1 so training "
            "events remain."
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    trainer = TrainPartClassifierUpperBound(args)
    trainer.load()
    trainer.build_node_datasets()
    trainer.plot_features()
    trainer.train()


if __name__ == "__main__":
    main()
