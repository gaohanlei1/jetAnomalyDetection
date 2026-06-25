"""
Train a supervised graph classifier as an approximate AUC ceiling.

The autoencoder scripts report anomaly-detection AUC from reconstruction loss.
This script answers a different question: if we give the model true
background/signal labels during training, how separable is this dataset with the
available graph features?

Example command:
python -u scripts/run_train_classifier.py \
    --background "data/processed/PT-200to400/scaledby_QCD/QCD_scaled.pkl" \
    --signal "data/processed/PT-200to400/scaledby_QCD/WJet_scaled.pkl" \
    --knn 16 \
    --smallest-dim 16 \
    --num-reduced-edges 16 \
    --batch-size 64 \
    --epochs 20 \
    --learning-rate 1e-4 \
    --weight-decay 1e-4 \
    --seed 42 \
    --output-dir "plots/run-classifier-upper-bound"
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
from typing import List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add parent directory to import local project modules.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import constants as c
from helpers import helpers_main
from models.classifier import JetGraphAutoencoderClassification
from preprocess.make_graphs import graph_data_loader


config = helpers_main.load_config()
bg_file = os.path.join(
    config["data"]["processed_data_dir"],
    config["data"]["background_file"],
)
sg_file = os.path.join(
    config["data"]["processed_data_dir"],
    config["data"]["signal_file"],
)
DEVICE = helpers_main.get_device()

BASE_NODE_FEATURES = [
    "pt",
    "eta",
    "phi",
    "d0/d0Err",
    "dz/dzErr",
    "charge",
    "mass",
    "log_pt",
]
PDG_NODE_FEATURES = [
    "pdgId_-211",
    "pdgId_-13",
    "pdgId_-11",
    "pdgId_11",
    "pdgId_13",
    "pdgId_22",
    "pdgId_130",
    "pdgId_211",
]
DEFAULT_NODE_FEATURES = BASE_NODE_FEATURES + PDG_NODE_FEATURES


def parse_node_features(value: str) -> List[str]:
    features = [feature.strip() for feature in value.split(",")]
    features = [feature for feature in features if feature]
    if not features:
        raise argparse.ArgumentTypeError("At least one node feature is required.")
    return features


def remove_low_pt_muons(row: pd.Series) -> pd.Series:
    """Remove low-pT muons from WminusH signal rows, matching the AE scripts."""
    if "pdgId" not in row or "pt" not in row:
        return row

    pdg_id = row["pdgId"]
    pt = row["pt"]
    mask = (np.abs(pdg_id) != 13) | (pt >= 0.4)

    for col in row.index:
        val = row[col]
        if hasattr(val, "__getitem__") and not isinstance(val, str):
            try:
                if len(val) == len(mask):
                    row[col] = val[mask]
            except TypeError:
                pass
    return row


def is_array_like_particle_feature(value, expected_len: int) -> bool:
    if isinstance(value, str):
        return False
    if not hasattr(value, "__len__"):
        return False
    try:
        return len(value) == expected_len
    except TypeError:
        return False


def validate_node_features(
    df: pd.DataFrame,
    feature_names: Sequence[str],
    dataset_label: str,
    max_rows: int = 20,
) -> None:
    """Fail early when a requested feature is missing or is jet-level scalar data."""
    missing = [feature for feature in feature_names if feature not in df.columns]
    if missing:
        raise ValueError(
            f"{dataset_label} is missing requested node features: {missing}"
        )

    if "pt" not in df.columns:
        raise ValueError(f"{dataset_label} is missing required particle feature 'pt'.")

    scalar_features = set()
    rows_to_check = min(max_rows, len(df))
    for _, row in df.head(rows_to_check).iterrows():
        expected_len = len(row["pt"])
        for feature in feature_names:
            if not is_array_like_particle_feature(row[feature], expected_len):
                scalar_features.add(feature)

    if scalar_features:
        raise ValueError(
            f"{dataset_label} has requested features that are not per-particle "
            f"arrays with the same length as 'pt': {sorted(scalar_features)}. "
            "For this graph classifier, use node features such as "
            "'pt,eta,phi,d0/d0Err,dz/dzErr,charge,mass,log_pt' or the pdgId "
            "one-hot array columns. Jet-level columns like fj_particleNet_* are "
            "event features and cannot be used as node features without a "
            "different model."
        )


def split_one_class(
    graphs: Sequence[Data],
    val_fraction: float,
    test_fraction: float,
) -> Tuple[List[Data], List[Data], List[Data]]:
    if len(graphs) < 3:
        raise ValueError(
            "Need at least 3 graphs per class to create train, validation, and "
            "test splits."
        )

    n_total = len(graphs)
    n_test = max(1, int(round(n_total * test_fraction)))
    n_val = max(1, int(round(n_total * val_fraction)))

    if n_test + n_val >= n_total:
        n_test = 1
        n_val = 1

    n_train = n_total - n_val - n_test
    if n_train < 1:
        raise ValueError(
            f"Invalid split for {n_total} graphs: train={n_train}, "
            f"val={n_val}, test={n_test}."
        )

    train_graphs = list(graphs[:n_train])
    val_graphs = list(graphs[n_train : n_train + n_val])
    test_graphs = list(graphs[n_train + n_val :])
    return train_graphs, val_graphs, test_graphs


def shuffled_copy(graphs: Sequence[Data], rng: random.Random) -> List[Data]:
    graphs_copy = list(graphs)
    rng.shuffle(graphs_copy)
    return graphs_copy


def combine_and_shuffle(
    bg_graphs: Sequence[Data],
    sg_graphs: Sequence[Data],
    rng: random.Random,
    balance: bool,
) -> List[Data]:
    bg_graphs = list(bg_graphs)
    sg_graphs = list(sg_graphs)

    if balance:
        n_per_class = min(len(bg_graphs), len(sg_graphs))
        bg_graphs = bg_graphs[:n_per_class]
        sg_graphs = sg_graphs[:n_per_class]

    combined = bg_graphs + sg_graphs
    rng.shuffle(combined)
    return combined


def plot_training_curves(
    metrics: pd.DataFrame,
    save_path: str,
) -> None:
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
    plt.title("Supervised Classifier Score Distribution")
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
    plt.title("Supervised Classifier ROC")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return float(auc_score)


class TrainClassifierUpperBound:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.bg_file = args.background
        self.sg_file = args.signal
        self.bg_name = helpers_main.trim_name(self.bg_file)
        self.sg_name = helpers_main.trim_name(self.sg_file)
        self.method = args.method
        self.knn = args.knn
        self.smallest_dim = args.smallest_dim
        self.num_reduced_edges = args.num_reduced_edges
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.initial_lr = args.learning_rate
        self.weight_decay = args.weight_decay
        self.seed = args.seed
        self.max_background_events = args.max_background_events
        self.max_signal_events = args.max_signal_events
        self.val_fraction = args.val_fraction
        self.test_fraction = args.test_fraction
        self.balance_train = args.balance_train
        self.node_features = args.node_features
        self.output_dir = args.output_dir
        self.feature_plots_dir = os.path.join(self.output_dir, "features")
        self.rng = random.Random(self.seed)

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.session_name = (
            f"logs/train_classifier_{self.bg_name}_{self.sg_name}_"
            f"{self.method}_{helpers_main.curr_time()}.log"
        )
        helpers_main.log_config(self.session_name)

    def load(self) -> None:
        self.bg_data = pd.read_pickle(self.bg_file)
        self.sg_data = pd.read_pickle(self.sg_file)

        if self.max_background_events is not None:
            self.bg_data = self.bg_data.head(self.max_background_events)
        if self.max_signal_events is not None:
            self.sg_data = self.sg_data.head(self.max_signal_events)

        rawfj_pt_col = c.RAW_FATJET_PROPERTIES_PREFIX + "pt"
        if rawfj_pt_col in self.bg_data:
            self.bg_data = self.bg_data[
                (self.bg_data[rawfj_pt_col] > c.PT_MIN)
                & (self.bg_data[rawfj_pt_col] < c.PT_MAX)
            ]
        if rawfj_pt_col in self.sg_data:
            self.sg_data = self.sg_data[
                (self.sg_data[rawfj_pt_col] > c.PT_MIN)
                & (self.sg_data[rawfj_pt_col] < c.PT_MAX)
            ]

        if "WminusH" in self.sg_file:
            self.sg_data = self.sg_data.apply(remove_low_pt_muons, axis=1)

        if len(self.bg_data) == 0:
            raise ValueError("No background events remain after loading and pT slicing.")
        if len(self.sg_data) == 0:
            raise ValueError("No signal events remain after loading and pT slicing.")

        logging.info(f"Background events after slicing: {len(self.bg_data)}")
        logging.info(f"Signal events after slicing: {len(self.sg_data)}")
        logging.info(f"Node features: {self.node_features}")

        if rawfj_pt_col in self.bg_data:
            print("Background fj_pt stats after slicing:")
            print(self.bg_data[rawfj_pt_col].describe())
        if rawfj_pt_col in self.sg_data:
            print("Signal fj_pt stats after slicing:")
            print(self.sg_data[rawfj_pt_col].describe())

    def build_graphs(self) -> None:
        validate_node_features(self.bg_data, self.node_features, "background")
        validate_node_features(self.sg_data, self.node_features, "signal")

        print("Building background graphs...")
        self.bg_graphs = graph_data_loader(
            self.bg_data,
            data_label=0,
            nearest_neighbors=self.knn,
            device="cpu",
            method=self.method,
            alpha=config["training"]["alpha"],
            node_feature_names=self.node_features,
        )

        print("Building signal graphs...")
        self.sg_graphs = graph_data_loader(
            self.sg_data,
            data_label=1,
            nearest_neighbors=self.knn,
            device="cpu",
            method=self.method,
            alpha=config["training"]["alpha"],
            node_feature_names=self.node_features,
        )

        if len(self.bg_graphs) < 3:
            raise ValueError(f"Only {len(self.bg_graphs)} background graphs were built.")
        if len(self.sg_graphs) < 3:
            raise ValueError(f"Only {len(self.sg_graphs)} signal graphs were built.")

        shuffled_bg = shuffled_copy(self.bg_graphs, self.rng)
        shuffled_sg = shuffled_copy(self.sg_graphs, self.rng)

        self.bg_train_graphs, self.bg_val_graphs, self.bg_test_graphs = split_one_class(
            shuffled_bg,
            self.val_fraction,
            self.test_fraction,
        )
        self.sg_train_graphs, self.sg_val_graphs, self.sg_test_graphs = split_one_class(
            shuffled_sg,
            self.val_fraction,
            self.test_fraction,
        )

        self.train_graphs = combine_and_shuffle(
            self.bg_train_graphs,
            self.sg_train_graphs,
            self.rng,
            self.balance_train,
        )
        self.val_graphs = combine_and_shuffle(
            self.bg_val_graphs,
            self.sg_val_graphs,
            self.rng,
            balance=False,
        )
        self.test_graphs = combine_and_shuffle(
            self.bg_test_graphs,
            self.sg_test_graphs,
            self.rng,
            balance=False,
        )

        print(f"Background graphs: {len(self.bg_graphs)}")
        print(f"Signal graphs: {len(self.sg_graphs)}")
        print(
            "Train split:",
            f"bg={len(self.bg_train_graphs)}",
            f"signal={len(self.sg_train_graphs)}",
            f"used={len(self.train_graphs)}",
        )
        print(
            "Validation split:",
            f"bg={len(self.bg_val_graphs)}",
            f"signal={len(self.sg_val_graphs)}",
        )
        print(
            "Test split:",
            f"bg={len(self.bg_test_graphs)}",
            f"signal={len(self.sg_test_graphs)}",
        )

    def compute_stats(self) -> None:
        train_features = torch.cat([graph.x for graph in self.train_graphs], dim=0)
        self.num_features = train_features.shape[1]
        self.means = train_features.mean(dim=0)
        self.stds = train_features.std(dim=0)
        logging.info(f"Feature means: {self.means}")
        logging.info(f"Feature stds: {self.stds}")
        logging.info(f"Number of node features: {self.num_features}")

    def plot_features(self) -> None:
        os.makedirs(self.feature_plots_dir, exist_ok=True)
        train_features = torch.cat([graph.x for graph in self.train_graphs], dim=0)

        for i, name in enumerate(self.node_features):
            plt.figure(figsize=(7, 5))
            plt.hist(
                train_features[:, i].cpu().numpy(),
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
                os.path.join(
                    self.feature_plots_dir,
                    f"feature_{i + 1}_{safe_name}.png",
                )
            )
            plt.close()

    def evaluate(
        self,
        model: torch.nn.Module,
        graphs: Sequence[Data],
        criterion: torch.nn.Module,
        description: str,
    ) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
        loader = DataLoader(graphs, batch_size=self.batch_size, shuffle=False)
        model.eval()
        total_loss = 0.0
        all_scores = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=description):
                batch = batch.to(DEVICE)
                scores = model(batch).view(-1)
                labels = batch.y.float().view(-1).to(DEVICE)
                loss = criterion(scores, labels)

                total_loss += loss.item() * batch.num_graphs
                all_scores.extend(scores.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())

        labels_np = np.asarray(all_labels, dtype=np.int64)
        scores_np = np.asarray(all_scores, dtype=np.float64)
        preds_np = (scores_np >= 0.5).astype(np.int64)

        mean_loss = total_loss / max(len(graphs), 1)
        accuracy = accuracy_score(labels_np, preds_np)
        auc_score = roc_auc_score(labels_np, scores_np)
        return mean_loss, accuracy, auc_score, labels_np, scores_np

    def train(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = JetGraphAutoencoderClassification(
            num_features=self.train_graphs[0].x.shape[1],
            smallest_dim=self.smallest_dim,
            num_reduced_edges=self.num_reduced_edges,
        ).to(DEVICE)

        num_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        logging.info(f"Model summary:\n{self.model}")
        logging.info(f"Number of trainable parameters: {num_params}")
        print(f"Model Summary:\n{self.model}")
        print(f"Number of trainable parameters: {num_params}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )
        criterion = torch.nn.BCELoss()
        train_loader = DataLoader(
            self.train_graphs,
            batch_size=self.batch_size,
            shuffle=True,
        )

        metrics = []
        best_val_auc = -float("inf")
        best_val_loss = float("inf")
        best_state_dict = None
        best_epoch = 0
        best_model_path = os.path.join(self.output_dir, "best_model.pt")
        timer = helpers_main.LeTimer()

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_loss = 0.0
            train_scores = []
            train_labels = []

            pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{self.epochs}")
            for batch in pbar:
                batch = batch.to(DEVICE)
                labels = batch.y.float().view(-1).to(DEVICE)

                optimizer.zero_grad()
                scores = self.model(batch).view(-1)
                loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch.num_graphs
                train_scores.extend(scores.detach().cpu().numpy().tolist())
                train_labels.extend(labels.detach().cpu().numpy().tolist())
                pbar.set_postfix({"loss": f"{loss.item():.5g}"})

            train_loss = epoch_loss / len(self.train_graphs)
            train_labels_np = np.asarray(train_labels, dtype=np.int64)
            train_scores_np = np.asarray(train_scores, dtype=np.float64)
            train_acc = accuracy_score(
                train_labels_np,
                (train_scores_np >= 0.5).astype(np.int64),
            )
            train_auc = roc_auc_score(train_labels_np, train_scores_np)

            val_loss, val_acc, val_auc, _, _ = self.evaluate(
                self.model,
                self.val_graphs,
                criterion,
                description=f"Val Epoch {epoch}/{self.epochs}",
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
                self.epochs,
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
            self.test_graphs,
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
            "background": self.bg_file,
            "signal": self.sg_file,
            "method": self.method,
            "knn": self.knn,
            "smallest_dim": self.smallest_dim,
            "num_reduced_edges": self.num_reduced_edges,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.initial_lr,
            "weight_decay": self.weight_decay,
            "seed": self.seed,
            "device": DEVICE,
            "max_background_events": self.max_background_events,
            "max_signal_events": self.max_signal_events,
            "node_features": self.node_features,
            "val_fraction": self.val_fraction,
            "test_fraction": self.test_fraction,
            "balance_train": self.balance_train,
            "auc": float(test_auc),
            "accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "best_epoch": int(best_epoch),
            "best_val_auc": float(best_val_auc),
            "best_val_loss": float(best_val_loss),
            "background_graphs": len(self.bg_graphs),
            "signal_graphs": len(self.sg_graphs),
            "background_train_graphs": len(self.bg_train_graphs),
            "signal_train_graphs": len(self.sg_train_graphs),
            "training_graphs_used": len(self.train_graphs),
            "background_val_graphs": len(self.bg_val_graphs),
            "signal_val_graphs": len(self.sg_val_graphs),
            "background_test_graphs": len(self.bg_test_graphs),
            "signal_test_graphs": len(self.sg_test_graphs),
            "num_trainable_parameters": int(num_params),
        }

        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logging.info(f"Saved run summary to {summary_path}")
        logging.info(f"Final supervised classifier AUC: {test_auc}")
        print(f"Final supervised classifier AUC: {test_auc:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Train Classifier Upper Bound",
        description=(
            "Train a supervised graph classifier on background and signal "
            "labels, then report held-out ROC AUC as an approximate ceiling."
        ),
    )
    parser.add_argument(
        "--background",
        "-b",
        type=str,
        default=bg_file,
        help=(
            "Path to processed .pkl background dataset. Defaults to "
            "background_file in config.yaml."
        ),
    )
    parser.add_argument(
        "--signal",
        "-s",
        type=str,
        default=sg_file,
        help=(
            "Path to processed .pkl signal dataset. Defaults to signal_file "
            "in config.yaml."
        ),
    )
    parser.add_argument(
        "--method",
        "-m",
        choices=c.GRAPH_METHODS,
        default="eta_phi",
        help="Method for building graph edges. Default: eta_phi.",
    )
    parser.add_argument(
        "--knn",
        "-n",
        type=int,
        default=config["misc"]["k_nearest_neighbors"],
        help="Nearest-neighbor count for input graph construction.",
    )
    parser.add_argument(
        "--smallest-dim",
        type=int,
        default=config["model"]["smallest_dim"],
        help="Latent graph embedding dimension.",
    )
    parser.add_argument(
        "--num-reduced-edges",
        type=int,
        default=config["model"]["num_reduced_edges"],
        help="Latent kNN edge count inside the classifier.",
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
        default=config["training"]["epochs"],
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config["training"]["initial_lr"],
        help="Initial AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python, NumPy, and PyTorch.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/run-classifier-upper-bound",
        help="Directory for plots, score arrays, metrics, and summary.json.",
    )
    parser.add_argument(
        "--max-background-events",
        type=int,
        help="Optional background row limit for a smoke test.",
    )
    parser.add_argument(
        "--max-signal-events",
        type=int,
        help="Optional signal row limit for a smoke test.",
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
        help=(
            "Downsample the larger class in the training split. This keeps BCE "
            "training from being dominated by the much larger QCD sample."
        ),
    )
    parser.add_argument(
        "--node-features",
        type=parse_node_features,
        default=DEFAULT_NODE_FEATURES,
        help=(
            "Comma-separated per-particle node features. Defaults to the same "
            "base particle features plus pdgId one-hot arrays used by the "
            "current autoencoder experiments."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.val_fraction <= 0 or args.test_fraction <= 0:
        raise ValueError("--val-fraction and --test-fraction must be positive.")
    if args.val_fraction + args.test_fraction >= 1:
        raise ValueError(
            "--val-fraction + --test-fraction must be less than 1 so training "
            "graphs remain."
        )
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    trainer = TrainClassifierUpperBound(args)
    trainer.load()
    trainer.build_graphs()
    trainer.compute_stats()
    trainer.plot_features()
    trainer.train()


if __name__ == "__main__":
    main()
