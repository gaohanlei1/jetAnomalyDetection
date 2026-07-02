"""
Evaluate a LeJEPA ParticleTransformer representation model with Mahalanobis scores.

This script loads a finished training run directory containing:

    summary.json
    best_model.pth

It then:
    1. Loads the background and signal datasets recorded in summary.json.
    2. Reconstructs the same background train/validation split using the saved seed.
    3. Computes full-jet latent representations without augmentation.
    4. Fits a background-only Mahalanobis model using background train latents.
    5. Computes per-event anomaly scores for:
        - background train vs signal
        - background validation vs signal
    6. Plots anomaly-score distributions and ROC curves.

The anomaly score is not batch-averaged. Each event receives one scalar score:

    score(z) = (z - mu_bg)^T Sigma_bg^{-1} (z - mu_bg)

Example:

python -u scripts/run_eval_lejepa_part.py \
    --run-dir plots/run-lejepa-part
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from helpers import helpers_main
from visualize.plot_metrics import plot_anomaly_score, plot_roc_curve

DEVICE = torch.device(helpers_main.get_device())
TRAIN_SPLIT = 0.8


def parse_node_features_from_summary(summary: Dict) -> List[str]:
    node_features = summary.get("node_features")
    if node_features is None:
        raise KeyError("summary.json does not contain `node_features`.")
    if not isinstance(node_features, list) or len(node_features) == 0:
        raise ValueError(f"Invalid node_features in summary.json: {node_features}")
    return [str(name) for name in node_features]


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
) -> Tuple[List[torch.Tensor], List[int]]:
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


def compute_feature_stats(node_tensors: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
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
    return [(x - mean) / std for x in node_tensors]


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


def load_torch_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    Load a full-model checkpoint saved with torch.save(model, path).
    """

    try:
        model = torch.load(model_path, map_location="cpu", weights_only=False)
        # cpu load to prevent 
        # SystemError: <built-in method __setstate__ of torch._C.Generator object at 0x7f1d91231eb0> returned a result with an exception set
    except TypeError:
        model = torch.load(model_path, map_location="cpu")

    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_representations(
    model: torch.nn.Module,
    loader: DataLoader,
    precision: str,
    normalize_output: bool,
) -> np.ndarray:
    model.eval()
    latents: List[np.ndarray] = []
    dtype = precision_to_dtype(precision)
    use_autocast = autocast_enabled_for_precision(precision)

    for batch in tqdm(loader, desc="Collecting latents"):
        x = batch["x"].to(DEVICE, non_blocking=True)
        padding_mask = batch["padding_mask"].to(DEVICE, non_blocking=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=dtype,
            enabled=use_autocast,
        ):
            z = model(
                x,
                padding_mask=padding_mask,
                normalize_output=normalize_output,
            )

        latents.append(z.detach().float().cpu().numpy())

    return np.concatenate(latents, axis=0)


def fit_mahalanobis_background(
    bg_train_latents: np.ndarray,
    cov_eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a Gaussian background model from background train latents.
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
    cov = cov + cov_eps * np.eye(cov.shape[0], dtype=np.float64)
    precision = np.linalg.pinv(cov)

    return mean.astype(np.float64), precision.astype(np.float64)


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
    scores = np.einsum("nd,dd,nd->n", centered, precision, centered)
    return scores.astype(np.float64)


def compute_auc(background_scores: np.ndarray, signal_scores: np.ndarray) -> float:
    y_true = np.concatenate(
        [
            np.zeros(len(background_scores), dtype=np.int64),
            np.ones(len(signal_scores), dtype=np.int64),
        ]
    )
    y_score = np.concatenate([background_scores, signal_scores])
    return float(roc_auc_score(y_true, y_score))


class LeJEPAEvaluator:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_dir = args.run_dir
        self.summary_path = os.path.join(self.run_dir, "summary.json")
        self.model_path = os.path.join(self.run_dir, "best_model.pth")
        self.output_dir = args.output_dir or os.path.join(self.run_dir, "mahalanobis_eval")

        os.makedirs(self.output_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def load_summary(self) -> None:
        if not os.path.exists(self.summary_path):
            raise FileNotFoundError(f"summary.json not found: {self.summary_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"best_model.pth not found: {self.model_path}")

        with open(self.summary_path, "r") as f:
            self.summary = json.load(f)

        self.background_path = self.summary["background"]
        self.signal_path = self.summary["signal"]
        self.node_feature_names = parse_node_features_from_summary(self.summary)
        self.min_nodes = int(self.summary.get("min_nodes", self.args.min_nodes))
        self.seed = int(self.summary.get("seed", self.args.seed))
        self.precision = self.args.precision or self.summary.get("precision", "bf16")
        self.normalize_features = bool(self.summary.get("normalize_features", False))
        self.batch_size = int(self.args.batch_size or self.summary.get("batch_size", 128))

        self.normalize_output = bool(
            self.args.normalize_output_representations
            or self.summary.get("normalize_output_representations", False)
        )

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        logging.info(f"Loaded summary from {self.summary_path}")
        logging.info(f"Background: {self.background_path}")
        logging.info(f"Signal: {self.signal_path}")
        logging.info(f"Node features: {self.node_feature_names}")
        logging.info(f"Device: {DEVICE}")

    def load_datasets(self) -> None:
        bg_df = pd.read_pickle(self.background_path)
        sg_df = pd.read_pickle(self.signal_path)

        if self.args.max_background_events is not None:
            bg_df = bg_df.head(self.args.max_background_events)
        if self.args.max_signal_events is not None:
            sg_df = sg_df.head(self.args.max_signal_events)

        logging.info(f"Loaded background rows: {len(bg_df)}")
        logging.info(f"Loaded signal rows: {len(sg_df)}")

        print("Loading background node tensors...")
        bg_nodes, bg_labels = dataframe_to_node_tensors(
            df=bg_df,
            node_feature_names=self.node_feature_names,
            label=0,
            min_nodes=self.min_nodes,
        )

        print("Loading signal node tensors...")
        sg_nodes, sg_labels = dataframe_to_node_tensors(
            df=sg_df,
            node_feature_names=self.node_feature_names,
            label=1,
            min_nodes=self.min_nodes,
        )

        if len(bg_nodes) == 0:
            raise ValueError("No valid background events were loaded.")
        if len(sg_nodes) == 0:
            raise ValueError("No valid signal events were loaded.")

        train_size = int(TRAIN_SPLIT * len(bg_nodes))
        train_size = max(1, min(train_size, len(bg_nodes) - 1))

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

        if self.normalize_features:
            logging.warning(
                "Applying feature normalization from background train statistics "
                "to match the training script."
            )
            feature_mean, feature_std = compute_feature_stats(self.bg_train_nodes)
            self.bg_train_nodes = apply_feature_normalization(
                self.bg_train_nodes,
                feature_mean,
                feature_std,
            )
            self.bg_val_nodes = apply_feature_normalization(
                self.bg_val_nodes,
                feature_mean,
                feature_std,
            )
            self.sg_nodes = apply_feature_normalization(
                self.sg_nodes,
                feature_mean,
                feature_std,
            )

        self.bg_train_dataset = JetNodeDataset(self.bg_train_nodes, self.bg_train_labels)
        self.bg_val_dataset = JetNodeDataset(self.bg_val_nodes, self.bg_val_labels)
        self.sg_dataset = JetNodeDataset(self.sg_nodes, self.sg_labels)

        logging.info(f"Background train events: {len(self.bg_train_dataset)}")
        logging.info(f"Background val events: {len(self.bg_val_dataset)}")
        logging.info(f"Signal events: {len(self.sg_dataset)}")

    def make_loaders(self) -> None:
        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=collate_node_tensors,
        )

        self.bg_train_loader = DataLoader(self.bg_train_dataset, **loader_kwargs)
        self.bg_val_loader = DataLoader(self.bg_val_dataset, **loader_kwargs)
        self.signal_loader = DataLoader(self.sg_dataset, **loader_kwargs)

    def load_model(self) -> None:
        self.model = load_torch_model(self.model_path, DEVICE)
        logging.info(f"Loaded model from {self.model_path}")

    def collect_all_latents(self) -> None:
        logging.info("Collecting background train latents...")
        self.bg_train_latents = collect_representations(
            model=self.model,
            loader=self.bg_train_loader,
            precision=self.precision,
            normalize_output=self.normalize_output,
        )

        logging.info("Collecting background validation latents...")
        self.bg_val_latents = collect_representations(
            model=self.model,
            loader=self.bg_val_loader,
            precision=self.precision,
            normalize_output=self.normalize_output,
        )

        logging.info("Collecting signal latents...")
        self.signal_latents = collect_representations(
            model=self.model,
            loader=self.signal_loader,
            precision=self.precision,
            normalize_output=self.normalize_output,
        )

        np.save(os.path.join(self.output_dir, "background_train_latents.npy"), self.bg_train_latents)
        np.save(os.path.join(self.output_dir, "background_val_latents.npy"), self.bg_val_latents)
        np.save(os.path.join(self.output_dir, "signal_latents.npy"), self.signal_latents)

        logging.info(f"Background train latents: {self.bg_train_latents.shape}")
        logging.info(f"Background val latents: {self.bg_val_latents.shape}")
        logging.info(f"Signal latents: {self.signal_latents.shape}")

    def compute_scores(self) -> None:
        mean, precision = fit_mahalanobis_background(
            bg_train_latents=self.bg_train_latents,
            cov_eps=self.args.cov_eps,
        )

        self.background_train_scores = mahalanobis_scores(
            self.bg_train_latents,
            mean,
            precision,
        )
        self.background_val_scores = mahalanobis_scores(
            self.bg_val_latents,
            mean,
            precision,
        )
        self.signal_scores = mahalanobis_scores(
            self.signal_latents,
            mean,
            precision,
        )

        self.auc_bgtrain_vs_signal = compute_auc(
            self.background_train_scores,
            self.signal_scores,
        )
        self.auc_bgval_vs_signal = compute_auc(
            self.background_val_scores,
            self.signal_scores,
        )

        np.save(
            os.path.join(self.output_dir, "background_train_mahalanobis_scores.npy"),
            self.background_train_scores,
        )
        np.save(
            os.path.join(self.output_dir, "background_val_mahalanobis_scores.npy"),
            self.background_val_scores,
        )
        np.save(
            os.path.join(self.output_dir, "signal_mahalanobis_scores.npy"),
            self.signal_scores,
        )

        metrics = {
            "auc_bgtrain_vs_signal": self.auc_bgtrain_vs_signal,
            "auc_bgval_vs_signal": self.auc_bgval_vs_signal,
            "background_train_score_mean": float(np.mean(self.background_train_scores)),
            "background_val_score_mean": float(np.mean(self.background_val_scores)),
            "signal_score_mean": float(np.mean(self.signal_scores)),
            "background_train_score_median": float(np.median(self.background_train_scores)),
            "background_val_score_median": float(np.median(self.background_val_scores)),
            "signal_score_median": float(np.median(self.signal_scores)),
            "cov_eps": self.args.cov_eps,
        }

        metrics_path = os.path.join(self.output_dir, "mahalanobis_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Mahalanobis AUC, QCD train vs WJet: {self.auc_bgtrain_vs_signal:.6f}")
        print(f"Mahalanobis AUC, QCD val vs WJet: {self.auc_bgval_vs_signal:.6f}")
        logging.info(f"Saved metrics to {metrics_path}")

    def plot(self) -> None:
        plot_anomaly_score(
            self.background_val_scores,
            self.signal_scores,
            background_label="QCD (Val)",
            signal_label="WJet",
            save_path=os.path.join(self.output_dir, "bgval-vs-signal-mahalanobis-score.png"),
        )

        plot_anomaly_score(
            self.background_train_scores,
            self.signal_scores,
            background_label="QCD (Train)",
            signal_label="WJet",
            save_path=os.path.join(self.output_dir, "bgtrain-vs-signal-mahalanobis-score.png"),
        )

        plot_roc_curve(
            self.background_val_scores,
            self.signal_scores,
            background_label="QCD (Val)",
            signal_label="WJet",
            savepath=os.path.join(self.output_dir, "roc-bgval-vs-signal-mahalanobis.png"),
            examples=False,
            loss_fn=torch.nn.MSELoss(reduction="mean"),
        )

        plot_roc_curve(
            self.background_train_scores,
            self.signal_scores,
            background_label="QCD (Train)",
            signal_label="WJet",
            savepath=os.path.join(self.output_dir, "roc-bgtrain-vs-signal-mahalanobis.png"),
            examples=False,
            loss_fn=torch.nn.MSELoss(reduction="mean"),
        )

    def run(self) -> None:
        self.load_summary()
        self.load_datasets()
        self.make_loaders()
        self.load_model()
        self.collect_all_latents()
        self.compute_scores()
        self.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Evaluate LeJEPA ParticleTransformer with Mahalanobis scores",
        description="Load a LeJEPA ParticleTransformer run and evaluate Mahalanobis anomaly scores.",
    )

    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Training run directory containing summary.json and best_model.pth.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Default: <run-dir>/mahalanobis_eval.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size. Defaults to summary.json batch_size.",
    )
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default=None,
        help="Precision for latent extraction. Defaults to summary.json precision.",
    )
    parser.add_argument(
        "--normalize-output-representations",
        action="store_true",
        help="L2-normalize model output representations during latent extraction.",
    )
    parser.add_argument(
        "--cov-eps",
        type=float,
        default=1e-4,
        help="Diagonal regularization added to the background covariance. Default: 1e-4.",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=4,
        help="Fallback minimum valid nodes if missing from summary.json. Default: 4.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Fallback seed if missing from summary.json. Default: 42.",
    )
    parser.add_argument(
        "--max-background-events",
        type=int,
        default=None,
        help="Optional background row limit for quick tests.",
    )
    parser.add_argument(
        "--max-signal-events",
        type=int,
        default=None,
        help="Optional signal row limit for quick tests.",
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

    evaluator = LeJEPAEvaluator(parser.parse_args())
    evaluator.run()