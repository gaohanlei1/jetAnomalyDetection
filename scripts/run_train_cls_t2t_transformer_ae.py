"""
Train the class-token token-to-token transformer autoencoder for jet anomaly detection.

This script trains a full reconstruction autoencoder on background jets. The
encoder compresses the event into the final CLS token. The decoder reconstructs
particles token-to-token from the shared CLS latent context and each particle's
eta/phi positional query.

Unlike the previous slot-based version, this model does not use learned decoder
slots and does not use Hungarian matching. Reconstruction is optimized with a
token-wise MSE loss.

Example:
python -u scripts/run_train_cls_t2t_transformer_ae.py \
    --background "data/processed/PT-200to400/scaledby_QCD/QCD_scaled.pkl" \
    --signal "data/processed/PT-200to400/scaledby_QCD/WJet_scaled.pkl" \
    --hidden-dim 64 \
    --latent-dim 16 \
    --num-layers 4 \
    --num-heads 4 \
    --batch-size 64 \
    --epochs 20 \
    --learning-rate 1e-4 \
    --weight-decay 1e-4 \
    --output-dir "plots/run-cls-t2t-transformer-ae"

Backward compatibility:
    --num-slots and --loss-type are still accepted so old launch commands do
    not fail, but they are ignored by this token-to-token architecture.
"""

import argparse
import json
import logging
import os
import random
import sys
from typing import List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

import constants as c
from helpers import helpers_main
from models.cls_t2t_transformer_ae import JetClassT2TTransformerAE
from preprocess.make_graphs import graph_data_loader
from visualize.plot_metrics import plot_anomaly_score, plot_roc_curve


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

DEFAULT_FEATURE_NAMES = [
    "eta",
    "phi",
    "pt",
    "d0/d0Err",
    "dz/dzErr",
    # "charge",
    # "mass",
    # "log_pt",
]


def remove_low_pt_muons(row):
    pdg_id = row["pdgId"]
    pt = row["pt"]
    mask = (np.abs(pdg_id) != 13) | (pt >= 0.4)
    for col in row.index:
        val = row[col]
        if hasattr(val, "__getitem__") and not isinstance(val, str):
            row[col] = val[mask]
    return row


def normalize_graph_features(
    graphs: List[Data],
    mean: torch.Tensor = None,
    std: torch.Tensor = None,
) -> Tuple[List[Data], torch.Tensor, torch.Tensor]:
    if mean is None or std is None:
        all_features = torch.cat([graph.x for graph in graphs], dim=0)
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        std[std == 0] = 1.0

    for graph in graphs:
        graph.x = (graph.x - mean) / std

    return graphs, mean, std


class TrainClassT2TTransformerAE:
    TRAIN_SPLIT = 0.8

    def __init__(self):
        self.args = parser.parse_args()

        self.bg_file = self.args.background
        self.sg_file = self.args.signal
        self.bg_name = helpers_main.trim_name(self.bg_file)
        self.sg_name = helpers_main.trim_name(self.sg_file)

        self.hidden_dim = self.args.hidden_dim
        self.latent_dim = self.args.latent_dim
        self.num_layers = self.args.num_layers
        self.num_heads = self.args.num_heads
        self.ffn_dim = self.args.ffn_dim
        self.dropout = self.args.dropout
        self.num_slots = self.args.num_slots
        self.loss_type = self.args.loss_type
        self.ignored_legacy_args = {}

        if self.args.num_slots is not None:
            self.ignored_legacy_args["num_slots"] = self.args.num_slots

        if self.args.loss_type is not None:
            self.ignored_legacy_args["loss_type"] = self.args.loss_type

        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.initial_lr = self.args.learning_rate
        self.weight_decay = self.args.weight_decay
        self.normalize_features = self.args.normalize_features
        self.seed = self.args.seed
        self.max_background_events = self.args.max_background_events
        self.max_signal_events = self.args.max_signal_events

        self.output_dir = self.args.output_dir
        self.feature_plots_dir = os.path.join(self.output_dir, "features")
        self.feature_names = self.args.feature_names

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.session_name = (
            f"logs/train_cls_transformer_ae_{self.bg_name}_{self.sg_name}_"
            f"{helpers_main.curr_time()}.log"
        )
        helpers_main.log_config(self.session_name)
        if self.ignored_legacy_args:
            logging.warning(
                "Ignoring legacy slot/Hungarian arguments for token-to-token model: "
                f"{self.ignored_legacy_args}"
            )
            print(
                "Warning: ignoring legacy slot/Hungarian arguments for token-to-token model: "
                f"{self.ignored_legacy_args}"
            )

    def load(self):
        if not os.path.exists(self.bg_file):
            raise FileNotFoundError(f"Background file does not exist: {self.bg_file}")
        if not os.path.exists(self.sg_file):
            raise FileNotFoundError(f"Signal file does not exist: {self.sg_file}")

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

        logging.info(f"Loaded background rows: {len(self.bg_data)}")
        logging.info(f"Loaded signal rows: {len(self.sg_data)}")
        print(f"Loaded background rows: {len(self.bg_data)}")
        print(f"Loaded signal rows: {len(self.sg_data)}")

    def build_graphs(self):
        if self.feature_names[:2] != ["eta", "phi"]:
            raise ValueError("feature_names must start with ['eta', 'phi'] for positional embedding.")

        print("Building background graphs...")
        self.bg_graphs = graph_data_loader(
            self.bg_data,
            data_label=0,
            nearest_neighbors=config["misc"]["k_nearest_neighbors"],
            device="cpu",
            method="eta_phi",
            alpha=config["training"]["alpha"],
            node_feature_names=self.feature_names,
        )

        print("Building signal graphs...")
        self.sg_graphs = graph_data_loader(
            self.sg_data,
            data_label=1,
            nearest_neighbors=config["misc"]["k_nearest_neighbors"],
            device="cpu",
            method="eta_phi",
            alpha=config["training"]["alpha"],
            node_feature_names=self.feature_names,
        )

        train_size = int(self.TRAIN_SPLIT * len(self.bg_graphs))
        # make sure to shuffle the background graphs before splitting
        np.random.shuffle(self.bg_graphs)
        self.bg_train_graphs = self.bg_graphs[:train_size]
        self.bg_test_graphs = self.bg_graphs[train_size:]

        if len(self.bg_train_graphs) == 0:
            raise ValueError("No background training graphs were created.")
        if len(self.bg_test_graphs) == 0:
            raise ValueError("No background validation graphs were created.")
        if len(self.sg_graphs) == 0:
            raise ValueError("No signal graphs were created.")

        if self.normalize_features:
            self.bg_train_graphs, self.bg_train_mean, self.bg_train_std = normalize_graph_features(
                self.bg_train_graphs
            )
            self.bg_test_graphs, _, _ = normalize_graph_features(
                self.bg_test_graphs,
                mean=self.bg_train_mean,
                std=self.bg_train_std,
            )
            self.sg_graphs, _, _ = normalize_graph_features(
                self.sg_graphs,
                mean=self.bg_train_mean,
                std=self.bg_train_std,
            )

        self._summarize_multiplicity()

        print("Feature names:", self.feature_names)
        print("Example graph.x shape:", self.bg_graphs[0].x.shape)
        print("Background graphs:", len(self.bg_train_graphs), "train,", len(self.bg_test_graphs), "val")
        print("Signal graphs:", len(self.sg_graphs))

    def _summarize_multiplicity(self):
        all_graphs = self.bg_train_graphs + self.bg_test_graphs + self.sg_graphs
        multiplicities = np.asarray([graph.x.size(0) for graph in all_graphs])

        print("Particle multiplicity summary:")
        print(pd.Series(multiplicities).describe(percentiles=[0.5, 0.9, 0.95, 0.99]))

    def compute_stats(self):
        self.all_features = torch.cat([graph.x for graph in self.bg_train_graphs], dim=0)
        self.num_features = self.all_features.shape[1]

        self.means = self.all_features.mean(dim=0)
        self.stds = self.all_features.std(dim=0)
        logging.info(f"Feature Means: {self.means}")
        logging.info(f"Feature Stds: {self.stds}")
        logging.info(f"Number of features: {self.num_features}")

    def plot_features(self):
        os.makedirs(self.feature_plots_dir, exist_ok=True)
        for i, name in enumerate(self.feature_names):
            if i >= self.num_features:
                continue

            plt.figure()
            plt.hist(
                self.all_features[:, i].cpu().numpy(),
                bins=50,
                density=True,
                color="skyblue",
                edgecolor="black",
            )
            plt.title(f"Feature {i}: {name}")
            plt.xlabel("Value")
            plt.ylabel("Density")
            plt.grid(True)
            plt.tight_layout()

            safe_name = name.replace("/", "_")
            plt.savefig(
                os.path.join(
                    self.feature_plots_dir,
                    f"feature_{self.bg_name}_{self.sg_name}_{i+1}_{safe_name}_{helpers_main.curr_time()}.png",
                )
            )
            plt.close()

    def _make_loaders(self):
        train_loader = DataLoader(
            self.bg_train_graphs,
            batch_size=self.batch_size,
            shuffle=True,
        )
        background_val_loader = DataLoader(
            self.bg_test_graphs,
            batch_size=self.batch_size,
            shuffle=False,
        )
        signal_loader = DataLoader(
            self.sg_graphs,
            batch_size=self.batch_size,
            shuffle=False,
        )
        return train_loader, background_val_loader, signal_loader

    def _evaluate_loader(self, loader, desc):
        losses = []
        data = []

        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=desc):
                batch = batch.to(DEVICE)
                output = self.model(batch)
                batch_losses = self.model.loss(output, per_event=True)
                losses.extend(batch_losses.detach().cpu().tolist())
                data.append(batch.x.detach().cpu())

        return losses, data

    def train(self):
        os.makedirs(self.output_dir, exist_ok=True)

        self.model = JetClassT2TTransformerAE(
            num_features=self.bg_train_graphs[0].x.shape[1],
            hidden_dim=self.hidden_dim,
            latent_dim=self.latent_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
        ).to(DEVICE)

        logging.info(f"Model Summary:\n{self.model}")
        print(f"Model Summary:\n{self.model}")

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"Number of trainable parameters: {num_params}")
        print(f"Number of trainable parameters: {num_params}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)

        train_loader, background_val_loader, signal_loader = self._make_loaders()

        self.model.train_hist = []
        self.model.val_hist = []
        self.model.signal_hist = []

        training_loss_history = []
        validation_loss_history = []
        epoch_end_steps = []

        best_val_loss = float("inf")
        best_model_path = os.path.join(self.output_dir, "best_model.pth")
        timer = helpers_main.LeTimer()

        def plot_progress():
            if len(training_loss_history) == 0:
                return

            train_np = np.asarray(training_loss_history)
            val_np = np.asarray(validation_loss_history)
            steps_np = np.arange(1, len(train_np) + 1)
            epoch_steps_np = np.asarray(epoch_end_steps)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(steps_np, train_np, label="Training Loss", alpha=0.7)

            if len(val_np) > 0:
                repeat_count = int(np.ceil(len(train_np) / len(val_np)))
                repeated_val = np.repeat(val_np, repeat_count)[: len(train_np)]
                ax.plot(steps_np, repeated_val, label="Validation Loss", alpha=0.7)

            if len(epoch_steps_np) > 0:
                epoch_ids = np.arange(1, len(epoch_steps_np) + 1)
                stride = max(1, int(np.ceil(len(epoch_steps_np) / 12)))
                top_ax = ax.secondary_xaxis("top")
                top_ax.set_xticks(epoch_steps_np[::stride])
                top_ax.set_xticklabels(epoch_ids[::stride])
                top_ax.set_xlabel("Epoch")

                for step_idx in epoch_steps_np[::stride]:
                    ax.axvline(step_idx, color="gray", ls="--", lw=0.6, alpha=0.35)

            if np.isfinite(best_val_loss):
                ax.axhline(
                    y=best_val_loss,
                    color="black",
                    linestyle="--",
                    linewidth=1,
                    alpha=0.2,
                    label=f"Min Val Loss: {best_val_loss:.4g}",
                )

            ax.set_xlabel("Step Number")
            ax.set_ylabel("Loss")
            ax.set_title("Loss Curves")
            ax.set_yscale("log")
            ax.legend()
            ax.grid(False)
            fig.tight_layout()
            fig.savefig(os.path.join(self.output_dir, "loss.png"))
            plt.close(fig)

        for epoch in range(self.epochs):
            logging.info(f"\nEpoch [{epoch + 1}/{self.epochs}]")
            logging.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6g}")

            self.model.train()
            epoch_train_losses = []

            pbar = tqdm(train_loader, desc=f"Train Epoch {epoch + 1}/{self.epochs}")
            for batch in pbar:
                batch = batch.to(DEVICE)

                optimizer.zero_grad()
                output = self.model(batch)
                loss = self.model.loss(output)
                loss.backward()
                optimizer.step()

                step_loss = loss.item()
                epoch_train_losses.append(step_loss)
                training_loss_history.append(step_loss)
                pbar.set_postfix({"Train Loss": f"{step_loss:.6g}"})

            mean_train_loss = float(np.nanmean(epoch_train_losses))

            self.model.eval()
            val_losses = []
            with torch.no_grad():
                pbar = tqdm(background_val_loader, desc=f"Val Epoch {epoch + 1}/{self.epochs}")
                for batch in pbar:
                    batch = batch.to(DEVICE)
                    output = self.model(batch)
                    loss = self.model.loss(output)
                    val_losses.append(loss.item())
                    pbar.set_postfix({"Val Loss": f"{loss.item():.6g}"})

            mean_val_loss = float(np.nanmean(val_losses))
            print(f"Debug: Epoch {epoch + 1}/{self.epochs}, Mean Val Loss: {mean_val_loss:.6g}, Val Loss Std: {np.nanstd(val_losses):.6g}, Val Loss Min: {np.nanmin(val_losses):.6g}, Val Loss Max: {np.nanmax(val_losses):.6g}")
            print(f"Debug: Epoch {epoch + 1}/{self.epochs}, Mean Train Loss: {mean_train_loss:.6g}, Train Loss Std: {np.nanstd(epoch_train_losses):.6g}, Train Loss Min: {np.nanmin(epoch_train_losses):.6g}, Train Loss Max: {np.nanmax(epoch_train_losses):.6g}")
            self.model.train_hist.append(mean_train_loss)
            self.model.val_hist.append(mean_val_loss)
            validation_loss_history.append(mean_val_loss)
            epoch_end_steps.append(len(training_loss_history))

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                torch.save(self.model, best_model_path)
                logging.info(f"Saved new best model to {best_model_path}")
                print(f"Saved new best model to {best_model_path}")

            plot_progress()
            scheduler.step()

            logging.info(f"train loss: {mean_train_loss}")
            logging.info(f"validation/background loss: {mean_val_loss}")
            logging.info(timer.time_taken())

        background_train_loss, background_train_data = self._evaluate_loader(
            train_loader,
            "Final Background (Training) Evaluation",
        )
        background_test_loss, background_test_data = self._evaluate_loader(
            background_val_loader,
            "Final Background (Testing) Evaluation",
        )
        signal_loss, signal_data = self._evaluate_loader(
            signal_loader,
            "Final Signal Evaluation",
        )

        self.model.background_train_loss = background_train_loss
        self.model.background_test_loss = background_test_loss
        self.model.signal_loss = signal_loss
        self.model.train_data = background_train_data
        self.model.test_data = background_test_data
        self.model.signal_data = signal_data
        self.model.signal_hist.append(float(np.nanmean(signal_loss)))

        plot_anomaly_score(
            self.model.background_test_loss,
            self.model.signal_loss,
            background_label="QCD (Test)",
            signal_label="WJet",
            save_path=os.path.join(self.output_dir, "bgtest-vs-signal-anomaly-score.png"),
        )
        
        plot_anomaly_score(
            self.model.background_train_loss,
            self.model.signal_loss,
            background_label="QCD (Train)",
            signal_label="WJet",
            save_path=os.path.join(self.output_dir, "bgtrain-vs-signal-anomaly-score.png"),
        )

        plot_roc_curve(
            self.model.background_test_loss,
            self.model.signal_loss,
            background_label="QCD (Test)",
            signal_label="WJet",
            savepath=os.path.join(self.output_dir, "roc-bgtest-vs-signal.png"),
            examples=False,
            loss_fn=torch.nn.MSELoss(reduction="mean"),
        )

        plot_roc_curve(
            self.model.background_train_loss,
            self.model.signal_loss,
            background_label="QCD (Train)",
            signal_label="WJet",
            savepath=os.path.join(self.output_dir, "roc-bgtrain-vs-signal.png"),
            examples=False,
            loss_fn=torch.nn.MSELoss(reduction="mean"),
        )

        np.save(
            os.path.join(self.output_dir, "background_test_loss.npy"),
            np.asarray(self.model.background_test_loss),
        )
        np.save(
            os.path.join(self.output_dir, "signal_loss.npy"),
            np.asarray(self.model.signal_loss),
        )

        auc_score = roc_auc_score(
            np.concatenate(
                [
                    np.zeros(len(self.model.background_test_loss)),
                    np.ones(len(self.model.signal_loss)),
                ]
            ),
            np.concatenate(
                [
                    self.model.background_test_loss,
                    self.model.signal_loss,
                ]
            ),
        )

        summary = {
            "background": self.bg_file,
            "signal": self.sg_file,
            "feature_names": self.feature_names,
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim if self.ffn_dim is not None else self.hidden_dim * 4,
            "dropout": self.dropout,
            "architecture": "cls_t2t_transformer_ae",
            "decoder_type": "position_conditioned_token_to_token",
            "loss_type": "mse",
            "ignored_legacy_args": self.ignored_legacy_args,
            "epochs": self.epochs,
            "learning_rate": self.initial_lr,
            "weight_decay": self.weight_decay,
            "lr_scheduler": "CosineAnnealingLR",
            "normalize_features": self.normalize_features,
            "seed": self.seed,
            "device": DEVICE,
            "max_background_events": self.max_background_events,
            "max_signal_events": self.max_signal_events,
            "auc": float(auc_score),
            "background_train_graphs": len(self.bg_train_graphs),
            "background_test_graphs": len(self.bg_test_graphs),
            "signal_graphs": len(self.sg_graphs),
            "best_val_loss": float(best_val_loss),
            "final_val_loss": float(np.nanmean(self.model.background_test_loss)),
            "final_signal_loss": float(np.nanmean(self.model.signal_loss)),
            "num_trainable_parameters": int(num_params),
        }

        summary_path = os.path.join(self.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logging.info(f"Saved run summary to {summary_path}")
        logging.info(f"Final AUC: {auc_score}")
        print(f"Final AUC: {auc_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train Class-Token Transformer AE",
        description="trains the class-token transformer autoencoder on processed jet data",
    )
    parser.add_argument(
        "--background",
        "-b",
        type=str,
        default=bg_file,
        help="Path to processed .pkl background dataset. Defaults to config.yaml.",
    )
    parser.add_argument(
        "--signal",
        "-s",
        type=str,
        default=sg_file,
        help="Path to processed .pkl signal dataset. Defaults to config.yaml.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=config["model"]["hidden_dim"],
        help="Transformer hidden dimension. Defaults to config.",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=config["model"]["smallest_dim"],
        help="Class-token bottleneck dimension. Defaults to model.smallest_dim in config.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers in both encoder and decoder. Default: 4.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of transformer attention heads. Default: 4.",
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=None,
        help="Transformer FFN hidden dimension. Default: 4 * hidden_dim.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout used inside transformer blocks. Default: 0.1.",
    )
    parser.add_argument(
        "--num-slots",
        type=int,
        default=None,
        help=(
            "Deprecated compatibility argument from the slot-based model. "
            "Accepted but ignored by the token-to-token architecture."
        ),
    )
    parser.add_argument(
        "--loss-type",
        choices=["hungarian", "index", "mse"],
        default=None,
        help=(
            "Deprecated compatibility argument from earlier reconstruction losses. "
            "Accepted but ignored; this model always uses token-wise MSE over valid particle tokens"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config["model"]["batch_size"],
        help="Training batch size. Defaults to config.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config["training"]["epochs"],
        help="Number of training epochs. Defaults to config.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config["training"]["initial_lr"],
        help="Initial AdamW learning rate. Defaults to config.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay. Default: 1e-4.",
    )
    parser.add_argument(
        "--normalize-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize graph features using background-training statistics.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Python, NumPy, and PyTorch. Default: 42.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/run-cls-t2t-transformer-ae",
        help="Directory for plots, losses, checkpoints, and summary.json.",
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
        "--feature-names",
        nargs="+",
        default=DEFAULT_FEATURE_NAMES,
        help="Ordered node feature names. The first two must be eta phi.",
    )

    trainer = TrainClassT2TTransformerAE()
    trainer.load()
    trainer.build_graphs()
    trainer.compute_stats()
    trainer.plot_features()
    trainer.train()
