"""
Script to train a transformer-based masked encoder for jet anomaly detection.

This script:
- Loads configuration and preprocessed datasets.
- Constructs PyG event objects using node features.
- Trains the JetTransformerMaskedEncoder model on background data.
- Evaluates the model on background and signal samples using repeated random masks.
- Plots anomaly scores, ROC curves, and training loss histories.

Example command:
python -u scripts/run_train_transformer.py \
    --background "data/processed/qcd-vs-wjet-pt-200to400/QCD_scaled_scaled.pkl" \
    --signal "data/processed/qcd-vs-wjet-pt-200to400/WJet_scaled_scaled.pkl" \
    --hidden-dim 16 \
    --num-layers 4 \
    --num-heads 4 \
    --batch-size 128 \
    --epochs 20 \
    --learning-rate 1e-4 \
    --weight-decay 1e-4 \
    --train-mask-ratio 0.3 \
    --test-mask-ratio 0.3 \
    --eval-mask-repeats 10 \
    --no-lr-scheduler \
    --no-normalize-features \
    --seed 42 \
    --output-dir "plots/run-transformer"
"""

import sys
import os
import json
import random

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import constants as c
from helpers import helpers_main

import torch
import pandas as pd
import numpy as np
import yaml
import argparse
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from models.transformer import JetTransformerMaskedEncoder
from preprocess.make_graphs import graph_data_loader
from visualize.plot_metrics import plot_loss, plot_anomaly_score, plot_roc_curve
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from typing import List, Tuple
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from helpers import join_dfs
config = helpers_main.load_config()

import logging

# File paths for background and signal data
bg_file = os.path.join(config['data']['processed_data_dir'], config['data']['background_file'])
sg_file = os.path.join(config['data']['processed_data_dir'], config['data']['signal_file'])
DEVICE = helpers_main.get_device()

# Only for WminsH, to remove leptonic jets 
def remove_low_pt_muons(row):
    pdgId = row['pdgId']
    pt = row['pt']
    mask = (np.abs(pdgId) != 13) | (pt >= 0.4)
    for col in row.index:
        val = row[col]
        # Apply mask only if val is indexable (e.g., np.ndarray or list)
        if hasattr(val, "__getitem__") and not isinstance(val, str):
            row[col] = val[mask]
    return row

"""
Training utility functions for JetTransformerMaskedEncoder models.

This module includes:
- `train_loop`: One epoch of training over a data loader.
- `eval_loop`: Evaluation on a dataset with optional labeling for background/signal.
- `train_model`: Full training procedure across multiple epochs for background/signal separation.
"""

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch_geometric.data import Data
from typing import List, Tuple

import os 
import sys

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import constants as c
from helpers import helpers_main
config = helpers_main.load_config()

import logging
helpers_main.log_config(f"logs/train_{helpers_main.curr_time()}.log")


def normalize_graph_features(
    graphs: List[Data],
    mean: torch.Tensor = None,
    std: torch.Tensor = None
) -> Tuple[List[Data], torch.Tensor, torch.Tensor]:
    """
    Normalize features across all graphs using the global mean and std.

    Args:
        graphs (List[Data]): List of PyG graphs with node features.
        mean (torch.Tensor, optional): If given, use this mean to normalize.
        std (torch.Tensor, optional): If given, use this std to normalize.

    Returns:
        Tuple containing:
        - List[Data]: Normalized graphs
        - torch.Tensor: Feature mean used
        - torch.Tensor: Feature std used
    """
    if mean is None or std is None:
        all_features = torch.cat([graph.x for graph in graphs], dim=0)
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        std[std == 0] = 1.0  # Prevent divide-by-zero

    # Normalize each graph
    for graph in graphs:
        graph.x = (graph.x - mean) / std

    return graphs, mean, std

class TrainMaskedTransformer:
    # Packaged into a class for variable management
    TRAIN_SPLIT = 0.8
    FEATURE_PLOTS_PATH = "plots/test-plots/features"
    TRAIN_PLOTS_PATH   = "plots/test-plots"
    
    def __init__(self):
        self.args = parser.parse_args()
        self.bg_file, self.sg_file = self.args.background, self.args.signal
        self.bg_name, self.sg_name = helpers_main.trim_name(self.bg_file), helpers_main.trim_name(self.sg_file)
        self.hidden_dim = self.args.hidden_dim
        self.num_heads = self.args.num_heads
        self.ffn_dim = self.args.ffn_dim
        self.dropout = self.args.dropout
        self.train_mask_ratio = self.args.train_mask_ratio
        self.test_mask_ratio = self.args.test_mask_ratio
        self.eval_mask_repeats = self.args.eval_mask_repeats
        self.batch_size = self.args.batch_size
        self.epochs = self.args.epochs
        self.initial_lr = self.args.learning_rate
        self.weight_decay = self.args.weight_decay
        self.lr_scheduler = self.args.lr_scheduler
        self.normalize_features = self.args.normalize_features
        self.seed = self.args.seed
        self.max_background_events = self.args.max_background_events
        self.max_signal_events = self.args.max_signal_events
        self.TRAIN_PLOTS_PATH = self.args.output_dir
        self.FEATURE_PLOTS_PATH = os.path.join(self.args.output_dir, "features")
        self.num_layers = self.args.num_layers

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        self.session_name = f"logs/train_transformer_{self.bg_name}_{self.sg_name}_{helpers_main.curr_time()}.log"
        helpers_main.log_config(self.session_name)

    def load(self):
        # Load datasets from pickle files
        self.bg_data = pd.read_pickle(self.bg_file)
        self.sg_data = pd.read_pickle(self.sg_file)
        if self.max_background_events is not None:
            self.bg_data = self.bg_data.head(self.max_background_events)
        if self.max_signal_events is not None:
            self.sg_data = self.sg_data.head(self.max_signal_events)

        print(self.bg_data.columns.tolist())

        # Slice pT; modify bounds in constants
        pt_max = c.PT_MAX
        pt_min = c.PT_MIN

        rawfj_pt_col = c.RAW_FATJET_PROPERTIES_PREFIX + "pt"
        if rawfj_pt_col in self.bg_data:
            self.bg_data = self.bg_data[
                (self.bg_data[rawfj_pt_col] > pt_min) 
                & (self.bg_data[rawfj_pt_col] < pt_max)
            ]
        # logging.info(f"Signal Data Columns: {self.sg_data.columns.tolist()}")
        if rawfj_pt_col in self.sg_data:
            self.sg_data = self.sg_data[
                (self.sg_data[rawfj_pt_col] > pt_min)
                & (self.sg_data[rawfj_pt_col] < pt_max)
            ]
        # Only for WminusH - This removes the leptonic jet
        if "WminusH" in self.sg_file: self.sg_data = self.sg_data.apply(remove_low_pt_muons, axis=1)

        print("Background pt stats after slicing:")
        print(self.bg_data[rawfj_pt_col].describe())
        print("Signal pt stats after slicing:")
        print(self.sg_data[rawfj_pt_col].describe())

        logging.info(f"Number of training events after slicing: {len(self.bg_data)}")
        logging.info(f"Number of test events after removing leptonic jet: {len(self.sg_data)}")
        logging.info(f"\nSample background pt values:\n{self.bg_data['pt'].head().to_string()}")
        logging.info(f"Sample signal pt values:\n{self.sg_data['pt'].head().to_string()}")

    def build_graphs(self):
        print("Building background graphs...")
        print(f"Number of background rows before graph construction: {len(self.bg_data)}")
        print("Transformer model ignores graph edges; graph construction is used only to package node features.")

        self.feature_names = [
            'eta', 'phi', 'pt', 'd0/d0Err', 'dz/dzErr', 'charge', 
            'mass', 'log_pt'
        ] # Note: 'eta' and 'phi' must be the first two features, as they are used for positional encoding in the transformer model!!
        # pdg_features = [
        #     "pdgId_-211",
        #     "pdgId_-13",
        #     "pdgId_-11",
        #     "pdgId_11",
        #     "pdgId_13",
        #     "pdgId_22",
        #     "pdgId_130",
        #     "pdgId_211",
        # ]
        # self.feature_names += pdg_features
        self.bg_graphs = graph_data_loader(
            self.bg_data,
            data_label=0,
            nearest_neighbors=config["misc"]["k_nearest_neighbors"],
            device="cpu",
            method="eta_phi",
            alpha=config["training"]["alpha"],
            node_feature_names=self.feature_names,
        )

        print("Finished background graphs.")
        print(f"Number of background graphs: {len(self.bg_graphs)}")

        print("Building signal graphs...")
        print(f"Number of signal rows before graph construction: {len(self.sg_data)}")

        self.sg_graphs = graph_data_loader(
            self.sg_data,
            data_label=1,
            nearest_neighbors=config["misc"]["k_nearest_neighbors"],
            device="cpu",
            method="eta_phi",
            alpha=config["training"]["alpha"],
            node_feature_names=self.feature_names,
        )

        print("Finished signal graphs.")
        print(f"Number of signal graphs: {len(self.sg_graphs)}")

        train_size = int(self.TRAIN_SPLIT * len(self.bg_graphs))
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
        
        print("Feature names:", self.feature_names)
        print("Example graph.x shape:", self.bg_graphs[0].x.shape)
        print("Example graph.y:", getattr(self.bg_graphs[0], "y", None))
        print("Example bg x first rows:", self.bg_graphs[0].x[:5])
        print("Example sg x first rows:", self.sg_graphs[0].x[:5])
        bg_x = torch.cat([g.x for g in self.bg_graphs], dim=0).numpy()
        sg_x = torch.cat([g.x for g in self.sg_graphs], dim=0).numpy()

        for i, name in enumerate(self.feature_names):
            if i < bg_x.shape[1]:
                print(
                    i,
                    name,
                    "bg mean/std:", bg_x[:, i].mean(), bg_x[:, i].std(),
                    "sg mean/std:", sg_x[:, i].mean(), sg_x[:, i].std(),
                )
        
    def compute_stats(self):
        self.all_features = torch.cat([graph.x for graph in self.bg_train_graphs], dim=0)
        self.num_features = self.all_features.shape[1]

        # Compute mean and std per feature dimension
        self.means = self.all_features.mean(dim=0)
        self.stds  = self.all_features.std(dim=0)
        logging.info(f"Feature Means: {self.means}")
        logging.info(f"Feature Stds: {self.stds}")
        logging.info(f"Number of features: {self.num_features}")
    
    def plot_features(self):
        # Plot each feature's distribution
        os.makedirs(self.FEATURE_PLOTS_PATH, exist_ok=True)
        for i in range(self.num_features):
            plt.figure()
            plt.hist(
                self.all_features[:, i].cpu().numpy(), bins=50, density=True, color='skyblue', edgecolor='black'
            )
            plt.title(f"Feature {i}: {self.feature_names[i] if i < len(self.feature_names) else f'Feature {i}'}")
            plt.xlabel("Value")
            plt.ylabel("Count")
            plt.grid(True)
            plt.tight_layout()

            safe_name = self.feature_names[i].replace('/', '_') if i < len(self.feature_names) else str(i)
            plt.savefig(os.path.join(
                self.FEATURE_PLOTS_PATH,
                f"feature_{self.bg_name}_{self.sg_name}_{i+1}_{safe_name}_{helpers_main.curr_time()}.png"
            ))
            plt.clf()
    
    def train(self):
        """
        Train the transformer, validate on held-out background graphs, and finally
        evaluate anomaly performance by comparing background validation graphs
        against signal graphs.
        """
        os.makedirs(self.TRAIN_PLOTS_PATH, exist_ok=True)

        # ------------------------------------------------------------------
        # 1. Setup model
        # ------------------------------------------------------------------
        self.model = JetTransformerMaskedEncoder(
            num_features=self.bg_train_graphs[0].x.shape[1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            mask_ratio=self.train_mask_ratio,
        ).to(DEVICE)

        logging.info(f"Model Summary:\n{self.model}")
        # also print
        print(f"Model Summary:\n{self.model}")
        num_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logging.info(f"Number of trainable parameters: {num_params}")
        # also print
        print(f"Number of trainable parameters: {num_params}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
        )

        scheduler = (
            StepLR(optimizer, step_size=10, gamma=0.7)
            if self.lr_scheduler
            else None
        )

        # ------------------------------------------------------------------
        # 2. Setup dataloaders
        # ------------------------------------------------------------------
        train_loader = DataLoader(
            self.bg_train_graphs,
            batch_size=self.batch_size,
            shuffle=True,
        )

        # This is validation background, not final test in the usual ML sense.
        background_val_loader = DataLoader(
            self.bg_test_graphs,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # Signal is only used for final anomaly-score / ROC evaluation.
        signal_loader = DataLoader(
            self.sg_graphs,
            batch_size=self.batch_size,
            shuffle=False,
        )

        # ------------------------------------------------------------------
        # 3. Histories
        # ------------------------------------------------------------------
        self.model.train_hist = []          # epoch-level train loss
        self.model.val_hist = []            # epoch-level background-val loss
        self.model.signal_hist = []         # kept for compatibility, final only

        training_loss_history = []          # step-level train loss
        validation_loss_history = []        # epoch-level val loss
        epoch_end_steps = []

        best_val_loss = float("inf")
        best_model_path = os.path.join(self.TRAIN_PLOTS_PATH, "best_model.pth")
        timer = helpers_main.LeTimer()

        # ------------------------------------------------------------------
        # 4. Loss plotting helper
        # ------------------------------------------------------------------
        def plot_progress():
            if len(training_loss_history) == 0:
                return

            training_loss_history_np = np.asarray(training_loss_history)
            validation_loss_history_np = np.asarray(validation_loss_history)

            step_num_history_np = np.arange(
                1, len(training_loss_history_np) + 1
            )
            epoch_end_steps_np = np.asarray(epoch_end_steps)

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(
                step_num_history_np,
                training_loss_history_np,
                label="Training Loss",
                alpha=0.7,
            )

            if len(validation_loss_history_np) > 0:
                # Repeat each epoch-level validation loss across roughly one
                # epoch of training steps, matching the template style.
                repeat_count = int(
                    np.ceil(
                        len(training_loss_history_np)
                        / len(validation_loss_history_np)
                    )
                )
                repeated_val_loss = np.repeat(
                    validation_loss_history_np,
                    repeat_count,
                )[: len(training_loss_history_np)]

                ax.plot(
                    step_num_history_np,
                    repeated_val_loss,
                    label="Validation Loss",
                    alpha=0.7,
                )

            for step_idx in epoch_end_steps_np:
                ax.axvline(
                    step_idx,
                    color="gray",
                    ls="--",
                    lw=0.6,
                    alpha=0.35,
                )

            if len(epoch_end_steps_np) > 0:
                epoch_ids = np.arange(1, len(epoch_end_steps_np) + 1)
                max_labels = 12
                stride = max(
                    1,
                    int(np.ceil(len(epoch_end_steps_np) / max_labels)),
                )

                top_ticks = epoch_end_steps_np[::stride]
                top_labels = epoch_ids[::stride]

                top_ax = ax.secondary_xaxis("top")
                top_ax.set_xticks(top_ticks)
                top_ax.set_xticklabels(top_labels)
                top_ax.set_xlabel("Epoch")

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
            ax.legend()
            ax.grid(False)

            fig.tight_layout()
            fig.savefig(os.path.join(self.TRAIN_PLOTS_PATH, "loss.png"))
            plt.close(fig)

        # ------------------------------------------------------------------
        # 5. Training + validation loop
        # ------------------------------------------------------------------
        for epoch in range(self.epochs):
            logging.info(f"\nEpoch [{epoch + 1}/{self.epochs}]")

            current_lr = optimizer.param_groups[0]["lr"]
            logging.info(f"Learning Rate: {current_lr:.6f}")

            # ------------------------------
            # Training phase
            # ------------------------------
            self.model.train()
            epoch_train_losses = []

            pbar = tqdm(
                train_loader,
                desc=f"Train Epoch {epoch + 1}/{self.epochs}",
            )

            for batch in pbar:
                batch = batch.to(DEVICE)

                optimizer.zero_grad()
                output = self.model(batch, mask_ratio=self.train_mask_ratio)
                loss = self.model.loss(output)
                loss.backward()
                optimizer.step()

                step_loss = loss.item()
                epoch_train_losses.append(step_loss)
                training_loss_history.append(step_loss)

                pbar.set_postfix({"Train Loss": f"{step_loss:.6g}"})

            mean_train_loss = float(np.nanmean(epoch_train_losses))

            # ------------------------------
            # Validation phase
            # ------------------------------
            self.model.eval()
            val_losses = []

            pbar = tqdm(
                background_val_loader,
                desc=f"Val Epoch {epoch + 1}/{self.epochs}",
            )

            with torch.no_grad():
                for batch in pbar:
                    batch = batch.to(DEVICE)
                    output = self.model(batch, mask_ratio=self.test_mask_ratio)
                    loss = self.model.loss(output)
                    val_loss = loss.item()

                    val_losses.append(val_loss)
                    pbar.set_postfix({"Val Loss": f"{val_loss:.6g}"})

            mean_val_loss = float(np.nanmean(val_losses))

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

            if scheduler is not None:
                scheduler.step()

            logging.info(f"train loss: {mean_train_loss}")
            logging.info(f"validation/background loss: {mean_val_loss}")
            logging.info(timer.time_taken())

        # ------------------------------------------------------------------
        # 6. Final evaluation for anomaly detection
        # ------------------------------------------------------------------
        self.model.eval()

        background_train_loss = []
        background_train_data = []
        
        # reconstruct the train, val, and test loader to use batch size of 1
        # to ensure event-level losses, as using batch-average could reduce
        # std of loss distribution.
        
        train_loader = DataLoader(
            self.bg_train_graphs,
            batch_size=1,
            shuffle=True,
        )

        # This is validation background, not final test in the usual ML sense.
        background_val_loader = DataLoader(
            self.bg_test_graphs,
            batch_size=1,
            shuffle=False,
        )

        # Signal is only used for final anomaly-score / ROC evaluation.
        signal_loader = DataLoader(
            self.sg_graphs,
            batch_size=1,
            shuffle=False,
        )
        
        with torch.no_grad():
            for batch in tqdm(
                train_loader,
                desc="Final Background (Training) Evaluation",
            ):
                batch = batch.to(DEVICE)
                repeated_losses = []
                for _ in range(self.eval_mask_repeats):
                    output = self.model(batch, mask_ratio=self.test_mask_ratio)
                    repeated_losses.append(self.model.loss(output).item())
                background_train_loss.append(float(np.mean(repeated_losses)))
                background_train_data.append(batch.x.detach().cpu())
        
        background_test_loss = []
        background_test_data = []

        with torch.no_grad():
            for batch in tqdm(
                background_val_loader,
                desc="Final Background (Testing) Evaluation",
            ):
                batch = batch.to(DEVICE)
                repeated_losses = []
                for _ in range(self.eval_mask_repeats):
                    output = self.model(batch, mask_ratio=self.test_mask_ratio)
                    repeated_losses.append(self.model.loss(output).item())
                background_test_loss.append(float(np.mean(repeated_losses)))
                background_test_data.append(batch.x.detach().cpu())

        signal_loss = []
        signal_data = []

        with torch.no_grad():
            for batch in tqdm(
                signal_loader,
                desc="Final Signal Evaluation",
            ):
                batch = batch.to(DEVICE)
                repeated_losses = []
                for _ in range(self.eval_mask_repeats):
                    output = self.model(batch, mask_ratio=self.test_mask_ratio)
                    repeated_losses.append(self.model.loss(output).item())
                signal_loss.append(float(np.mean(repeated_losses)))
                signal_data.append(batch.x.detach().cpu())

        self.model.background_test_loss = background_test_loss
        self.model.background_train_loss = background_train_loss
        self.model.signal_loss = signal_loss

        self.model.train_data = background_train_data
        self.model.test_data = background_test_data
        self.model.signal_data = signal_data
        self.model.signal_hist.append(float(np.nanmean(signal_loss)))

        # ------------------------------------------------------------------
        # 7. Final plots
        # ------------------------------------------------------------------
        plot_anomaly_score(
            self.model.background_test_loss,
            self.model.signal_loss,
            background_label="QCD (Test)",
            signal_label="WJet",
            save_path=os.path.join(self.TRAIN_PLOTS_PATH, "anomaly_score.png"),
        )

        plot_roc_curve(
            self.model.background_test_loss,
            self.model.signal_loss,
            background_label="QCD (Test)",
            signal_label="WJet",
            savepath=os.path.join(self.TRAIN_PLOTS_PATH, "roc-bgtest-vs-signal.png"),
            examples=False,
            loss_fn=torch.nn.MSELoss(reduction="mean"),
        )
        
        plot_roc_curve(
            self.model.background_train_loss,
            self.model.signal_loss,
            background_label="QCD (Train)",
            signal_label="WJet",
            savepath=os.path.join(self.TRAIN_PLOTS_PATH, "roc-bgtrain-vs-signal.png"),
            examples=False,
            loss_fn=torch.nn.MSELoss(reduction="mean"),
        )

        # ------------------------------------------------------------------
        # 8. Save losses and summary
        # ------------------------------------------------------------------
        np.save(
            os.path.join(self.TRAIN_PLOTS_PATH, "background_test_loss.npy"),
            np.asarray(self.model.background_test_loss),
        )

        np.save(
            os.path.join(self.TRAIN_PLOTS_PATH, "signal_loss.npy"),
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
            "batch_size": self.batch_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "train_mask_ratio": self.train_mask_ratio,
            "test_mask_ratio": self.test_mask_ratio,
            "eval_mask_repeats": self.eval_mask_repeats,
            "epochs": self.epochs,
            "learning_rate": self.initial_lr,
            "weight_decay": self.weight_decay,
            "lr_scheduler": self.lr_scheduler,
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

        summary_path = os.path.join(self.TRAIN_PLOTS_PATH, "summary.json")

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logging.info(f"Saved run summary to {summary_path}")
        logging.info(f"Final AUC: {auc_score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train Transformer Masked Encoder",
        description="trains the transformer masked encoder model on processed data"
    )
    parser.add_argument(
        "--background", "-b", type=str, default=bg_file,
        help="Path to processed .pkl background dataset (QCD). Defaults to background_file in config.yaml"
    )
    parser.add_argument(
        "--signal", "-s", type=str, default=sg_file,
        help="Path to processed .pkl signal dataset (WJet). Defaults to signal_file in config.yaml"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=config["model"]["hidden_dim"],
        help="Transformer hidden dimension. Defaults to config."
    )
    parser.add_argument(
        "--num-heads", type=int, default=4,
        help="Number of transformer attention heads. Default: 4."
    )
    parser.add_argument(
        "--ffn-dim", type=int, default=None,
        help="Transformer FFN hidden dimension. Default: 4 * hidden_dim."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1,
        help="Dropout used inside transformer blocks. Default: 0.1."
    )
    parser.add_argument(
        "--train-mask-ratio", type=float, default=0.3,
        help="Fraction of valid non-CLS tokens to mask during training. Default: 0.3."
    )
    parser.add_argument(
        "--test-mask-ratio", type=float, default=0.3,
        help="Fraction of valid non-CLS tokens to mask during validation and final evaluation. Default: 0.3."
    )
    parser.add_argument(
        "--eval-mask-repeats", type=int, default=10,
        help="Number of random masks to average for each event during final evaluation. Default: 10."
    )
    parser.add_argument(
        "--batch-size", type=int, default=config["model"]["batch_size"],
        help="Training batch size. Defaults to config."
    )
    parser.add_argument(
        "--epochs", type=int, default=config["training"]["epochs"],
        help="Number of training epochs. Defaults to config."
    )
    parser.add_argument(
        "--learning-rate", type=float, default=config["training"]["initial_lr"],
        help="Initial AdamW learning rate. Defaults to config."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4,
        help="AdamW weight decay. Default: 1e-4."
    )
    parser.add_argument(
        "--lr-scheduler", action=argparse.BooleanOptionalAction, default=True,
        help="Decay the learning rate by 30 percent every 10 epochs."
    )
    parser.add_argument(
        "--normalize-features", action=argparse.BooleanOptionalAction, default=True,
        help="Normalize graph features using background-training statistics."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for Python, NumPy, and PyTorch. Default: 42."
    )
    parser.add_argument(
        "--output-dir", default="plots/test-plots",
        help="Directory for plots, losses, and summary.json."
    )
    parser.add_argument(
        "--max-background-events", type=int,
        help="Optional background row limit for a smoke test."
    )
    parser.add_argument(
        "--max-signal-events", type=int,
        help="Optional signal row limit for a smoke test."
    )
    parser.add_argument(
        "--num-layers", type=int, default=4,
        help="Number of transformer encoder layers. Default: 4."
    )

    trainer = TrainMaskedTransformer()
    trainer.load()
    trainer.build_graphs()
    trainer.compute_stats()
    trainer.plot_features()
    trainer.train()
