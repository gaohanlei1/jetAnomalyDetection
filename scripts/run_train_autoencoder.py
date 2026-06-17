"""
Script to train a graph-based autoencoder for jet anomaly detection.

This script:
- Loads configuration and preprocessed datasets.
- Constructs graph representations of jet events.
- Trains the JetGraphAutoencoder model on background data.
- Evaluates the model on background and signal samples.
- Plots anomaly scores, ROC curves, and training loss histories.
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
from models.autoencoder import JetGraphAutoencoder
from train.utils_training import train_loop, eval_loop, train_model, normalize_graph_features
from preprocess.make_graphs import graph_data_loader
from visualize.plot_metrics import plot_loss, plot_anomaly_score, plot_roc_curve
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from typing import List, Tuple
from sklearn.metrics import roc_auc_score

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


class TrainAutoencoder:
    # Packaged into a class for variable management
    TRAIN_SPLIT = 0.8
    FEATURE_PLOTS_PATH = "plots/test-plots/features"
    TRAIN_PLOTS_PATH   = "plots/test-plots"
    
    def __init__(self):
        self.args = parser.parse_args()
        self.bg_file, self.sg_file = self.args.background, self.args.signal
        self.bg_name, self.sg_name = helpers_main.trim_name(self.bg_file), helpers_main.trim_name(self.sg_file)
        self.method = self.args.method
        self.knn = self.args.knn
        self.smallest_dim = self.args.smallest_dim
        self.num_reduced_edges = self.args.num_reduced_edges
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

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        
        self.session_name = f"logs/train_ae_{self.bg_name}_{self.sg_name}_{self.method}_{helpers_main.curr_time()}.log"
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
            self.bg_data = self.bg_data[(self.bg_data[rawfj_pt_col] > pt_min) & (self.bg_data[rawfj_pt_col] < pt_max)]
        # logging.info(f"Signal Data Columns: {self.sg_data.columns.tolist()}")

        # Only for WminusH - This removes the leptonic jet
        if "WminusH" in self.sg_file: self.sg_data = self.sg_data.apply(remove_low_pt_muons, axis=1)

        logging.info(f"Number of training events after slicing: {len(self.bg_data)}")
        logging.info(f"Number of test events after removing leptonic jet: {len(self.sg_data)}")
        logging.info(f"\nSample background pt values:\n{self.bg_data['pt'].head().to_string()}")
        logging.info(f"Sample signal pt values:\n{self.sg_data['pt'].head().to_string()}")
    
    def build_graphs(self):
        # Keep the complete graph collection in CPU memory. Training moves one
        # batch at a time to the selected accelerator.
        self.bg_graphs = graph_data_loader(
            self.bg_data, data_label=0, nearest_neighbors=self.knn, device="cpu", method=self.method, alpha=config['training']['alpha']
        )
        self.sg_graphs = graph_data_loader(
            self.sg_data, data_label=1, nearest_neighbors=self.knn, device="cpu", method=self.method, alpha=config['training']['alpha']
        )
        logging.info(f"Number of background graphs: {len(self.bg_graphs)}")
        logging.info(f"Number of signal graphs: {len(self.sg_graphs)}")

        # Split background dataset into training and test portions
        train_size = int(self.TRAIN_SPLIT * len(self.bg_graphs))
        self.bg_train_graphs = self.bg_graphs[:train_size]
        self.bg_test_graphs  = self.bg_graphs[train_size:]
        # self.sg_graphs = self.sg_graphs

        if self.normalize_features:
            self.bg_train_graphs, self.bg_train_mean, self.bg_train_std = normalize_graph_features(
                self.bg_train_graphs
            )
            self.bg_test_graphs, _, _ = normalize_graph_features(
                self.bg_test_graphs, mean=self.bg_train_mean, std=self.bg_train_std
            )
            self.sg_graphs, _, _ = normalize_graph_features(
                self.sg_graphs, mean=self.bg_train_mean, std=self.bg_train_std
            )

    def compute_stats(self):
        self.all_features = torch.cat([graph.x for graph in self.bg_train_graphs], dim=0)
        self.num_features = self.all_features.shape[1]
        self.feature_names = config["misc"]["node_feature_names"]

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
        os.makedirs(self.TRAIN_PLOTS_PATH, exist_ok=True)

        # Execute the training routine
        self.model = run_autoencoder_training(
            self.bg_train_graphs, self.bg_test_graphs, self.sg_graphs,
            smallest_dim=self.smallest_dim,
            num_reduced_edges=self.num_reduced_edges,
            batch_size=self.batch_size,
            epochs=self.epochs,
            initial_lr=self.initial_lr,
            weight_decay=self.weight_decay,
            lr_scheduler=self.lr_scheduler,
            save_dir=self.TRAIN_PLOTS_PATH,
            run_metadata={
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
                "lr_scheduler": self.lr_scheduler,
                "normalize_features": self.normalize_features,
                "seed": self.seed,
                "device": DEVICE,
                "max_background_events": self.max_background_events,
                "max_signal_events": self.max_signal_events,
            },
        )

    # def plot_loss(self):
    #     # Plot per-graph reconstruction loss distribution
    #     plt.figure(figsize=(8, 5))
    #     plt.hist(self.model.background_test_loss, bins=50, alpha=0.6, label='Background (QCD)', color='blue', density=True)
    #     plt.hist(self.model.signal_loss, bins=50, alpha=0.6, label='Signal', color='red', density=True)
    #     plt.xlabel("Per-Graph Reconstruction Loss")
    #     plt.ylabel("Density")
    #     plt.title("Reconstruction Loss Distribution")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.tight_layout()

    #     # Save plot
    #     plt.savefig(os.path.join(
    #         self.TRAIN_PLOTS_PATH, f"loss_{self.bg_name}_{self.sg_name}_{helpers_main.curr_time()}.png"
    #     ))
    #     if config["dbg"]["show_plots"]: plt.show()
    #     plt.clf()


def run_autoencoder_training(
    train_graphs, test_graphs, signal_graphs, smallest_dim,
    num_reduced_edges, batch_size, epochs, initial_lr, weight_decay,
    lr_scheduler, save_dir="plots/test-plots", run_metadata=None
):
    """
    Trains the JetGraphAutoencoder and evaluates it on background and signal graphs.

    Args:
        train_graphs (List[Data]): List of training graphs (background only).
        test_graphs (List[Data]): List of testing graphs (background only).
        signal_graphs (List[Data]): List of testing graphs (signal events).
        smallest_dim (int): Latent bottleneck dimensionality in the autoencoder.
        num_reduced_edges (int): Number of nearest neighbors to use in the kNN graph.
        batch_size (int): Batch size used during training.
        epochs (int): Number of training epochs.
        initial_lr (float): Initial learning rate for the optimizer.

    Returns:
        model (JetGraphAutoencoder): Trained model.
    """

    model = JetGraphAutoencoder(
        num_features=train_graphs[0].x.shape[1],
        smallest_dim=smallest_dim,
        num_reduced_edges=num_reduced_edges
    ).to(DEVICE)
    
    # print model summary
    logging.info(f"Model Summary:\n{model}")
    # number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of trainable parameters: {num_params}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=initial_lr, weight_decay=weight_decay
    )
    scheduler = (
        StepLR(optimizer, step_size=10, gamma=0.7)
        if lr_scheduler
        else None
    )

    loss_fn = torch.nn.MSELoss()

    # Dataloaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)
    signal_loader = DataLoader(signal_graphs, batch_size=1, shuffle=False)

    # Train the model and track loss
    train_loss, val_loss, signal_loss = train_model(
        train_loader, test_loader, signal_loader,
        model, loss_fn, optimizer,
        epochs=epochs, batch_size=batch_size, 
        scheduler=scheduler
    )

    os.makedirs(save_dir, exist_ok=True)
    # Generate plots for analysis
    plot_anomaly_score(
        model.background_test_loss,
        model.signal_loss,
        background_label="QCD",
        signal_label="WJet",
        save_path=os.path.join(save_dir, "anomaly_score.png"),
    )
    plot_roc_curve(
        model,
        "signal",
        "background",
        savepath=os.path.join(save_dir, "roc.png"),
        examples=False,
        loss_fn=torch.nn.MSELoss(reduction='mean'),
    )
    plot_loss(
        model.train_hist,
        model.val_hist,
        save_path=os.path.join(save_dir, "loss.png"),
    )

    np.save(
        os.path.join(save_dir, "background_test_loss.npy"),
        np.asarray(model.background_test_loss),
    )
    np.save(
        os.path.join(save_dir, "signal_loss.npy"),
        np.asarray(model.signal_loss),
    )
    auc_score = roc_auc_score(
        np.concatenate([
            np.zeros(len(model.background_test_loss)),
            np.ones(len(model.signal_loss)),
        ]),
        np.concatenate([model.background_test_loss, model.signal_loss]),
    )
    summary = dict(run_metadata or {})
    summary.update({
        "auc": float(auc_score),
        "background_train_graphs": len(train_graphs),
        "background_test_graphs": len(test_graphs),
        "signal_graphs": len(signal_graphs),
    })
    with open(os.path.join(save_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Saved run summary to {os.path.join(save_dir, 'summary.json')}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train Autoencoder",
        description="trains the autoencoder model on processed data"
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
        "--method", "-m", choices=c.GRAPH_METHODS, default="eta_phi",
        help=f"Method for building graph edges. Default: eta_phi"
    )
    parser.add_argument(
        "--knn", "-n", type=int, default=config["misc"]["k_nearest_neighbors"],
        help=f"Nearest neighbours count. Defaults to config"
    )
    parser.add_argument(
        "--smallest-dim", type=int, default=config["model"]["smallest_dim"],
        help="Latent bottleneck dimension. Defaults to config."
    )
    parser.add_argument(
        "--num-reduced-edges", type=int, default=config["model"]["num_reduced_edges"],
        help="Decoder kNN edge count. Defaults to config."
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

    args = parser.parse_args()
    train_ae = TrainAutoencoder()
    train_ae.load()
    train_ae.build_graphs()
    train_ae.compute_stats()
    train_ae.plot_features()
    train_ae.train()
    # train_ae.plot_loss()
