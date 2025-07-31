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

# Add the parent directory to Python's path to allow local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

import constants as c
from helpers import helpers_main
from helpers import join_dfs
config = helpers_main.load_config()
import logging

# File paths for background and signal data
bg_file = config['data']['processed_data_dir'] + config['data']['background_file']
sg_file = config['data']['processed_data_dir'] + config['data']['signal_file']

class TrainAutoencoder:
    # Packaged into a class for variable management
    TRAIN_SPLIT = 0.8
    FEATURE_PLOTS_PATH = "plots/test-plots/features"
    TRAIN_PLOTS_PATH   = "plots/test-plots"
    
    def __init__(self):
        args = parser.parse_args()
        self.bg_file, self.sg_file = args.background, args.signal
        self.bg_name, self.sg_name = helpers_main.trim_name(self.bg_file), helpers_main.trim_name(self.sg_file)
        self.method = args.method
        self.knn = args.knn
        
        self.session_name = f"logs/train_ae_{self.bg_name}_{self.sg_name}_{self.method}_{helpers_main.curr_time()}.log"
        helpers_main.log_config(self.session_name)
    
    def load(self):
        # Load datasets from pickle files
        self.bg_data = pd.read_pickle(self.bg_file)
        self.sg_data = pd.read_pickle(self.sg_file)

        logging.info(f"Number of training events: {len(self.bg_data)}")
        logging.info(f"Number of test events: {len(self.sg_data)}")
        logging.info(f"\nSample background pt values:\n{self.bg_data['pt'].head()}")
        logging.info(f"Sample signal pt values:\n{self.sg_data['pt'].head()}")
    
    def build_graphs(self):
        # Convert datasets to PyG graph objects
        self.bg_graphs = graph_data_loader(
            self.bg_data, data_label=0, nearest_neighbors=self.knn, device='cpu', method=self.method
        )
        self.sg_graphs = graph_data_loader(
            self.sg_data, data_label=1, nearest_neighbors=self.knn, device='cpu', method=self.method
        )
        logging.info(f"Number of background graphs: {len(self.bg_graphs)}")
        logging.info(f"Number of signal graphs: {len(self.sg_graphs)}")

        # Split background dataset into training and test portions
        train_size = int(self.TRAIN_SPLIT * len(self.bg_graphs))
        self.bg_train_graphs = self.bg_graphs[:train_size]
        self.bg_test_graphs  = self.bg_graphs[train_size:]
        # self.sg_graphs = self.sg_graphs

        # Normalize features
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
        logging.info("Feature Means:", self.means)
        logging.info("Feature Stds:", self.stds)
        logging.info("Number of features:", self.num_features)
    
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
            smallest_dim=config['model']['smallest_dim'],
            num_reduced_edges=config['model']['num_reduced_edges'],
            batch_size=config['model']['batch_size'],
            epochs=config['training']['epochs'],
            initial_lr=config['training']['initial_lr']
        )

    def plot_loss(self):
        # Plot per-graph reconstruction loss distribution
        plt.figure(figsize=(8, 5))
        plt.hist(model.background_test_loss, bins=50, alpha=0.6, label='Background (QCD)', color='blue', density=True)
        plt.hist(model.signal_loss, bins=50, alpha=0.6, label='Signal', color='red', density=True)
        plt.xlabel("Per-Graph Reconstruction Loss")
        plt.ylabel("Density")
        plt.title("Reconstruction Loss Distribution")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        plt.savefig(os.path.join(
            self.TRAIN_PLOTS_PATH, f"loss_{self.bg_name}_{self.sg_name}_{helpers_main.curr_time()}.png"
        ))
        plt.show()
        plt.clf()


def run_autoencoder_training(train_graphs, test_graphs, signal_graphs, smallest_dim, num_reduced_edges, batch_size, epochs, initial_lr):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = JetGraphAutoencoder(
        num_features=train_graphs[0].x.shape[1],
        smallest_dim=smallest_dim,
        num_reduced_edges=num_reduced_edges
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.7)  # Decay LR by 50% every 10 epochs

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

    # Generate plots for analysis
    plot_anomaly_score(model.background_test_loss, model.signal_loss, background_label="", signal_label="")
    plot_roc_curve(model, "signal", "background", savepath=save_dir + 'roc.png', examples=False, loss_fn=torch.nn.MSELoss(reduction='mean'))
    plot_loss(model.train_hist, model.val_hist, save_path='plots/test-plots/loss.png')

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
        "--method", "-m", choices=c.GRAPH_METHODS, default="mass_knn",
        help=f"Method for building graph edges. Default: mass_knn"
    )
    parser.add_argument(
        "--knn", "-n", type=int, default=config["misc"]["k_nearest_neighbors"],
        help=f"Nearest neighbours count. Defaults to config"
    )

    train_ae = TrainAutoencoder()
    train_ae.load()
    train_ae.build_graphs()
    train_ae.compute_stats()
    train_ae.plot_features()
    train_ae.train()
    train_ae.plot_loss()
