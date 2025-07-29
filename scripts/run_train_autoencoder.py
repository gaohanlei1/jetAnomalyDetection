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
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from models.autoencoder import JetGraphAutoencoder
from train.utils_training import train_loop, eval_loop, train_model, normalize_graph_features
from preprocess.make_graphs import graph_data_loader
from visualize.plot_metrics import plot_loss, plot_anomaly_score, plot_roc_curve
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from typing import List, Tuple

# Load YAML configuration for data and hyperparameters
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# File paths for training and signal data
train_file = config['data']['processed_data_dir'] + config['data']['train_file']
test_file = config['data']['processed_data_dir'] + config['data']['test_file']

print(train_file)
# Load datasets from pickle files
datatype1 = pd.read_pickle(train_file)
datatype2 = pd.read_pickle(test_file)

print(f"Number of training events: {len(datatype1)}")
print("Sample background pt values:", datatype1['pt'].head())
print("Sample signal pt values:", datatype2['pt'].head())


# Convert datasets to PyG graph objects
datatype1_graphs = graph_data_loader(datatype1, data_label=0, nearest_neighbors=config['misc']['k_nearest_neighbors'], device='cpu', method='mass_knn')
datatype2_graphs = graph_data_loader(datatype2, data_label=1, nearest_neighbors=config['misc']['k_nearest_neighbors'], device='cpu', method='mass_knn')

print(f"Number of training graphs: {len(datatype1_graphs)}")

# Split background dataset into training and test portions
train_size = int(0.8 * len(datatype1_graphs))
train_graphs = datatype1_graphs[:train_size]
test_graphs = datatype1_graphs[train_size:]
signal_graphs = datatype2_graphs

# Normalize features
train_graphs, mean, std = normalize_graph_features(train_graphs)
test_graphs, _, _ = normalize_graph_features(test_graphs, mean=mean, std=std)
signal_graphs, _, _ = normalize_graph_features(signal_graphs, mean=mean, std=std)

os.makedirs("plots/test-plots/features", exist_ok=True)
all_features = torch.cat([graph.x for graph in train_graphs], dim=0)

# Compute mean and std per feature dimension
means = all_features.mean(dim=0)
stds = all_features.std(dim=0)

print("Feature Means:", means)
print("Feature Stds:", stds)

num_features = all_features.shape[1]

# Plot each feature's distribution
feature_names = config['misc']['node_feature_names']

for i in range(num_features):
    plt.figure()
    plt.hist(all_features[:, i].cpu().numpy(), bins=50, density=True, color='skyblue', edgecolor='black')
    plt.title(f"Feature {i}: {feature_names[i] if i < len(feature_names) else f'Feature {i}'}")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    safe_name = feature_names[i].replace('/', '_') if i < len(feature_names) else str(i)
    plt.savefig(f"plots/test-plots/features/feature_{i+1}_{safe_name}.png")
    plt.close()

# Ensure output directory exists
save_dir = 'plots/test-plots/'
os.makedirs(save_dir, exist_ok=True)

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

# Execute the training routine
model = run_autoencoder_training(
    train_graphs, test_graphs, signal_graphs,
    smallest_dim=config['model']['smallest_dim'],
    num_reduced_edges=config['model']['num_reduced_edges'],
    batch_size=config['model']['batch_size'],
    epochs=config['training']['epochs'],
    initial_lr=config['training']['initial_lr']
)

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
plt.savefig(os.path.join(save_dir, "loss_distribution.png"))
plt.show()
