"""
Script to perform a grid search over JetGraphAutoencoder hyperparameters.

This script:
- Loads processed training and test datasets.
- Builds k-NN graphs with different neighborhood sizes.
- Trains the JetGraphAutoencoder for all combinations of:
    - Learning rate
    - Weight decay
    - k-NN edges
    - Latent dimension size
- Records training/validation/signal loss curves.
- Saves all metrics and model outputs for downstream visualization and analysis.
"""

import sys
import os
import numpy as np
import itertools
import torch
import pandas as pd
import yaml
from torch_geometric.loader import DataLoader
from models.autoencoder import JetGraphAutoencoder
from train.utils_training import train_loop, eval_loop
from preprocess.make_graphs import graph_data_loader

# Add parent directory to Python path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load experiment configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Dataset paths
train_file = config['data']['processed_data_dir'] + config['data']['train_file']
test_file = config['data']['processed_data_dir'] + config['data']['test_file']

# Load preprocessed data
datatype1 = pd.read_pickle(train_file)
datatype2 = pd.read_pickle(test_file)

# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cpu")

# Hyperparameter grids for sweeping
learning_rates = [1e-5, 5e-5, 1e-4]
weight_decays = [0.0, 1e-4, 1e-3]
nearest_neighbors = [8, 16, 32]
smallest_dims = [8, 16]
batch_size = config['model']['batch_size']
epochs = config['training']['epochs']

# Collect results from all runs
results = []

# Ensure output directory exists
if not os.path.exists('sweeps'):
    os.makedirs('sweeps')

# Iterate over all hyperparameter combinations
for idx, (lr, wd, k_nn, s_dim) in enumerate(itertools.product(learning_rates, weight_decays, nearest_neighbors, smallest_dims)):
    print(f"Training with lr={lr}, weight_decay={wd}, k_nn={k_nn}, smallest_dim={s_dim}")

    # Rebuild graphs with updated k-NN parameter
    graphs_train = graph_data_loader(datatype1, data_label=0, nearest_neighbors=k_nn)
    graphs_test = graph_data_loader(datatype2, data_label=1, nearest_neighbors=k_nn)

    # Train/validation split (80/20)
    train_size = int(0.8 * len(graphs_train))
    train_graphs = graphs_train[:train_size]
    val_graphs = graphs_train[train_size:]
    signal_graphs = graphs_test

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size)
    signal_loader = DataLoader(signal_graphs, batch_size=batch_size)

    # Initialize model
    model = JetGraphAutoencoder(
        num_features=train_graphs[0].x.shape[1],
        smallest_dim=s_dim,
        num_reduced_edges=k_nn
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = torch.nn.MSELoss()

    model.train_hist = []
    model.val_hist = []
    model.signal_hist = []

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        epoch_train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        val_losses = eval_loop(val_loader, model, loss_fn)
        signal_losses = eval_loop(signal_loader, model, loss_fn)

        epoch_val_loss = np.nanmean(val_losses)
        epoch_signal_loss = np.nanmean(signal_losses)

        model.train_hist.append(epoch_train_loss)
        model.val_hist.append(epoch_val_loss)
        model.signal_hist.append(epoch_signal_loss)

        # Track best validation loss for model selection
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss

        # Store per-sample losses
        model.background_test_loss = val_losses
        model.signal_loss = signal_losses

    # Record result for this run
    run_results = {
        'learning_rate': lr,
        'weight_decay': wd,
        'nearest_neighbors': k_nn,
        'smallest_dim': s_dim,
        'train_loss_hist': model.train_hist,
        'val_loss_hist': model.val_hist,
        'signal_loss_hist': model.signal_hist,
        'best_val_loss': best_val_loss
    }
    results.append(run_results)

    # Save per-epoch losses
    run_df = pd.DataFrame({
        'epoch': list(range(1, epochs + 1)),
        'train_loss': model.train_hist,
        'val_loss': model.val_hist,
        'signal_loss': model.signal_hist
    })
    run_df.to_csv(f'sweeps/run_{idx}_losses.csv', index=False)

    # Save hyperparameters
    params_df = pd.DataFrame([{
        'learning_rate': lr,
        'weight_decay': wd,
        'nearest_neighbors': k_nn,
        'smallest_dim': s_dim
    }])
    params_df.to_csv(f'sweeps/run_{idx}_params.csv', index=False)

    # Save per-sample loss arrays for visualization
    np.save(f'sweeps/run_{idx}_background_test_loss.npy', np.array(model.background_test_loss))
    np.save(f'sweeps/run_{idx}_signal_loss.npy', np.array(model.signal_loss))

# Save all results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('sweeps/autoencoder_param_sweep.csv', index=False)
