"""
Training utility functions for JetGraphAutoencoder models.

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
from helpers import helpers_main
config = helpers_main.load_config()

import logging
helpers_main.log_config(f"logs/train_{helpers_main.curr_time()}.log")

def train_loop(dataloader, model, loss_fn, optimizer):
    """
    Executes one epoch of training.

    Args:
        dataloader (DataLoader): Dataloader for training graphs.
        model (torch.nn.Module): The autoencoder model.
        loss_fn (callable): Loss function (e.g., MSE).
        optimizer (torch.optim.Optimizer): Optimizer instance.

    Returns:
        float: Mean training loss over the epoch.
    """
    model.train()
    total_loss = []

    for batch, X in enumerate(dataloader):
        optimizer.zero_grad()
        pred = model(X)
        pred = pred[:, :X.x.shape[1]]  # Only reconstruct original feature dimensions

        loss = loss_fn(pred, X.x)

        total_loss.append(float(loss))

        loss.backward()
        optimizer.step()

    return np.nanmean(total_loss)


def eval_loop(dataloader, model, loss_fn, test=False, signal=False):
    """
    Evaluates the model on a given dataset.

    Args:
        dataloader (DataLoader): Loader containing the evaluation set.
        model (torch.nn.Module): Trained model to evaluate.
        loss_fn (callable): Loss function to use.
        test (bool): If True, stores losses as `background_test_loss`.
        signal (bool): If True, stores losses as `signal_loss`.

    Returns:
        List[float]: List of per-graph losses.
    """
    model.eval()
    loss = []
    data = []

    with torch.no_grad():
        for X in dataloader:
            pred = model(X)
            pred = pred[:, :X.x.shape[1]]
            loss.append(float(loss_fn(pred, X.x)))
            data.append(X.x)

    # Store for downstream use if flagged
    if test:
        model.test_data = data
        model.background_test_loss = loss
    elif signal:
        model.signal_data = data
        model.signal_loss = loss

    return loss

# def normalize_graph_features(
#     graphs: List[Data]
# ) -> Tuple[List[Data], torch.Tensor, torch.Tensor]:
#     """
#     Normalize features across all graphs using the global mean and std.

#     Args:
#         graphs (List[Data]): List of PyG graphs with node features.

#     Returns:
#         Tuple containing:
#         - List[Data]: Normalized graphs
#         - torch.Tensor: Feature mean
#         - torch.Tensor: Feature std
#     """
#     # Stack all node features to compute global stats
#     all_features = torch.cat([graph.x for graph in graphs], dim=0)
#     mean = all_features.mean(dim=0)
#     std = all_features.std(dim=0)

#     # Avoid divide-by-zero
#     std[std == 0] = 1.0

#     # Normalize each graph
#     for graph in graphs:
#         graph.x = (graph.x - mean) / std

#     return graphs, mean, std

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


def train_model(train_dataloader, test_dataloader, signal_dataloader, model, loss_fn, optimizer, epochs, batch_size, scheduler=None):
    """
    Trains the model across multiple epochs and tracks performance on validation and signal sets.

    Args:
        train_dataloader (DataLoader): Training data loader.
        test_dataloader (DataLoader): Background validation set.
        signal_dataloader (DataLoader): Signal evaluation set.
        model (torch.nn.Module): JetGraphAutoencoder model.
        loss_fn (callable): Loss function (e.g., MSE).
        optimizer (torch.optim.Optimizer): Optimizer instance.
        epochs (int): Number of training epochs.
        batch_size (int): Size of batches used in training.

    Returns:
        Tuple[List[float], List[float], List[float]]:
            train_loss: Mean training losses per epoch.
            val_loss: Mean validation losses per epoch.
            signal_loss: Mean signal losses per epoch.
    """
    model.train_hist = []
    model.val_hist = []
    model.signal_hist = []

    # start timer
    helpers_main.secs_since_last_ping()

    for epoch in range(epochs):
        logging.info(f"\nEpoch [{epoch+1}/{epochs}]")

        train_loss = []
        val_loss = []
        signal_loss = []

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Learning Rate: {current_lr:.6f}")

        # Run training step
        # train_loop(train_dataloader, model, loss_fn, optimizer)

        # # Evaluate on validation and signal
        # val_loss.extend(eval_loop(test_dataloader, model, loss_fn, test=True, signal=False))
        # train_loss.extend(eval_loop(train_dataloader, model, loss_fn, test=False, signal=False))
        # signal_loss.extend(eval_loop(signal_dataloader, model, loss_fn, test=False, signal=True))
        # Run training and capture true training loss
        mean_train_loss = train_loop(train_dataloader, model, loss_fn, optimizer)

        # Evaluate on validation and signal sets
        val_loss.extend(eval_loop(test_dataloader, model, loss_fn, test=True, signal=False))
        signal_loss.extend(eval_loop(signal_dataloader, model, loss_fn, test=False, signal=True))

        # Append epoch results
        # model.train_hist.append(np.nanmean(train_loss))
        # model.val_hist.append(np.nanmean(val_loss))
        # model.signal_hist.append(np.nanmean(signal_loss))
        
        model.train_hist.append(mean_train_loss)
        model.val_hist.append(np.nanmean(val_loss))
        model.signal_hist.append(np.nanmean(signal_loss))


        if scheduler is not None:
            scheduler.step()

        logging.info(f"train loss: {mean_train_loss}")
        logging.info(f"test loss: {np.nanmean(val_loss)}")
        logging.info(f"signal loss: {np.nanmean(signal_loss)}")
        logging.info(f"(took {helpers_main.secs_since_last_ping()} seconds)")

    # Final assignment for later ROC/score plotting
    model.background_test_loss = val_loss
    model.background_train_loss = train_loss
    model.signal_loss = signal_loss

    #return train_loss, val_loss, eval_loop(signal_dataloader, model, loss_fn, test=False, signal=True)
    return model.train_hist, model.val_hist, model.signal_hist
