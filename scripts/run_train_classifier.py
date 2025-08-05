"""
Script to train a graph-based autoencoder classifier for binary signal/background discrimination.

This script:
- Loads and parses configuration settings.
- Loads preprocessed particle physics datasets.
- Constructs graph representations of jet events using k-nearest neighbors.
- Trains a JetGraphAutoencoderClassification model using binary cross-entropy loss.
- Logs accuracy and loss metrics and saves model weights after each epoch.
"""

import sys
import os

# Add the parent directory to Python's path for local module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
import yaml
from torch_geometric.loader import DataLoader
from models.classifier import JetGraphAutoencoderClassification
from preprocess.make_graphs import graph_data_loader
from sklearn.metrics import accuracy_score
import logging

# Load configuration settings from YAML file
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# File paths for training and test data
train_file = config['data']['processed_data_dir'] + config['data']['train_file']
test_file = config['data']['processed_data_dir'] + config['data']['test_file']

# Load preprocessed datasets
datatype1 = pd.read_pickle(train_file)
datatype2 = pd.read_pickle(test_file)

# Convert to PyTorch Geometric Data objects (graphs)
datatype1_graphs = graph_data_loader(datatype1, data_label=0, nearest_neighbors=config['misc']['k_nearest_neighbors'])
datatype2_graphs = graph_data_loader(datatype2, data_label=1, nearest_neighbors=config['misc']['k_nearest_neighbors'])

# Combine background and signal graphs into one training set
graphs = datatype1_graphs + datatype2_graphs


def run_classifier_training(graphs, save_dir='checkpoints', smallest_dim=16, num_reduced_edges=16, batch_size=10, epochs=20, initial_lr=1e-3):
    """
    Train the JetGraphAutoencoderClassification model on a set of background and signal graphs.

    Args:
        graphs (List[Data]): List of PyG Data objects (each one a jet graph).
        save_dir (str): Directory to save model weights and metrics.
        smallest_dim (int): Latent dimension for GNN layers.
        num_reduced_edges (int): Number of neighbors to use in dynamic kNN graphs.
        batch_size (int): Number of graphs per training batch.
        epochs (int): Number of training epochs.
        initial_lr (float): Learning rate for optimizer.

    Returns:
        model (nn.Module): Trained classifier model.
        losses (List[float]): Loss values per epoch.
        accuracies (List[float]): Accuracy values per epoch.
    """
    device = torch.device(config["training"]["device"])

    model = JetGraphAutoencoderClassification(
        num_features=graphs[0].x.shape[1],
        smallest_dim=smallest_dim,
        num_reduced_edges=num_reduced_edges
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr)
    criterion = torch.nn.BCELoss()
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    losses, accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_loss, all_preds, all_labels = 0, [], []

        for data in loader:
            optimizer.zero_grad()
            output = model(data).squeeze()
            truth = data.y.float().to(device)
            loss = criterion(output, truth)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.num_graphs
            preds = (output >= 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(truth.cpu().numpy())

        avg_loss = total_loss / len(graphs)
        acc = accuracy_score(all_labels, all_preds)
        losses.append(avg_loss)
        accuracies.append(acc)

        torch.save(model.state_dict(), f"{save_dir}/classifier_epoch_{epoch+1}.pt")
        logging.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    # Save training metrics to CSV
    metrics_df = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'loss': losses,
        'accuracy': accuracies
    })
    metrics_df.to_csv(f"{save_dir}/classifier_metrics.csv", index=False)

    return model, losses, accuracies


# Execute training using parameters from config
model, losses, accuracies = run_classifier_training(
    graphs,
    smallest_dim=config['model']['smallest_dim'],
    num_reduced_edges=config['model']['num_reduced_edges'],
    batch_size=config['model']['batch_size'],
    epochs=config['training']['epochs'],
    initial_lr=config['training']['initial_lr']
)
