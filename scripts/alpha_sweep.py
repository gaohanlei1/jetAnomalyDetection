import os
import sys
import random
import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.tune import Tuner, TuneConfig, RunConfig
from torch_geometric.loader import DataLoader

# Add your project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.autoencoder import JetGraphAutoencoder
from train.utils_training import train_loop, eval_loop
from preprocess.make_graphs import graph_data_loader

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import constants as c
from helpers import helpers_main
config = helpers_main.load_config()

train_file = os.path.join(config['data']['processed_data_dir'], config['data']['background_file'])
test_file = os.path.join(config['data']['processed_data_dir'], config['data']['signal_file'])

device = helpers_main.get_device()

def train_autoencoder_alpha(config_ray):
    alpha = config_ray['alpha']

    # Load data (you may want to cache this outside if very large)
    train_df = pd.read_pickle(train_file)
    test_df = pd.read_pickle(test_file)

    # Here you MUST pass alpha to your graph building method if it uses it
    graphs_train = graph_data_loader(train_df, data_label=0, nearest_neighbors=config['misc']['k_nearest_neighbors'], alpha=alpha)
    graphs_test = graph_data_loader(test_df, data_label=1, nearest_neighbors=config['misc']['k_nearest_neighbors'], alpha=alpha)

    train_size = int(0.8 * len(graphs_train))
    train_graphs = graphs_train[:train_size]
    val_graphs = graphs_train[train_size:]
    signal_graphs = graphs_test

    train_loader = DataLoader(train_graphs, batch_size=config['model']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config['model']['batch_size'])
    signal_loader = DataLoader(signal_graphs, batch_size=config['model']['batch_size'])

    model = JetGraphAutoencoder(
        num_features=train_graphs[0].x.shape[1],
        smallest_dim=config['model']['smallest_dim'],
        num_reduced_edges=config['misc']['k_nearest_neighbors'] # <-- pass alpha here if your model uses it
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['initial_lr'],
        weight_decay=1e-04
    )
    loss_fn = torch.nn.MSELoss()

    best_auc = -float("inf")

    for epoch in range(config['training']['epochs']):
        train_loss = train_loop(train_loader, model, loss_fn, optimizer)
        val_losses = eval_loop(val_loader, model, loss_fn)
        val_loss = np.nanmean(val_losses)

        signal_losses = eval_loop(signal_loader, model, loss_fn)
        signal_loss = np.nanmean(signal_losses)

        labels = np.array([0] * len(val_losses) + [1] * len(signal_losses))
        scores = np.array(val_losses + signal_losses)
        auc = roc_auc_score(labels, -scores)

        if auc > best_auc:
            best_auc = auc
            checkpoint_path = os.path.join(session.get_trial_dir(), "best_model.pt")
            torch.save(model.state_dict(), checkpoint_path)

            # Optional: save ROC curve plot
            fpr, tpr, _ = roc_curve(labels, -scores)
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            plt.plot([0, 1], [0, 1], 'k--', label="Random")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(session.get_trial_dir(), "roc_curve.png"))
            plt.close()

        # tune.report(val_loss=val_loss, train_loss=train_loss, signal_loss=signal_loss, auc=auc, epoch=epoch)
        session.report({
            "val_loss": val_loss,
            "train_loss": train_loss,
            "signal_loss": signal_loss,
            "auc": auc,
            "epoch": epoch,
        })


# Define the search space: only sweep alpha here
search_space = {
    'alpha': tune.grid_search([0.0, 0.25, 0.5, 0.75, 1.0]),
}

scheduler = ASHAScheduler(
    max_t=config['training']['epochs'],
    grace_period=3,
    reduction_factor=2
)

storage_abs_path = "/home/anagaman/jet-anomaly-summer25/jetAnomalyDetection_updated/jetAnomalyDetection/sweeps"
os.makedirs(storage_abs_path, exist_ok=True)

trainable = tune.with_resources(train_autoencoder_alpha, {"cpu": 16, "gpu": 1 if device.type == "cuda" else 0})

tuner = Tuner(
    trainable,
    param_space=search_space,
    tune_config=TuneConfig(
        scheduler=scheduler,
        metric="auc",
        mode="max"
    ),
    run_config=RunConfig(
        name="alpha_sweep",
        storage_path=f"file://{storage_abs_path}",
    )
)

results = tuner.fit()

df = results.get_dataframe()
df.to_csv(os.path.join(storage_abs_path, "alpha_sweep_results.csv"))

print("Best alpha config (val loss):", results.get_best_result(metric="val_loss", mode="min").config)
print("Best alpha config (AUC):", results.get_best_result(metric="auc", mode="max").config)
