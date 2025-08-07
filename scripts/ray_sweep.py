# """
# Ray Tune version of JetGraphAutoencoder hyperparameter search.
# """
# import os
# import sys
# import random
# import numpy as np
# import pandas as pd
# import yaml
# import torch
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score, roc_curve

# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
# from ray.air import session
# from ray.tune import Tuner, TuneConfig, RunConfig
# from torch_geometric.loader import DataLoader

# # === Add project root to path ===
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from models.autoencoder import JetGraphAutoencoder
# from train.utils_training import train_loop, eval_loop
# from preprocess.make_graphs import graph_data_loader

# # === Set seeds for reproducibility ===
# torch.manual_seed(42)
# np.random.seed(42)
# random.seed(42)

# # === Load configuration ===
# with open("configs/config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# train_file = config['data']['processed_data_dir'] + config['data']['train_file']
# test_file = config['data']['processed_data_dir'] + config['data']['test_file']
# epochs = config['training']['epochs']
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def train_autoencoder_ray(config_ray):
#     # === Load data ===
#     datatype1 = pd.read_pickle(train_file)
#     datatype2 = pd.read_pickle(test_file)

#     graphs_train = graph_data_loader(datatype1, data_label=0, nearest_neighbors=config_ray['nearest_neighbors'])
#     graphs_test = graph_data_loader(datatype2, data_label=1, nearest_neighbors=config_ray['nearest_neighbors'])

#     train_size = int(0.8 * len(graphs_train))
#     train_graphs = graphs_train[:train_size]
#     val_graphs = graphs_train[train_size:]
#     signal_graphs = graphs_test

#     train_loader = DataLoader(train_graphs, batch_size=config_ray['batch_size'], shuffle=True)
#     val_loader = DataLoader(val_graphs, batch_size=config_ray['batch_size'])
#     signal_loader = DataLoader(signal_graphs, batch_size=config_ray['batch_size'])

#     model = JetGraphAutoencoder(
#         num_features=train_graphs[0].x.shape[1],
#         smallest_dim=config_ray['smallest_dim'],
#         num_reduced_edges=config_ray['nearest_neighbors']
#     ).to(device)

#     optimizer = torch.optim.AdamW(
#         model.parameters(), 
#         lr=float(config_ray['lr']), 
#         weight_decay=float(config_ray['weight_decay'])
#     )
#     loss_fn = torch.nn.MSELoss()

#     best_val_loss = float("inf")

#     for epoch in range(config_ray['epochs']):
#         train_loss = train_loop(train_loader, model, loss_fn, optimizer)
#         val_losses = eval_loop(val_loader, model, loss_fn)
#         val_loss = np.nanmean(val_losses)

#         signal_losses = eval_loop(signal_loader, model, loss_fn)
#         labels = np.array([0] * len(val_losses) + [1] * len(signal_losses))
#         scores = np.array(val_losses + signal_losses)
#         auc = roc_auc_score(labels, -scores)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             checkpoint_path = os.path.join(session.get_trial_dir(), "best_model.pt")
#             torch.save(model.state_dict(), checkpoint_path)

#             # === Plot ROC ===
#             fpr, tpr, _ = roc_curve(labels, -scores)
#             plt.figure()
#             plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
#             plt.plot([0, 1], [0, 1], 'k--', label="Random")
#             plt.xlabel("False Positive Rate")
#             plt.ylabel("True Positive Rate")
#             plt.title("ROC Curve")
#             plt.legend()
#             plt.grid()
#             plt.tight_layout()
#             plt.savefig(os.path.join(session.get_trial_dir(), "roc_curve.png"))
#             plt.close()

#         tune.report({"val_loss": val_loss, "train_loss": train_loss, "auc": auc, "epoch": epoch})

# # === Define search space ===
# search_space = {
#     'lr': tune.grid_search(config['sweep']['lr']),
#     'weight_decay': tune.grid_search(config['sweep']['weight_decay']),
#     'nearest_neighbors': tune.grid_search(config['sweep']['k_nearest_neighbors']),
#     'smallest_dim': tune.grid_search(config['sweep']['smallest_dim']),
#     'batch_size': tune.grid_search([64]) if isinstance(config['model']['batch_size'], int)
#                   else tune.grid_search(config['model']['batch_size']),
#     'epochs': epochs,
# }

# scheduler = ASHAScheduler(grace_period=5)

# trainable = tune.with_resources(train_autoencoder_ray, {"cpu": 16, "gpu": 1 if device.type == "cuda" else 0})

# # === Run Ray Tune ===
# tuner = Tuner(
#     trainable,
#     param_space=search_space,
#     tune_config=TuneConfig(
#         scheduler=scheduler,
#         metric="val_loss",
#         mode="min",

#         # choose number of concurrent trials if desired
#         # max_concurrent_trials=2
#     ),
#     run_config=RunConfig(
#         name="autoencoder_ray_sweep",
#         storage_path="sweeps/ray_results",
#     )
# )

# results = tuner.fit()

# # === Save and display results ===
# df = results.get_dataframe()
# df.to_csv("sweeps/ray_results/autoencoder_ray_sweep.csv")

# logging.info("Best config (by val_loss):", results.get_best_result(metric="val_loss", mode="min").config)
# logging.info("Best config (by AUC):", results.get_best_result(metric="auc", mode="max").config)

"""
Ray Tune version of JetGraphAutoencoder hyperparameter search (AUC-optimized).
"""
import os
import sys
import random
import numpy as np
import pandas as pd
import yaml
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import logging

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from ray.tune import Tuner, TuneConfig, RunConfig
from torch_geometric.loader import DataLoader

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import constants as c
from helpers import helpers_main

from models.autoencoder import JetGraphAutoencoder
from train.utils_training import train_loop, eval_loop
from preprocess.make_graphs import graph_data_loader

# === Set seeds for reproducibility ===
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# === Load configuration ===
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

BASE_DIR = "/home/anagaman/jet-anomaly-summer25/jetAnomalyDetection_updated/jetAnomalyDetection"

train_file = os.path.join(BASE_DIR, config['data']['processed_data_dir'], config['data']['background_file'])
test_file = os.path.join(BASE_DIR, ['data']['processed_data_dir'], config['data']['signal_file'])

device = helpers_main.get_device()

def train_autoencoder_ray(config_ray):
    # === Load data ===
    datatype1 = pd.read_pickle(train_file)
    datatype2 = pd.read_pickle(test_file)

    graphs_train = graph_data_loader(
        datatype1, 
        data_label=0, 
        nearest_neighbors=config_ray['nearest_neighbors'],
        method=hybrid_knn,
        alpha=config_ray['alpha'])

    graphs_test = graph_data_loader(
        datatype2, 
        data_label=1, 
        nearest_neighbors=config_ray['nearest_neighbors'],
        alpha=config_ray['alpha'])

    train_size = int(0.8 * len(graphs_train))
    train_graphs = graphs_train[:train_size]
    val_graphs = graphs_train[train_size:]
    signal_graphs = graphs_test

    train_loader = DataLoader(train_graphs, batch_size=config_ray['batch_size'], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=config_ray['batch_size'])
    signal_loader = DataLoader(signal_graphs, batch_size=config_ray['batch_size'])

    model = JetGraphAutoencoder(
        num_features=train_graphs[0].x.shape[1],
        smallest_dim=config_ray['smallest_dim'],
        num_reduced_edges=config_ray['nearest_neighbors']
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=float(config_ray['lr']), 
        weight_decay=float(config_ray['weight_decay'])
    )
    loss_fn = torch.nn.MSELoss()

    best_auc = -float("inf")

    for epoch in range(config_ray['epochs']):
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

            # === Plot ROC ===
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

        # Report all metrics
        tune.report({
            "val_loss": val_loss,
            "train_loss": train_loss,
            "signal_loss": signal_loss,
            "auc": auc,
            "epoch": epoch
        })

# === Define search space ===
# search_space = {
#     'lr': tune.grid_search(config['sweep']['lr']),
#     'weight_decay': tune.grid_search(config['sweep']['weight_decay']),
#     'nearest_neighbors': tune.grid_search(config['sweep']['k_nearest_neighbors']),
#     'smallest_dim': tune.grid_search(config['sweep']['smallest_dim']),
#     'batch_size': tune.grid_search([64]) if isinstance(config['model']['batch_size'], int)
#                   else tune.grid_search(config['model']['batch_size']),
#     'epochs': 20  # allow ASHA to stop early
# }
search_space = {
    'lr': 0.0005,
    'weight_decay': 1e-04,
    'nearest_neighbors': [8, 16, 32],
    'smallest_dim': 8,
    'batch_size': 64,
    'epochs': 20,
    'alpha': [0, 0.25, 0.5, 0.75, 1]
}

scheduler = ASHAScheduler(
    max_t=20,      # total number of epochs to train
    grace_period=5 # let trials run for at least 5 epochs before stopping
)

trainable = tune.with_resources(train_autoencoder_ray, {"cpu": 16, "gpu": 1 if device.type == "cuda" else 0})

# === Run Ray Tune ===
tuner = Tuner(
    trainable,
    param_space=search_space,
    tune_config=TuneConfig(
        scheduler=scheduler,
        metric="auc",
        mode="max",
    ),
    run_config=RunConfig(
        name="autoencoder_auc_sweep",
        storage_path="/home/anagaman/jet-anomaly-summer25/jetAnomalyDetection_updated/jetAnomalyDetection/sweeps/ray_results",
    )
)

results = tuner.fit()

# === Save and display results ===
df = results.get_dataframe()
df.to_csv("sweeps/ray_results/autoencoder_auc_sweep.csv")

logging.info("Best config (by AUC):", results.get_best_result(metric="auc", mode="max").config)
logging.info("Best config (by val_loss):", results.get_best_result(metric="val_loss", mode="min").config)
