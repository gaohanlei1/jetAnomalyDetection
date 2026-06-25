"""
Plot latent space for a trained JetGraphAutoencoder run.

Usage example:

python -u visualize/plot_latent_space.py \
    plots/run-8h-2s-3e \
    "QCD" \
    "WJet"

Or specify output path:

python -u visualize/plot_latent_space.py \
    plots/run-8h-2s-3e \
    "QCD" \
    "WJet" \
    --output plots/run-8h-2s-3e/latent_space.png
"""

import os
import sys
import json
import argparse
import random
from typing import List, Tuple

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from sklearn.decomposition import PCA

import constants as c
from helpers import helpers_main
from models.autoencoder import JetGraphAutoencoder
from models.cls_transformer_ae import JetClassTokenTransformerAE
from preprocess.make_graphs import graph_data_loader


config = helpers_main.load_config()

# These helper functions mirror the data processing logic in run_train_autoencoder.py 
# to ensure consistency
# Note: If you changed the data processing logic in training, make sure to update 
# these functions accordingly!

def remove_low_pt_muons(row):
    """
    Same helper logic as run_train_autoencoder.py.
    Only relevant for WminusH-style signal files.
    """
    pdgId = row["pdgId"]
    pt = row["pt"]
    mask = (np.abs(pdgId) != 13) | (pt >= 0.4)

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
    """
    Same normalization logic as run_train_autoencoder.py.
    """
    if mean is None or std is None:
        all_features = torch.cat([graph.x for graph in graphs], dim=0)
        mean = all_features.mean(dim=0)
        std = all_features.std(dim=0)
        std[std == 0] = 1.0

    for graph in graphs:
        graph.x = (graph.x - mean) / std

    return graphs, mean, std


def load_summary(run_dir: str) -> dict:
    summary_path = os.path.join(run_dir, "summary.json")

    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Could not find summary.json at {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    return summary


def load_data_from_summary(summary: dict):
    bg_file = summary["background"]
    sg_file = summary["signal"]

    bg_data = pd.read_pickle(bg_file)
    sg_data = pd.read_pickle(sg_file)

    max_background_events = summary.get("max_background_events", None)
    max_signal_events = summary.get("max_signal_events", None)

    if max_background_events is not None:
        bg_data = bg_data.head(max_background_events)

    if max_signal_events is not None:
        sg_data = sg_data.head(max_signal_events)

    pt_max = c.PT_MAX
    pt_min = c.PT_MIN
    rawfj_pt_col = c.RAW_FATJET_PROPERTIES_PREFIX + "pt"

    if rawfj_pt_col in bg_data:
        bg_data = bg_data[
            (bg_data[rawfj_pt_col] > pt_min)
            & (bg_data[rawfj_pt_col] < pt_max)
        ]

    if "WminusH" in sg_file:
        sg_data = sg_data.apply(remove_low_pt_muons, axis=1)

    return bg_data, sg_data


def build_graphs_from_summary(summary: dict, bg_data, sg_data):
    method = summary.get("method", "eta_phi")
    knn = summary.get("knn", 3)

    bg_graphs = graph_data_loader(
        bg_data,
        data_label=0,
        nearest_neighbors=knn,
        device="cpu",
        method=method,
        alpha=config["training"]["alpha"],
        node_feature_names=summary.get("feature_names", None)
    )

    sg_graphs = graph_data_loader(
        sg_data,
        data_label=1,
        nearest_neighbors=knn,
        device="cpu",
        method=method,
        alpha=config["training"]["alpha"],
        node_feature_names=summary.get("feature_names", None)
    )

    if len(bg_graphs) == 0:
        raise ValueError("No background graphs were created.")

    if len(sg_graphs) == 0:
        raise ValueError("No signal graphs were created.")

    if summary.get("normalize_features", False):
        train_split = 0.8
        train_size = int(train_split * len(bg_graphs))

        bg_train_graphs = bg_graphs[:train_size]
        bg_test_graphs = bg_graphs[train_size:]

        bg_train_graphs, bg_train_mean, bg_train_std = normalize_graph_features(
            bg_train_graphs
        )

        bg_test_graphs, _, _ = normalize_graph_features(
            bg_test_graphs,
            mean=bg_train_mean,
            std=bg_train_std,
        )

        sg_graphs, _, _ = normalize_graph_features(
            sg_graphs,
            mean=bg_train_mean,
            std=bg_train_std,
        )

        bg_graphs = bg_train_graphs + bg_test_graphs

    return bg_graphs, sg_graphs


def load_model_from_run(summary: dict, run_dir: str, device: torch.device):
    model_path = os.path.join(run_dir, "best_model.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find best_model.pth at {model_path}")

    # Construct the model from summary.json.
    # We infer num_features later from the loaded model if possible, but the
    # architecture still needs the same hidden_dim/smallest_dim settings.
    loaded_obj = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    if isinstance(loaded_obj, JetGraphAutoencoder) or isinstance(loaded_obj, JetClassTokenTransformerAE):
        model = loaded_obj.to(device)
        model.eval()
        return model

    if isinstance(loaded_obj, dict):
        raise TypeError(
            "best_model.pth appears to contain a dict/checkpoint, not a full "
            "JetGraphAutoencoder object. The current training code saves the "
            "full model with torch.save(self.model, path). If you changed that "
            "to state_dict saving, this loader needs num_features to construct "
            "the model before loading state_dict."
        )

    raise TypeError(f"Unsupported model object type: {type(loaded_obj)}")


def get_graph_level_latents(model, graphs, batch_size: int, device: torch.device):
    loader = DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=False,
    )

    latent_batches = []

    model.eval()

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            if isinstance(model, JetGraphAutoencoder):
                # This mirrors JetGraphAutoencoder.forward(), but stops after encoder.
                x = model.embedding(batch.x)
                z_nodes = model.encoder(
                    x,
                    batch.edge_index,
                    batch,
                    training=False,
                )

                # Convert node-level latent vectors into graph/event-level latent vectors.
                z_graph = global_mean_pool(z_nodes, batch.batch)
            elif isinstance(model, JetClassTokenTransformerAE):
                # This mirrors JetClassTokenTransformerAE.forward(), but stops after encoder.
                x, valid_mask = model._to_dense(batch)
                eta_phi = x[:, :, :2]

                particle_tokens = model.feature_embedding(x)
                particle_tokens = particle_tokens + model.position_embedding(eta_phi)

                cls_token = model.cls_embedding.expand(batch.num_graphs, -1, -1)
                encoder_input = torch.cat([cls_token, particle_tokens], dim=1)

                cls_valid_mask = torch.ones(
                    batch.num_graphs, 1, dtype=torch.bool, device=x.device
                )
                encoder_valid_mask = torch.cat([cls_valid_mask, valid_mask], dim=1)
                key_padding_mask = ~encoder_valid_mask

                encoded = model.encoder(
                    encoder_input,
                    src_key_padding_mask=key_padding_mask,
                )
                cls_hidden = encoded[:, 0]
                latent = model.to_latent(cls_hidden)

                # Convert node-level latent vectors into graph/event-level latent vectors.
                z_graph = latent
            else:
                raise TypeError(f"Unsupported model type: {type(model)}")

            latent_batches.append(z_graph.detach().cpu())

    latents = torch.cat(latent_batches, dim=0).numpy()
    return latents


def reduce_to_2d(bg_latents, sg_latents):
    all_latents = np.concatenate([bg_latents, sg_latents], axis=0)

    if all_latents.shape[1] == 2:
        return bg_latents, sg_latents, "Latent dim 1", "Latent dim 2"

    if all_latents.shape[1] < 2:
        raise ValueError(
            f"Latent dimension is {all_latents.shape[1]}, cannot plot 2D latent space."
        )

    pca = PCA(n_components=2)
    all_latents_2d = pca.fit_transform(all_latents)

    bg_count = len(bg_latents)
    bg_latents_2d = all_latents_2d[:bg_count]
    sg_latents_2d = all_latents_2d[bg_count:]

    x_label = f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)"
    y_label = f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)"

    return bg_latents_2d, sg_latents_2d, x_label, y_label


def plot_latent_space(
    bg_latents_2d,
    sg_latents_2d,
    background_label: str,
    signal_label: str,
    output_path: str,
    x_label: str = "Latent dim 1",
    y_label: str = "Latent dim 2",
):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(
        bg_latents_2d[:, 0],
        bg_latents_2d[:, 1],
        label=background_label,
        alpha=0.35,
        s=8,
    )

    ax.scatter(
        sg_latents_2d[:, 0],
        sg_latents_2d[:, 1],
        label=signal_label,
        alpha=0.35,
        s=8,
    )

    ax.set_xlabel("Latent dim 1")
    ax.set_ylabel("Latent dim 2")
    ax.set_title("Latent Space")
    range_lim_low = min(bg_latents_2d.min(), sg_latents_2d.min()) - 0.1
    range_lim_high = max(bg_latents_2d.max(), sg_latents_2d.max()) + 0.1
    ax.set_xlim(range_lim_low, range_lim_high)
    ax.set_ylim(range_lim_low, range_lim_high)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        prog="Plot Autoencoder Latent Space",
        description="Plot graph-level latent space from a trained JetGraphAutoencoder run.",
    )

    parser.add_argument(
        "model_run_dir",
        type=str,
        help="Model run directory, e.g. plots/run-8h-2s-3e.",
    )

    parser.add_argument(
        "background_name",
        type=str,
        help="Background dataset name for plotting, e.g. QCD.",
    )

    parser.add_argument(
        "signal_name",
        type=str,
        help="Signal dataset name for plotting, e.g. WJet.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for latent space plot. Defaults to model_run_dir/latent_space.png.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "mps", "cuda"],
        help="Device for latent extraction. Default: cpu.",
    )

    args = parser.parse_args()

    run_dir = args.model_run_dir

    if args.output is None:
        output_path = os.path.join(run_dir, "latent_space.png")
    else:
        output_path = args.output

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device("cuda")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    summary = load_summary(run_dir)

    seed = summary.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Loading datasets...")
    bg_data, sg_data = load_data_from_summary(summary)

    print("Building graphs...")
    bg_graphs, sg_graphs = build_graphs_from_summary(summary, bg_data, sg_data)

    print(f"Background graphs: {len(bg_graphs)}")
    print(f"Signal graphs: {len(sg_graphs)}")

    print("Loading model...")
    model = load_model_from_run(summary, run_dir, device)

    batch_size = summary.get("batch_size", 64)

    print("Extracting background latents...")
    bg_latents = get_graph_level_latents(
        model,
        bg_graphs,
        batch_size=batch_size,
        device=device,
    )

    print("Extracting signal latents...")
    sg_latents = get_graph_level_latents(
        model,
        sg_graphs,
        batch_size=batch_size,
        device=device,
    )

    bg_latents_2d, sg_latents_2d, x_label, y_label = reduce_to_2d(
        bg_latents,
        sg_latents,
    )

    print("Plotting latent space...")
    plot_latent_space(
        bg_latents_2d,
        sg_latents_2d,
        background_label=args.background_name,
        signal_label=args.signal_name,
        output_path=output_path,
        x_label=x_label,
        y_label=y_label,
    )
    print(f"Latent space plot saved to {output_path}")


if __name__ == "__main__":
    main()