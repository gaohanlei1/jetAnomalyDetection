import torch
import numpy as np
import pandas as pd
from typing import List
from torch_geometric.data import Data
from torch_cluster import knn_graph
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import logging

def sanitize_features(pt, eta, phi, mass):
    """Clip or adjust physical features to avoid overflows or invalid values."""
    pt = np.nan_to_num(pt, nan=0.0, posinf=0.0, neginf=0.0)
    eta = np.nan_to_num(eta, nan=0.0, posinf=0.0, neginf=0.0)
    phi = np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0)
    mass = np.nan_to_num(mass, nan=0.0, posinf=0.0, neginf=0.0)
    return pt, eta, phi, mass


def build_mass_knn_edges(pt, eta, phi, mass, k, device='cpu'):
    """Build kNN edge index using 1 / invariant mass as a distance."""
    # Compute 4-momenta
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

    N = len(pt)
    dist = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            E_sum = E[i] + E[j]
            px_sum = px[i] + px[j]
            py_sum = py[i] + py[j]
            pz_sum = pz[i] + pz[j]

            mass_sq = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
            m = np.sqrt(np.abs(mass_sq))  # safe square root

            dist[i, j] = 1.0 / (m + 1e-6)  # inverse mass as distance

    # Build kNN graph from custom distance matrix
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='precomputed')
    nbrs.fit(dist)
    knn_graph_matrix = nbrs.kneighbors_graph(dist, mode='connectivity')
    edge_index_np = np.array(knn_graph_matrix.nonzero())
    edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)
    return edge_index

def build_hybrid_knn_edges_vectorized(pt, eta, phi, mass, k, alpha=0.5, device='cpu'):
    """Vectorized: Build kNN edges using hybrid of ΔR and 1/invariant mass."""

    # 4-momentum components
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    pz = pt * np.sinh(eta)
    E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

    # ΔR
    eta_diff = eta[:, None] - eta[None, :]
    phi_diff = np.abs(phi[:, None] - phi[None, :])
    phi_diff = np.where(phi_diff > np.pi, 2 * np.pi - phi_diff, phi_diff)
    delta_r = np.sqrt(eta_diff**2 + phi_diff**2)

    # Invariant mass pairwise distance (vectorized)
    E_sum = E[:, None] + E[None, :]
    px_sum = px[:, None] + px[None, :]
    py_sum = py[:, None] + py[None, :]
    pz_sum = pz[:, None] + pz[None, :]
    mass_sq = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
    m = np.sqrt(np.clip(mass_sq, a_min=0, a_max=None))
    inv_mass_dist = 1.0 / (m + 1e-6)

    # Normalize distances
    norm_dR = (delta_r - delta_r.min()) / (delta_r.max() - delta_r.min() + 1e-6)
    norm_mass = (inv_mass_dist - inv_mass_dist.min()) / (inv_mass_dist.max() - inv_mass_dist.min() + 1e-6)

    # Hybrid metric
    hybrid_dist = alpha * norm_dR + (1 - alpha) * norm_mass

    # kNN graph from hybrid distance
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric='precomputed')
    nbrs.fit(hybrid_dist)
    knn_graph_matrix = nbrs.kneighbors_graph(hybrid_dist, mode='connectivity')
    edge_index_np = np.array(knn_graph_matrix.nonzero())
    edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)

    return edge_index

def make_graph(data: dict,
               data_label: int,
               node_feature_names=['pt', 'eta', 'phi', 'd0/d0Err', 'dz/dzErr'],
               nearest_neighbors=16,
               device='cpu',
               method='eta_phi',
               alpha: float = 0.5) -> Data:
    """
    Build a graph from particle features using specified edge method.
    """

    try:
        pt = np.array(data['pt'])
        eta = np.array(data['eta'])
        phi = np.array(data['phi'])
        mass = np.array(data.get('mass', np.zeros_like(pt)))

        pt, eta, phi, mass = sanitize_features(pt, eta, phi, mass)

        # Mask only based on required edge features, not all node features
        valid_mask = ~np.isnan(pt) & ~np.isnan(eta) & ~np.isnan(phi) & ~np.isnan(mass)

        if valid_mask.sum() < 2:
            raise ValueError("Too few valid particles to build a graph.")

        # Clean core features used for edge index
        pt = pt[valid_mask]
        eta = eta[valid_mask]
        phi = phi[valid_mask]
        mass = mass[valid_mask]

        # Clean all node features
        clean_data = {}
        for name in node_feature_names:
            if name in data:
                values = np.array(data[name])
                if len(values) != len(valid_mask):
                    raise ValueError(f"Length mismatch for {name}")
                clean_data[name] = values[valid_mask]
            else:
                raise ValueError(f"Missing node feature: {name}")
        
        x = torch.tensor(np.column_stack([clean_data[name] for name in node_feature_names]), dtype=torch.float).to(device)
        k = min(nearest_neighbors, x.shape[0] - 1)

        if method == 'eta_phi':
            nn_features = torch.tensor(np.column_stack([eta, phi]), dtype=torch.float).to(device)
            edge_index = knn_graph(nn_features, k=k, loop=False).to(device)

        elif method == 'all_features':
            edge_index = knn_graph(x, k=k, loop=False).to(device)

        elif method == 'fully_connected':
            num_nodes = x.shape[0]
            row = torch.arange(num_nodes).repeat_interleave(num_nodes)
            col = torch.arange(num_nodes).repeat(num_nodes)
            edge_index = torch.stack([row, col], dim=0).to(device)

        elif method == 'mass_knn':
            edge_index = build_mass_knn_edges(pt, eta, phi, mass, k, device=device)
        
        elif method == 'hybrid_knn':
            edge_index = build_hybrid_knn_edges_vectorized(pt, eta, phi, mass, k, alpha=alpha, device=device)

        else:
            raise ValueError(f"Unknown method: {method}")

        y = torch.tensor([int(data_label)], dtype=torch.long).to(device)
        return Data(x=x, edge_index=edge_index, y=y)

    except Exception as e:
        raise RuntimeError(f"Graph construction failed: {e}")

def graph_data_loader(df: pd.DataFrame,
                      data_label: int,
                      nearest_neighbors: int = 16,
                      device: str = 'cpu',
                      method: str = 'eta_phi',
                      node_feature_names=['pt', 'eta', 'phi', 'd0/d0Err', 'dz/dzErr'],
                      alpha: float = 0.5) -> List[Data]:
    """
    Convert a dataframe of events into a list of PyTorch Geometric Data objects (graphs).
    """
    graphs = []
    # add tqdm here if desired
    for i in tqdm(range(len(df)), desc="Building graphs"):
        try:
            graph = make_graph(data=df.iloc[i],
                               data_label=data_label,
                               node_feature_names=node_feature_names,
                               nearest_neighbors=nearest_neighbors,
                               device=device,
                               method=method,
                               alpha=alpha)
            graphs.append(graph)
        except Exception as e:
            logging.info(f"Skipping event {i} due to error: {e}")
            continue
    return graphs