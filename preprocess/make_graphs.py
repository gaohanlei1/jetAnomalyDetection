# ORIGINAL METHOD
# import torch
# import numpy as np
# from torch_geometric.data import Data
# from torch_cluster import knn_graph
# import pandas as pd
# from typing import List



# def make_graph(data: dict, data_label: int, node_feature_names=['pt', 'eta', 'phi', 'd0/d0Err', 'dz/dzErr'], 
#                nearest_neighbors=16, device='cpu', method='eta_phi'):
#     """
#     Build a graph from particle features, supporting different edge construction methods.
#     """

#     # Always safely convert to arrays (avoiding pandas Series issues)
#     x = torch.tensor(np.column_stack([np.array(data[name]) for name in node_feature_names]), dtype=torch.float).to(device)

#     if method == 'eta_phi':
#         nn_features = torch.tensor(np.column_stack([np.array(data['eta']), np.array(data['phi'])]), dtype=torch.float).to(device)
#         edge_index = knn_graph(nn_features, k=nearest_neighbors, loop=False).to(device)

#     elif method == 'all_features':
#         nn_features = x
#         edge_index = knn_graph(nn_features, k=nearest_neighbors, loop=False).to(device)

#     elif method == 'fully_connected':
#         num_nodes = x.shape[0]
#         row = torch.arange(num_nodes).repeat_interleave(num_nodes)
#         col = torch.arange(num_nodes).repeat(num_nodes)
#         edge_index = torch.stack([row, col], dim=0).to(device)

#     else:
#         raise ValueError(f"Unknown method: {method}")

#     y = torch.tensor([int(data_label)], dtype=torch.long).to(device)
    
#     return Data(x=x, edge_index=edge_index, y=y)


# def graph_data_loader(df: pd.DataFrame,
#                       data_label: int,
#                       nearest_neighbors: int = 16,
#                       device: str = 'cpu') -> List[Data]:
#     """
#     Convert a dataframe of events into a list of graphs.
#     """
#     graphs = []
#     for i in range(len(df['eta'].tolist())):
#         graphs.append(make_graph(df.iloc[i], data_label=data_label, nearest_neighbors=nearest_neighbors, device=device))
#     return graphs

# TRIAL 1
# import torch
# import numpy as np
# from torch_geometric.data import Data
# from torch_cluster import knn_graph
# import pandas as pd
# from sklearn.neighbors import NearestNeighbors
# from typing import List
# import yaml

# with open("configs/config.yaml", "r") as f:
#     config = yaml.safe_load(f)

# def custom_distance_matrix(data: dict, alpha=0, beta=1):
#     """
#     Compute a pairwise custom distance matrix combining:
#     - Euclidean distance in eta/phi
#     - Inverse invariant mass

#     Assumes input `data` is a dict with keys 'pt', 'eta', 'phi', 'mass' (mass can be 0 if unknown).
#     """
    

#     pt = np.array(data['pt'])
#     eta = np.array(data['eta'])
#     phi = np.array(data['phi'])
#     mass = np.array(data.get('mass', np.zeros_like(pt)))

#     px = pt * np.cos(phi)
#     py = pt * np.sin(phi)
#     pz = pt * np.sinh(eta)
#     E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

#     coords = np.column_stack([eta, phi])
#     euclidean = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)

#     N = len(pt)
#     inv_mass = np.zeros((N, N))

#     for i in range(N):
#         for j in range(N):
#             E_sum = E[i] + E[j]
#             px_sum = px[i] + px[j]
#             py_sum = py[i] + py[j]
#             pz_sum = pz[i] + pz[j]
#             mass_sq = E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2)
#             m_ij = np.sqrt(np.abs(mass_sq))  # prevent complex values
#             inv_mass[i, j] = 1.0 / (m_ij + 1e-6)

#     #return alpha * euclidean + beta * inv_mass
#     return inv_mass

# # def build_custom_edge_index(data: dict, k=16, device='cpu', alpha=0, beta=1):
# #     """
# #     Build kNN edge index from a custom distance matrix.
# #     """
# #     dist_mat = custom_distance_matrix(data, alpha=alpha, beta=beta)
# #     nbrs = NearestNeighbors(n_neighbors=k + 1, metric='precomputed').fit(dist_mat)
# #     knn_graph = nbrs.kneighbors_graph(dist_mat, mode='connectivity')
# #     edge_index_np = np.array(knn_graph.nonzero())
# #     edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)
# #     return edge_index

# def build_custom_edge_index(data: dict, k=16, device='cpu', alpha=0, beta=1):
#     dist_mat = custom_distance_matrix(data, alpha=alpha, beta=beta)
#     nbrs = NearestNeighbors(n_neighbors=k + 1, metric='precomputed').fit(dist_mat)
#     knn_graph = nbrs.kneighbors_graph(dist_mat, mode='connectivity')
#     edge_index_np = np.array(knn_graph.nonzero())

#     # Extract weights from the distance matrix using the indices
#     edge_weights = dist_mat[edge_index_np[0], edge_index_np[1]]

#     edge_index = torch.tensor(edge_index_np, dtype=torch.long).to(device)
#     edge_weights = torch.tensor(edge_weights, dtype=torch.float).to(device)

#     return edge_index, edge_weights




# # def make_graph(data: dict, data_label: int,
# #                node_feature_names=config['misc']['node_feature_names'],
# #                nearest_neighbors=16, device='cpu', method='eta_phi', alpha=0, beta=1):
# #     """
# #     Build a graph from particle features, supporting various edge construction methods.
# #     """

# #     x = torch.tensor(np.column_stack([np.array(data[name]) for name in node_feature_names]), dtype=torch.float).to(device)
# #     # Build a mask to filter NaNs only when necessary

# #     k = min(nearest_neighbors, x.shape[0] - 1)

# #     if method == 'eta_phi':
# #         nn_features = torch.tensor(np.column_stack([np.array(data['eta']), np.array(data['phi'])]), dtype=torch.float).to(device)
# #         edge_index = knn_graph(nn_features, k=nearest_neighbors, loop=False).to(device)

# #     elif method == 'all_features':
# #         nn_features = x
# #         edge_index = knn_graph(nn_features, k=nearest_neighbors, loop=False).to(device)

# #     elif method == 'fully_connected':
# #         num_nodes = x.shape[0]
# #         row = torch.arange(num_nodes).repeat_interleave(num_nodes)
# #         col = torch.arange(num_nodes).repeat(num_nodes)
# #         edge_index = torch.stack([row, col], dim=0).to(device)

# #     elif method == 'custom_metric':
# #         edge_index = build_custom_edge_index(data, k=k, device=device, alpha=alpha, beta=beta)

# #     else:
# #         raise ValueError(f"Unknown method: {method}")

# #     y = torch.tensor([int(data_label)], dtype=torch.long).to(device)

# #     return Data(x=x, edge_index=edge_index, y=y)

# def make_graph(data: dict, data_label: int,
#                node_feature_names=config['misc']['node_feature_names'],
#                nearest_neighbors=16, device='cpu', method='eta_phi', alpha=0, beta=1):
#     x = torch.tensor(np.column_stack([np.array(data[name]) for name in node_feature_names]), dtype=torch.float).to(device)
#     k = min(nearest_neighbors, x.shape[0] - 1)

#     if method == 'eta_phi':
#         nn_features = torch.tensor(np.column_stack([np.array(data['eta']), np.array(data['phi'])]), dtype=torch.float).to(device)
#         edge_index = knn_graph(nn_features, k=nearest_neighbors, loop=False).to(device)
#         edge_attr = None

#     elif method == 'all_features':
#         nn_features = x
#         edge_index = knn_graph(nn_features, k=nearest_neighbors, loop=False).to(device)
#         edge_attr = None

#     elif method == 'fully_connected':
#         num_nodes = x.shape[0]
#         row = torch.arange(num_nodes).repeat_interleave(num_nodes)
#         col = torch.arange(num_nodes).repeat(num_nodes)
#         edge_index = torch.stack([row, col], dim=0).to(device)
#         edge_attr = None

#     elif method == 'custom_metric':
#         edge_index, edge_attr = build_custom_edge_index(data, k=k, device=device, alpha=alpha, beta=beta)

#     else:
#         raise ValueError(f"Unknown method: {method}")

#     y = torch.tensor([int(data_label)], dtype=torch.long).to(device)

#     return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# # ORIGINAL METHOD
# def graph_data_loader(df: pd.DataFrame,
#                       data_label: int,
#                       nearest_neighbors: int = 16,
#                       device: str = 'cpu',
#                       method: str = 'eta_phi') -> List[Data]:
#     """
#     Convert a dataframe of events into a list of graphs.
#     """
#     graphs = []
#     for i in range(len(df)):
#         try:
#             graphs.append(make_graph(df.iloc[i], data_label=data_label,
#                                     nearest_neighbors=nearest_neighbors,
#                                     device=device,
#                                     method=method))
#         except Exception as e:
#             print(f"Skipping event {i} due to error: {e}")
#             continue
#     return graphs

import torch
import numpy as np
import pandas as pd
from typing import List
from torch_geometric.data import Data
from torch_cluster import knn_graph
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


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


def make_graph(data: dict,
               data_label: int,
               node_feature_names=['pt', 'eta', 'phi', 'd0/d0Err', 'dz/dzErr'],
               nearest_neighbors=16,
               device='cpu',
               method='eta_phi') -> Data:
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
                      node_feature_names=['pt', 'eta', 'phi', 'd0/d0Err', 'dz/dzErr']) -> List[Data]:
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
                               method=method)
            graphs.append(graph)
        except Exception as e:
            print(f"Skipping event {i} due to error: {e}")
            continue
    return graphs