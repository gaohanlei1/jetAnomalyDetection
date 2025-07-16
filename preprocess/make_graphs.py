import torch
import numpy as np
from torch_geometric.data import Data
from torch_cluster import knn_graph
import pandas as pd
from typing import List



def make_graph(data: dict, data_label: int, node_feature_names=['pt', 'eta', 'phi', 'd0/d0Err', 'dz/dzErr'], 
               nearest_neighbors=16, device='cpu', method='eta_phi'):
    """
    Build a graph from particle features, supporting different edge construction methods.
    """

    # Always safely convert to arrays (avoiding pandas Series issues)
    x = torch.tensor(np.column_stack([np.array(data[name]) for name in node_feature_names]), dtype=torch.float).to(device)

    if method == 'eta_phi':
        nn_features = torch.tensor(np.column_stack([np.array(data['eta']), np.array(data['phi'])]), dtype=torch.float).to(device)
        edge_index = knn_graph(nn_features, k=nearest_neighbors, loop=False).to(device)

    elif method == 'all_features':
        nn_features = x
        edge_index = knn_graph(nn_features, k=nearest_neighbors, loop=False).to(device)

    elif method == 'fully_connected':
        num_nodes = x.shape[0]
        row = torch.arange(num_nodes).repeat_interleave(num_nodes)
        col = torch.arange(num_nodes).repeat(num_nodes)
        edge_index = torch.stack([row, col], dim=0).to(device)

    else:
        raise ValueError(f"Unknown method: {method}")

    y = torch.tensor([int(data_label)], dtype=torch.long).to(device)
    
    return Data(x=x, edge_index=edge_index, y=y)


def graph_data_loader(df: pd.DataFrame,
                      data_label: int,
                      nearest_neighbors: int = 16,
                      device: str = 'cpu') -> List[Data]:
    """
    Convert a dataframe of events into a list of graphs.
    """
    graphs = []
    for i in range(len(df['eta'].tolist())):
        graphs.append(make_graph(df.iloc[i], data_label=data_label, nearest_neighbors=nearest_neighbors, device=device))
    return graphs