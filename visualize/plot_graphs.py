"""
Graph visualization utility for inspecting kNN graphs of jet particle constituents.

This module provides:
- `visualize_graph`: A function that renders a 2D layout of particle graphs 
  using η (eta) and ϕ (phi) coordinates as spatial positions.
"""

import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# Add the parent directory to Python's path to allow local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from helpers import helpers_main
config = helpers_main.load_config()

def visualize_graph(data: Data, title: str, k: int = 16):
    """
    Visualizes a jet event as a k-nearest neighbor (kNN) graph using η and ϕ as coordinates.

    Nodes represent particle flow candidates (PFCands), and edges indicate graph connectivity
    defined by a kNN algorithm in η–ϕ space or feature space.

    Args:
        data (Data): PyTorch Geometric `Data` object representing a single graph.
                     Must contain `x` (node features) and `edge_index` (connectivity).
                     The second and third columns of `x` are assumed to be [η, ϕ].
        title (str): Title to display above the plot.
        k (int): The k-value used to construct the kNN graph (used for display only).

    Returns:
        None. Displays the graph using matplotlib.
    """
    # Convert edge index to NumPy for NetworkX compatibility
    edge_index = data.edge_index.cpu().numpy()

    # Use η (x[:,1]) and ϕ (x[:,2]) as 2D node positions
    pos = {i: [data.x[i, 1].item(), data.x[i, 2].item()] for i in range(data.num_nodes)}

    # Create undirected graph and add nodes/edges
    G = nx.Graph()
    for i in range(data.num_nodes):
        G.add_node(i)
    for start, end in edge_index.T:
        G.add_edge(int(start), int(end))

    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='k')
    nx.draw_networkx_edges(G, pos, edge_color='r', width=1)
    plt.title(f'Graph Visualization: {title} (k={k})')
    plt.axis('off')
    plt.tight_layout()
    if config["dbg"]["show_plots"]: plt.show()
    plt.clf()
