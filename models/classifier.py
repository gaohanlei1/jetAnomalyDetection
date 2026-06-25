import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import EdgeConv, TopKPooling, global_max_pool
from torch_cluster import knn_graph


class JetGraphAutoencoderClassification(nn.Module):
    """
    Graph-based autoencoder model for binary classification on jet data.

    This model uses EdgeConv layers from PyTorch Geometric to learn spatially-aware 
    representations of jet constituents, then aggregates these features and feeds 
    them through fully connected layers to predict the probability of signal presence.

    Attributes:
        num_features (int): Number of input features per node.
        smallest_dim (int): Latent dimensionality of the graph embeddings.
        topk (TopKPooling or None): Optional graph pooling layer.
        num_reduced_edges (int): Number of neighbors in kNN graph used for message passing.
    """

    def __init__(self, num_features: int = 3, smallest_dim: int = 8, topk=None, num_reduced_edges: int = 8):
        """
        Initialize the classification autoencoder model.

        Args:
            num_features (int): Dimensionality of input node features.
            smallest_dim (int): Dimensionality of the latent layers.
            topk (TopKPooling, optional): Pooling layer for downsampling the graph.
            num_reduced_edges (int): Number of edges in dynamically computed kNN graph.
        """
        super(JetGraphAutoencoderClassification, self).__init__()
        self.num_features = num_features
        self.smallest_dim = smallest_dim
        self.topk = topk
        self.num_reduced_edges = num_reduced_edges

        # Encoder: learn spatially-aware embeddings
        self.conv1 = EdgeConv(Sequential(
            Linear(2 * self.num_features, self.smallest_dim * 2),
            ReLU(),
            Linear(self.smallest_dim * 2, self.smallest_dim * 2)
        ), aggr='max')

        self.conv2 = EdgeConv(Sequential(
            Linear(self.smallest_dim * 2 * 2, self.smallest_dim),
            ReLU(),
            Linear(self.smallest_dim, self.smallest_dim)
        ), aggr='max')

        # Decoder: transform embeddings to global graph features
        self.conv3 = EdgeConv(Sequential(
            Linear(self.smallest_dim * 2, self.smallest_dim),
            ReLU(),
            Linear(self.smallest_dim, self.smallest_dim)
        ), aggr='max')

        self.conv4 = EdgeConv(Sequential(
            Linear(self.smallest_dim * 2, self.smallest_dim * 2),
            ReLU(),
            Linear(self.smallest_dim * 2, self.smallest_dim * 2)
        ), aggr='max')

        # Final prediction head
        self.fc3 = Linear(self.smallest_dim * 2, 2 * self.num_features)
        self.out = Linear(2 * self.num_features, 1)

    def batched_knn_graph(self, x, batch, k):
        """
        Recompute kNN edges inside each graph, clamping k for small jets.

        torch_cluster.knn_graph accepts one k value for the whole batch. Some
        jets have fewer than k + 1 valid particles, so building each graph
        separately avoids both cross-jet edges and too-large k values.
        """
        if batch is None:
            if x.size(0) < 2:
                return torch.empty((2, 0), dtype=torch.long, device=x.device)
            return knn_graph(x, k=min(k, x.size(0) - 1), loop=False).to(x.device)

        edge_indices = []
        for graph_id in batch.unique(sorted=True):
            node_indices = (batch == graph_id).nonzero(as_tuple=False).view(-1)
            if node_indices.numel() < 2:
                continue

            local_x = x[node_indices]
            local_k = min(k, local_x.size(0) - 1)
            local_edge_index = knn_graph(
                local_x,
                k=local_k,
                loop=False,
            ).to(x.device)
            edge_indices.append(node_indices[local_edge_index])

        if not edge_indices:
            return torch.empty((2, 0), dtype=torch.long, device=x.device)

        return torch.cat(edge_indices, dim=1)

    def encoder(self, x, edge_index, data):
        """
        Encode node features into a latent graph representation.

        Args:
            x (Tensor): Node feature matrix [num_nodes, num_features].
            edge_index (Tensor): Graph connectivity matrix.
            data (Data): PyG data object (passed for future compatibility).

        Returns:
            Tensor: Encoded node features.
        """
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x

    def decoder(self, x, edge_index, batch):
        """
        Decode latent node embeddings into a classification prediction.

        Args:
            x (Tensor): Encoded node features.
            edge_index (Tensor): Edge index used for decoding.
            batch (Tensor): Batch vector mapping nodes to graphs.

        Returns:
            Tensor: Sigmoid score for each graph (shape: [batch_size, 1]).
        """
        x = F.relu(self.conv3(x, edge_index))
        x = F.relu(self.conv4(x, edge_index))
        x = global_max_pool(x, batch)  # Global graph-level representation
        x = F.relu(self.fc3(x))
        x = self.out(x)
        return x

    def forward(self, data, knn=True, topk=False):
        """
        Forward pass of the classification model.

        Args:
            data (Data): PyG Data object containing node features and edge indices.
            knn (bool): If True, recomputes edge_index with kNN after encoding.
            topk (bool): If True, applies TopKPooling after encoding.

        Returns:
            Tensor: Sigmoid output prediction for each input graph.
        """
        x, edge_index = data.x, data.edge_index
        x = self.encoder(x, edge_index, data)

        if knn:
            edge_index = self.batched_knn_graph(
                x,
                batch=data.batch,
                k=self.num_reduced_edges,
            )

        if topk:
            x, edge_index, _, batch, _, _ = self.topk(x, edge_index.to(torch.int64), batch=data.batch)
        else:
            batch = data.batch

        x = self.decoder(x, edge_index, batch)
        return torch.sigmoid(x)
