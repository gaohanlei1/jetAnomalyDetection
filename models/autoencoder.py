import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LeakyReLU, Sequential, Linear, ReLU
from torch_geometric.nn import EdgeConv, TopKPooling, global_max_pool
from torch_cluster import knn_graph


class JetGraphAutoencoder(nn.Module):
    """
    Graph-based Autoencoder designed for jet anomaly detection using graph neural networks.
    
    This model uses EdgeConv layers to learn node representations based on k-nearest neighbor
    (kNN) graphs constructed from jet constituent features. It supports optional graph coarsening 
    via TopKPooling or dynamic recomputation of edges during decoding using `knn_graph`.
    
    Attributes:
        num_features (int): Number of input features per node.
        smallest_dim (int): Dimensionality of the bottleneck (latent space).
        topk (TopKPooling or None): Optional TopKPooling layer for pooling nodes.
        num_reduced_edges (int): Number of edges to use in the recomputed kNN graph.
        Various histories and losses for training analysis and evaluation.
    """

    def __init__(self, num_features: int = 3, smallest_dim: int = 16, hidden_dim: int = 32, num_layers: int = 2, topk=None, num_reduced_edges: int = 8):
        """
        Initialize the JetGraphAutoencoder.

        Args:
            num_features (int): Number of features per node (input dimensionality).
            smallest_dim (int): Latent dimensionality at the bottleneck layer.
            hidden_dim (int): Dimensionality of hidden layers.
            num_layers (int): Number of EdgeConv layers.
            topk (TopKPooling, optional): Optional pooling layer to downsample nodes.
            num_reduced_edges (int): Number of neighbors in recomputed edge graph (used during decoding if knn=True).
        """
        super(JetGraphAutoencoder, self).__init__()
        self.num_features = num_features
        self.smallest_dim = smallest_dim
        self.topk = topk
        self.num_reduced_edges = num_reduced_edges
        self.num_layers = num_layers

        # embeddng layer
        self.hidden_dim = hidden_dim
        self.embedding = Linear(num_features, self.hidden_dim)
        self.unembedding = Linear(self.hidden_dim, num_features)
        
        
        # Encoder
        self.encoder_layers = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        for i in range(num_layers):
            in_channels = self.hidden_dim
            out_channels = self.hidden_dim if i < num_layers - 1 else self.smallest_dim
            self.encoder_norms.append(nn.LayerNorm(in_channels))
            self.encoder_layers.append(EdgeConv(Sequential(
                Linear(in_channels * 2, self.hidden_dim * 8),
                LeakyReLU(0.1),
                Linear(self.hidden_dim * 8, out_channels),
                torch.nn.Tanh()
            ), aggr='max'))
        
        # Decoder
        self.decoder_layers = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        for i in range(num_layers-1):
            in_channels = self.smallest_dim if i == 0 else self.hidden_dim
            out_channels = self.hidden_dim
            self.decoder_norms.append(nn.LayerNorm(in_channels))
            self.decoder_layers.append(EdgeConv(Sequential(
                Linear(in_channels * 2, self.hidden_dim * 8),
                LeakyReLU(0.1),
                Linear(self.hidden_dim * 8, out_channels),
                torch.nn.Tanh()
            ), aggr='max'))
        final_decoder_in_channels = self.smallest_dim if num_layers == 1 else self.hidden_dim
        self.decoder_norms.append(nn.LayerNorm(final_decoder_in_channels))
        self.decoder_layers.append(EdgeConv(Sequential(
            Linear(final_decoder_in_channels * 2, self.hidden_dim * 8),
            LeakyReLU(0.1),
            Linear(self.hidden_dim * 8, self.hidden_dim),
        ), aggr='max'))

        # Tracking variables
        self.background_test_loss = None
        self.background_train_loss = None
        self.signal_loss = None
        self.train_hist = []
        self.val_hist = []
        self.signal_hist = []
        self.test_data = None
        self.signal_data = None

    def encoder(self, x, edge_index, data, training=True):
        """
        Encode input node features into a latent representation.

        Args:
            x (Tensor): Node feature matrix of shape [num_nodes, num_features].
            edge_index (Tensor): Edge connectivity matrix.
            data (Data): PyG Data object (not used but passed for flexibility).
            training (bool): Indicates if in training mode (unused).

        Returns:
            Tensor: Encoded node features.
        """
        for i, layer in enumerate(self.encoder_layers):
            h = self.encoder_norms[i](x)
            h = layer(h, edge_index)
            if h.shape == x.shape:
                x = x + h
            else:
                x = h
        return x

    def decoder(self, x, edge_index, batch, training=True):
        """
        Decode latent node features to reconstruct original features.

        Args:
            x (Tensor): Latent node features.
            edge_index (Tensor): Edge connectivity matrix.
            batch (Tensor): Batch indices for each node.
            training (bool): Indicates if in training mode (unused).

        Returns:
            Tensor: Reconstructed node features.
        """
        for i, layer in enumerate(self.decoder_layers):
            h = self.decoder_norms[i](x)
            h = layer(h, edge_index)
            if h.shape == x.shape:
                x = x + h
            else:
                x = h
        return x

    def forward(self, data, knn=False, topk=False, training=True):
        """
        Perform a full forward pass through the autoencoder.

        Args:
            data (Data): PyG graph data object containing 'x', 'edge_index', and optionally 'batch'.
            knn (bool): If True, recomputes the edge_index using kNN graph on encoded features.
            topk (bool): If True, applies TopKPooling (if defined) after encoding.
            training (bool): If True, model is in training mode (passed to submodules).

        Returns:
            Tensor: Reconstructed node features.
        """
        x, edge_index = data.x, data.edge_index
        x = self.encoder(self.embedding(x), edge_index, data, training)

        if knn:
            edge_index = knn_graph(x, k=self.num_reduced_edges, batch=data.batch, loop=False).to(data.x.device)

        if topk:
            x, edge_index, _, batch, _, _ = self.topk(x, edge_index.to(torch.int64), batch=data.batch)
        else:
            batch = data.batch

        x = self.decoder(x, edge_index, batch, training)
        x = self.unembedding(x)
        return x