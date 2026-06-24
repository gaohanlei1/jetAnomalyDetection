import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.utils import to_dense_batch

class JetTransformerMaskedEncoder(nn.Module):
    """
    Transformer-based masked autoencoder for jet constituent features.

    The first two input features are interpreted as eta-phi coordinates and are
    used only to build a learned positional encoding. The remaining features are
    embedded as input tokens. The model accepts either a dense tensor with shape
    [batch_size, num_nodes, num_features] or a PyG mini-batch where `data.x` has
    shape [total_nodes, num_features] and `data.batch` identifies each node's
    event. PyG mini-batches are padded internally and passed to the transformer
    with a key padding mask. During training, a random subset of valid non-CLS
    tokens is replaced by a learned mask embedding for each event, and the model
    predicts the masked non-position features.
    """

    def __init__(
        self,
        num_features: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        mask_ratio: float = 0.3,
    ):
        """
        Initialize the transformer masked autoencoder.

        Args:
            num_features: Total number of input features per node. The first two
                features are eta-phi and are not reconstructed.
            hidden_dim: Token embedding dimension.
            num_layers: Number of transformer encoder blocks.
            num_heads: Number of self-attention heads.
            ffn_dim: Hidden dimension of the transformer feed-forward network.
                Defaults to 4 * hidden_dim.
            dropout: Dropout used inside transformer blocks.
            mask_ratio: Default fraction of non-CLS input tokens to mask.
        """
        super().__init__()

        if num_features <= 2:
            raise ValueError("num_features must be greater than 2 because x[:, :2] is reserved for eta-phi.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1].")

        self.num_features = num_features
        self.reconstruction_dim = num_features - 2
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim if ffn_dim is not None else hidden_dim * 4
        self.dropout = dropout
        self.mask_ratio = mask_ratio

        self.feature_embedding = nn.Linear(self.reconstruction_dim, hidden_dim)
        self.position_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cls_embedding = nn.Parameter(torch.zeros(1, hidden_dim))
        self.cls_position_embedding = nn.Parameter(torch.zeros(1, hidden_dim))
        self.mask_embedding = nn.Parameter(torch.zeros(1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.prediction_head = nn.Linear(hidden_dim, self.reconstruction_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.cls_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_position_embedding, std=0.02)
        nn.init.trunc_normal_(self.mask_embedding, std=0.02)

    def _sample_mask(
        self,
        batch_size: int,
        num_nodes: int,
        device: torch.device,
        mask_ratio: float,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)

        if valid_mask is None:
            valid_mask = torch.ones(batch_size, num_nodes, dtype=torch.bool, device=device)

        for batch_idx in range(batch_size):
            valid_indices = torch.nonzero(valid_mask[batch_idx], as_tuple=False).flatten()
            num_valid = valid_indices.numel()
            num_masked = int(round(num_valid * mask_ratio))

            if num_masked == 0:
                continue

            shuffled_valid_indices = valid_indices[
                torch.randperm(num_valid, device=device)[:num_masked]
            ]
            mask[batch_idx, shuffled_valid_indices] = True

        return mask

    def forward(self, data, mask_ratio: float | None = None):
        """
        Run the masked transformer autoencoder.

        Args:
            data: Either a PyG-style mini-batch with `.x` and `.batch`, a
                PyG-style single graph with `.x`, or a raw dense tensor of shape
                [batch_size, num_nodes, num_features]. `edge_index` is
                intentionally unused.
            mask_ratio: Optional override for the fraction of non-CLS tokens to mask.

        Returns:
            A dictionary that can be passed directly to `self.loss(output)`.
        """
        if hasattr(data, "x"):
            if hasattr(data, "batch") and data.batch is not None: #PyG-style mini-batch
                x, valid_mask = to_dense_batch(data.x, data.batch)
            else: # PyG-style single graph
                x = data.x.unsqueeze(0)
                valid_mask = torch.ones(
                    1,
                    data.x.size(0),
                    dtype=torch.bool,
                    device=data.x.device,
                )
        else: # Raw dense tensor
            x = data
            valid_mask = torch.ones(
                x.size(0),
                x.size(1),
                dtype=torch.bool,
                device=x.device,
            )

        if x.dim() != 3:
            raise ValueError(
                f"Expected x to have shape [batch_size, num_nodes, feature_dim], got {tuple(x.shape)}."
            )
        if x.size(-1) != self.num_features:
            raise ValueError(f"Expected x.size(-1) == {self.num_features}, got {x.size(-1)}.")

        batch_size, num_nodes, _ = x.shape
        
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        if not 0.0 <= mask_ratio <= 1.0:
            raise ValueError("mask_ratio must be in [0, 1].")

        eta_phi = x[:, :, :2]
        target_features = x[:, :, 2:]

        token_embeddings = self.feature_embedding(target_features)
        mask = self._sample_mask(batch_size, num_nodes, x.device, mask_ratio, valid_mask)
        token_embeddings = token_embeddings.clone()
        token_embeddings[mask] = self.mask_embedding

        token_positions = self.position_embedding(eta_phi)
        token_embeddings = token_embeddings + token_positions

        cls_token = self.cls_embedding + self.cls_position_embedding
        cls_token = cls_token.unsqueeze(0).expand(batch_size, 1, -1)
        input_tokens = torch.cat([cls_token, token_embeddings], dim=1)

        cls_valid_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        input_valid_mask = torch.cat([cls_valid_mask, valid_mask], dim=1)
        key_padding_mask = ~input_valid_mask

        hidden_states = self.transformer(input_tokens, src_key_padding_mask=key_padding_mask)
        predictions = self.prediction_head(hidden_states)

        return {
            "hidden_states": hidden_states, # shape: [batch_size, num_nodes + 1, hidden_dim]
            "predictions": predictions, # shape: [batch_size, num_nodes + 1, reconstruction_dim]
            "mask": mask, # shape: [batch_size, num_nodes], True for masked positions
            "target": target_features, # shape: [batch_size, num_nodes, reconstruction_dim]
            "valid_mask": valid_mask, # shape: [batch_size, num_nodes], True for valid positions, False for padded positions
        }

    def loss(self, output: dict) -> torch.Tensor:
        """
        Compute MSE only on masked non-CLS tokens.

        Args:
            output: The dictionary returned by `forward()`.

        Returns:
            Scalar MSE loss over masked node positions. If no tokens are masked,
            the loss falls back to all non-CLS tokens to avoid an empty loss.
        """
        predictions = output["predictions"][:, 1:] # Exclude CLS token
        target = output["target"]
        mask = output["mask"]
        valid_mask = output["valid_mask"]

        loss_mask = mask & valid_mask
        if loss_mask.any():
            return F.mse_loss(predictions[loss_mask], target[loss_mask])

        return F.mse_loss(predictions[valid_mask], target[valid_mask])


# Backward-compatible alias for training scripts that still import the old name.
JetGraphAutoencoder = JetTransformerMaskedAutoencoder