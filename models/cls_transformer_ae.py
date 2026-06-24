import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

class JetClassTokenTransformerAE(nn.Module):
    """
    Transformer-based autoencoder with class token for jet anomaly detection.
    """
    def __init__(
        self,
        num_features: int = 3,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.1,
        num_slots: int = 64,
        loss_type: str = "hungarian",
    ):
        """
        Initialize the JetClassTokenTransformerAE.

        Args:
            num_features: Total number of input features per node. The first two
                features are eta-phi and are also used for positional embedding.
            hidden_dim: Token embedding dimension.
            latent_dim: Size of the class-token bottleneck.
            num_layers: Number of transformer encoder blocks.
            num_heads: Number of self-attention heads.
            ffn_dim: Hidden dimension of the transformer feed-forward network.
                Defaults to 4 * hidden_dim.
            dropout: Dropout used inside transformer blocks.
            num_slots: Fixed number of learned output slots. This must be at
                least as large as the number of particles you want to reconstruct.
            loss_type: Reconstruction loss to use. "hungarian" treats particles
                as an unordered set; "index" compares slot i to particle i.
        """

        super(JetClassTokenTransformerAE, self).__init__()

        if num_features <= 2:
            raise ValueError("num_features must be greater than 2 because x[:, :2] is reserved for eta-phi.")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if num_slots <= 0:
            raise ValueError("num_slots must be positive.")
        if not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be in [0, 1].")
        if loss_type not in {"hungarian", "index"}:
            raise ValueError("loss_type must be either 'hungarian' or 'index'.")

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim if ffn_dim is not None else hidden_dim * 4
        self.dropout = dropout
        self.num_slots = num_slots
        self.loss_type = loss_type

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.feature_embedding = nn.Linear(num_features, hidden_dim)
        self.position_embedding = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.to_latent = nn.Linear(hidden_dim, self.latent_dim)
        self.from_latent = nn.Linear(self.latent_dim, hidden_dim)

        self.decoder_slots = nn.Parameter(torch.zeros(1, self.num_slots, hidden_dim))

        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers)

        self.reconstruction_head = nn.Linear(hidden_dim, num_features)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.cls_embedding, std=0.02)
        nn.init.trunc_normal_(self.decoder_slots, std=0.02)

    def _to_dense(self, data):
        if hasattr(data, "x"):
            if hasattr(data, "batch") and data.batch is not None:
                return to_dense_batch(data.x, data.batch)

            x = data.x.unsqueeze(0)
            valid_mask = torch.ones(
                1,
                data.x.size(0),
                dtype=torch.bool,
                device=data.x.device,
            )
            return x, valid_mask

        x = data
        if x.dim() != 3:
            raise ValueError(
                f"Expected dense input [batch_size, num_nodes, num_features], got {tuple(x.shape)}."
            )
        valid_mask = torch.ones(
            x.size(0),
            x.size(1),
            dtype=torch.bool,
            device=x.device,
        )
        return x, valid_mask

    def forward(self, data):
        x, valid_mask = self._to_dense(data)

        if x.size(-1) != self.num_features:
            raise ValueError(f"Expected x.size(-1) == {self.num_features}, got {x.size(-1)}.")
        if x.size(1) > self.num_slots:
            raise ValueError(
                f"Input has {x.size(1)} particles, but model has only {self.num_slots} decoder slots. "
                "Increase num_slots or truncate/pad the dataset before training."
            )

        batch_size = x.size(0)
        eta_phi = x[:, :, :2]

        particle_tokens = self.feature_embedding(x)
        particle_tokens = particle_tokens + self.position_embedding(eta_phi)

        cls_token = self.cls_embedding.expand(batch_size, -1, -1)
        encoder_input = torch.cat([cls_token, particle_tokens], dim=1)

        cls_valid_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
        encoder_valid_mask = torch.cat([cls_valid_mask, valid_mask], dim=1)
        key_padding_mask = ~encoder_valid_mask

        encoded = self.encoder(encoder_input, src_key_padding_mask=key_padding_mask)
        cls_hidden = encoded[:, 0]
        latent = self.to_latent(cls_hidden)

        latent_context = self.from_latent(latent)
        decoder_input = self.decoder_slots.expand(batch_size, -1, -1)
        decoder_input = decoder_input + latent_context[:, None, :]

        decoded = self.decoder(decoder_input)
        reconstruction = self.reconstruction_head(decoded)

        return {
            "reconstruction": reconstruction,
            "target": x,
            "valid_mask": valid_mask,
            "latent": latent,
        }

    def _index_loss(self, output: dict, per_event: bool = False) -> torch.Tensor:
        reconstruction = output["reconstruction"]
        target = output["target"]
        valid_mask = output["valid_mask"]

        num_target_nodes = target.size(1)
        reconstruction = reconstruction[:, :num_target_nodes, :]

        squared_error = F.mse_loss(reconstruction, target, reduction="none")
        token_loss = squared_error.mean(dim=-1)

        if per_event:
            event_loss_sum = (token_loss * valid_mask).sum(dim=1)
            event_loss_count = valid_mask.sum(dim=1).clamp(min=1)
            return event_loss_sum / event_loss_count

        return token_loss[valid_mask].mean()

    def _hungarian_loss(self, output: dict, per_event: bool = False) -> torch.Tensor:
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:
            raise ImportError(
                "Hungarian set loss requires scipy. Install scipy or use loss_type='index'."
            ) from exc

        reconstruction = output["reconstruction"]
        target = output["target"]
        valid_mask = output["valid_mask"]

        event_losses = []
        for event_idx in range(target.size(0)):
            event_target = target[event_idx, valid_mask[event_idx]]
            event_reconstruction = reconstruction[event_idx]

            if event_target.size(0) == 0:
                event_losses.append(event_reconstruction.new_zeros(()))
                continue
            if event_target.size(0) > event_reconstruction.size(0):
                raise ValueError(
                    f"Event has {event_target.size(0)} valid particles, but only "
                    f"{event_reconstruction.size(0)} decoder slots."
                )

            pairwise_cost = (
                event_reconstruction[:, None, :] - event_target[None, :, :]
            ).pow(2).mean(dim=-1)

            row_idx, col_idx = linear_sum_assignment(pairwise_cost.detach().cpu().numpy())
            row_idx = torch.as_tensor(row_idx, dtype=torch.long, device=pairwise_cost.device)
            col_idx = torch.as_tensor(col_idx, dtype=torch.long, device=pairwise_cost.device)
            event_losses.append(pairwise_cost[row_idx, col_idx].mean())

        event_losses = torch.stack(event_losses)
        if per_event:
            return event_losses
        return event_losses.mean()

    def loss(self, output: dict, per_event: bool = False) -> torch.Tensor:
        if self.loss_type == "hungarian":
            return self._hungarian_loss(output, per_event=per_event)
        return self._index_loss(output, per_event=per_event)
