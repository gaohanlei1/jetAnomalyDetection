import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import lejepa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function

"""
Usage example:

model_config = ParticleTransformerConfig(
    input_dim=7,
    embed_dim=128,
    num_heads=8,
    num_layers=4,
    representation_dim=128,
    use_pairwise_bias=True,
    compute_dtype=torch.bfloat16,
    use_internal_autocast=False,
)

augmentation_config = MultiViewAugmentationConfig(
    num_global_views=2,
    num_local_views=6,
    global_drop_pt_frac_range=(0.0, 0.70),
    local_drop_pt_frac_range=(0.70, 0.95),
    min_nodes=4,
    pt_index=2,
    pt_drop_power=1.0,
)

loss_config = LeJEPALossConfig(
    invariant_weight=1.0,
    sigreg_weight=0.02,
    epps_pulley_num_points=17,
    num_slices=1024,
)

model = LeJEPAParticleTransformerRepresentation(
    model_config=model_config,
    augmentation_config=augmentation_config,
    loss_config=loss_config,
)

x = torch.randn(32, 64, 7, device="cuda")
padding_mask = torch.zeros(32, 64, dtype=torch.bool, device="cuda")

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    output = model.forward_pretrain(
        x,
        padding_mask=padding_mask,
        normalize_output=False,
    )

loss = output["total_loss"]
loss.backward()

print(output["total_loss"])
print(output["invariant_loss"])
print(output["sigreg_loss"])
print(output["z_views"].shape)

"""

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class ParticleTransformerConfig:
    """Configuration for the JetClass ParticleTransformer backbone.

    ``input_feature_names`` describes the complete tensor entering the model.
    The first four-vector features remain untouched for pairwise-bias
    construction, but are excluded from the learned node embedding. Selected
    derived node features are standardized over all valid particles in the
    current batch inside :meth:`ParticleTransformerBackbone.forward`.
    """

    input_dim: int = 16
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 4
    num_class_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.1
    class_dropout: float = 0.0
    representation_dim: int = 128

    use_pairwise_bias: bool = True
    pairwise_hidden_dim: int = 64
    pairwise_num_features: int = 4

    compute_dtype: torch.dtype = torch.bfloat16
    use_internal_autocast: bool = False
    eps: float = 1e-8

    input_feature_names: Tuple[str, ...] = (
        "part_px",
        "part_py",
        "part_pz",
        "part_energy",
        "part_pt",
        "log_pt_fraction",
        "part_deta",
        "part_dphi",
        "d0_sig",
        "dz_sig",
        "part_charge",
        "part_isChargedHadron",
        "part_isNeutralHadron",
        "part_isPhoton",
        "part_isElectron",
        "part_isMuon",
    )
    four_vector_feature_names: Tuple[str, ...] = (
        "part_px",
        "part_py",
        "part_pz",
        "part_energy",
    )
    standardized_feature_names: Tuple[str, ...] = (
        "log_pt_fraction",
        "d0_sig",
        "dz_sig",
    )

    feature_indices: Dict[str, int] = field(init=False, repr=False)
    node_feature_indices: Tuple[int, ...] = field(init=False)
    standardized_node_feature_indices: Tuple[int, ...] = field(init=False)
    px_index: int = field(init=False)
    py_index: int = field(init=False)
    pz_index: int = field(init=False)
    energy_index: int = field(init=False)
    pt_index: int = field(init=False)
    log_pt_fraction_index: int = field(init=False)
    node_input_dim: int = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(self.input_feature_names)
        if len(names) != len(set(names)):
            raise ValueError(f"input_feature_names contains duplicates: {names}")
        if self.input_dim != len(names):
            raise ValueError(
                f"input_dim={self.input_dim} does not match "
                f"len(input_feature_names)={len(names)}."
            )

        required = set(self.four_vector_feature_names) | {
            "part_pt",
            "log_pt_fraction",
        } | set(self.standardized_feature_names)
        missing = sorted(required - set(names))
        if missing:
            raise ValueError(f"Missing required model input features: {missing}")

        self.feature_indices = {name: i for i, name in enumerate(names)}
        self.px_index = self.feature_indices["part_px"]
        self.py_index = self.feature_indices["part_py"]
        self.pz_index = self.feature_indices["part_pz"]
        self.energy_index = self.feature_indices["part_energy"]
        self.pt_index = self.feature_indices["part_pt"]
        self.log_pt_fraction_index = self.feature_indices["log_pt_fraction"]

        excluded = set(self.four_vector_feature_names)
        self.node_feature_indices = tuple(
            i for i, name in enumerate(names) if name not in excluded
        )
        node_names = [names[i] for i in self.node_feature_indices]
        self.standardized_node_feature_indices = tuple(
            node_names.index(name) for name in self.standardized_feature_names
        )
        self.node_input_dim = len(self.node_feature_indices)


# ------------------------------------------------------------
# Basic modules
# ------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square LayerNorm.

    Submodule stage:
        Used inside node embedding, Transformer blocks, CLS pooling,
        and the final representation projection head.

    Difference from LayerNorm:
        RMSNorm normalizes by root mean square only. It does not subtract
        the mean. It is often used in modern Transformer implementations.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        rms = torch.rsqrt(x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_normed = x_float * rms
        return x_normed.to(dtype=x.dtype) * self.weight


class SwiGLU(nn.Module):
    """
    SwiGLU activation.

    Submodule stage:
        Used inside the feed-forward network of each Transformer block.

    Input shape:
        (..., 2 * hidden_dim)

    Output shape:
        (..., hidden_dim)

    It splits the last dimension into two halves:

        x = [x1, x2]
        output = x1 * SiLU(x2)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * F.silu(x2)


class SwiGLUFFN(nn.Module):
    """
    Feed-forward network using SwiGLU.

    Submodule stage:
        Used after self-attention in every Transformer block.

    Structure:
        Linear(dim -> 2 * hidden_dim)
        SwiGLU
        Linear(hidden_dim -> dim)
        Dropout

    The first Linear outputs twice the hidden dimension because SwiGLU
    splits it into gate/value halves.
    """

    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = dim * mult

        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.act = SwiGLU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# ------------------------------------------------------------
# Physics utilities
# ------------------------------------------------------------

def delta_phi(phi_i: torch.Tensor, phi_j: torch.Tensor) -> torch.Tensor:
    """
    Periodic phi difference.

    Returns values in [-pi, pi).
    """

    return (phi_i - phi_j + math.pi) % (2 * math.pi) - math.pi


def build_four_vector_from_jetclass(
    x: torch.Tensor,
    feature_config: "MultiViewAugmentationConfig",
) -> torch.Tensor:
    """Read particle four-vectors using configurable JetClass feature indices."""

    return torch.stack(
        [
            x[..., feature_config.px_index],
            x[..., feature_config.py_index],
            x[..., feature_config.pz_index],
            x[..., feature_config.energy_index],
        ],
        dim=-1,
    )


def pairwise_physics_features(
    x: torch.Tensor,
    feature_config: "MultiViewAugmentationConfig",
    padding_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Construct pairwise physics features for attention bias.

    Submodule stage:
        Used by PairwiseAttentionBias before the Transformer encoder.

    Input:
        x: (B, N, F)

    Output:
        pair_features: (B, N, N, 4)

    The four pairwise features are:

        1. log(1 + ΔR)
        2. log(1 + min(pt_i, pt_j) * ΔR)
        3. log(1 + min(pt_i, pt_j) / (pt_i + pt_j))
        4. log(1 + m_ij^2)

    where m_ij is the invariant mass of the pair.

    These are adapted from the original ParticleTransformer pairwise
    Lorentz features, but rewritten for our explicit node feature order.
    """

    if feature_config is None:
        raise ValueError(
            f"feature_config must be provided to compute pairwise physics features."
            f"This error might happen because you are calling the forward pass of models without "
            f"MultiView Augmentation. pairwise_physics_features use variable feature indices from "
            f"MultiViewAugmentationConfig, so it cannot be called without a feature_config."
            f"Consider implementing further functionality if you want to train models without MultiView Augmentation."
        )
    px_node = x[..., feature_config.px_index]
    py_node = x[..., feature_config.py_index]
    pz_node = x[..., feature_config.pz_index]
    pt = torch.sqrt(px_node.square() + py_node.square()).clamp(min=eps)
    eta = torch.asinh(pz_node / pt)
    phi = torch.atan2(py_node, px_node)

    eta_i = eta.unsqueeze(2)
    eta_j = eta.unsqueeze(1)

    phi_i = phi.unsqueeze(2)
    phi_j = phi.unsqueeze(1)

    pt_i = pt.unsqueeze(2)
    pt_j = pt.unsqueeze(1)

    d_eta = eta_i - eta_j
    d_phi = delta_phi(phi_i, phi_j)
    delta_r = torch.sqrt(d_eta.square() + d_phi.square() + eps)

    pt_min = torch.minimum(pt_i, pt_j)

    log_delta_r = torch.log1p(delta_r)
    log_kt = torch.log1p(pt_min * delta_r)
    log_z = torch.log1p(pt_min / (pt_i + pt_j).clamp(min=eps))

    p4 = build_four_vector_from_jetclass(
        x,
        feature_config=feature_config,
    )
    p4_i = p4.unsqueeze(2) # shape: (B, N, 1, 4)
    p4_j = p4.unsqueeze(1) # shape: (B, 1, N, 4)
    p4_ij = p4_i + p4_j # Add all pairs of four-vectors, shape: (B, N, N, 4)

    px = p4_ij[..., 0]
    py = p4_ij[..., 1]
    pz = p4_ij[..., 2]
    energy = p4_ij[..., 3]

    m2_ij = energy.square() - px.square() - py.square() - pz.square()
    m2_ij = m2_ij.clamp(min=eps)

    log_pair_mass2 = torch.log1p(m2_ij)

    pair_features = torch.stack(
        [log_delta_r, log_kt, log_z, log_pair_mass2],
        dim=-1,
    )

    if padding_mask is not None:
        valid = ~padding_mask
        valid_pair = valid.unsqueeze(1) & valid.unsqueeze(2)
        pair_features = pair_features.masked_fill(~valid_pair.unsqueeze(-1), 0.0)

    return pair_features


# ------------------------------------------------------------
# Embedding modules
# ------------------------------------------------------------

class NodeEmbedding(nn.Module):
    """
    ParT-style particle feature embedding.

    Input:
        x: (B, N, input_dim)

    Output:
        h: (B, N, embed_dim)

    For default embed_dim=128, this is:

        Linear(input_dim -> 128)
        GELU
        LayerNorm
        Linear(128 -> 512)
        GELU
        LayerNorm
        Linear(512 -> 128)

    This is closer to the baseline ParT particle embedding described in the
    paper.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        hidden_dim = 4 * embed_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PairwiseAttentionBias(nn.Module):
    """
    Pairwise attention bias from physics features.

    Submodule stage:
        Stage 2 of the main model.

    Input:
        x: (B, N, input_dim)
        padding_mask: (B, N), True means padded node

    Output:
        attn_bias: (B * num_heads, N, N)

    This tensor is passed to PyTorch MultiheadAttention as an additive
    attention mask.

    Important:
        This is not a binary padding mask. It is a learned pairwise bias.
        Each attention head gets its own scalar bias for each node pair.

    For attention logits:

        logits_ij = q_i dot k_j / sqrt(d_head) + pair_bias_ij

    So the model can learn that physically close or physically related
    particles should attend differently.
    """

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int = 64,
        num_pair_features: int = 4,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.eps = eps

        self.mlp = nn.Sequential(
            RMSNorm(num_pair_features),
            nn.Linear(num_pair_features, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with record_function("pairwise_feature_construction"):
            with torch.no_grad(): # Input features are not learned, so no gradients needed
                pair_features = pairwise_physics_features(
                    x,
                    feature_config=getattr(self, "feature_config", None),
                    padding_mask=padding_mask,
                    eps=self.eps,
                ) # shape: (B, N, N, num_pair_features)

        with record_function("pairwise_mlp"):
            bias = self.mlp(pair_features)
        with record_function("pairwise_bias_reshape"):
            bias = bias.permute(0, 3, 1, 2).contiguous()

            batch_size, num_heads, seq_len, _ = bias.shape
            bias = bias.view(batch_size * num_heads, seq_len, seq_len)

        return bias


# ------------------------------------------------------------
# Transformer blocks
# ------------------------------------------------------------

class ParticleTransformerBlock(nn.Module):
    """
    Pre-norm Transformer encoder block.

    Submodule stage:
        Repeated main encoder stage.

    Structure:
        h = h + MultiheadAttention(RMSNorm(h))
        h = h + SwiGLUFFN(RMSNorm(h))

    Input:
        h: (B, N, embed_dim)
        attn_bias: (B * num_heads, N, N), optional
        padding_mask: (B, N), optional, True means padded node

    Output:
        h: (B, N, embed_dim)

    This follows the spirit of:
        RMSNorm
        pre-norm residual block
        SwiGLU FFN

    but removes unused compatibility branches.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attn_norm = RMSNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = RMSNorm(embed_dim)
        self.ffn = SwiGLUFFN(
            dim=embed_dim,
            mult=ffn_mult,
            dropout=dropout,
        )

    def forward(
        self,
        h: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = h
        h_norm = self.attn_norm(h)

        attn_out, _ = self.self_attn(
            h_norm,
            h_norm,
            h_norm,
            attn_mask=attn_bias,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        h = residual + self.attn_dropout(attn_out)

        residual = h
        h = residual + self.ffn(self.ffn_norm(h))

        if padding_mask is not None:
            h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        return h


class ParticleTransformerEncoder(nn.Module):
    """
    Stack of ParticleTransformerBlock layers.

    Submodule stage:
        Main token-mixing encoder.

    Input:
        h: (B, N, embed_dim)
        attn_bias: (B * num_heads, N, N), optional
        padding_mask: (B, N), optional

    Output:
        h: (B, N, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                ParticleTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        h: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            h = layer(
                h,
                attn_bias=attn_bias,
                padding_mask=padding_mask,
            )

        return h


# ------------------------------------------------------------
# CLS pooling and representation head
# ------------------------------------------------------------

class ClassAttentionBlock(nn.Module):
    """
    One ParT-style class attention block.

    The CLS token is the query.
    Encoded particle tokens are keys and values.
    Only the CLS token is updated.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_mult * embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        cls: torch.Tensor,
        h: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = cls

        query = self.query_norm(cls)
        context = self.context_norm(h)

        cls_out, _ = self.attn(
            query,
            context,
            context,
            key_padding_mask=padding_mask,
            need_weights=False,
        )

        cls = residual + self.attn_dropout(cls_out)
        cls = cls + self.ffn(self.ffn_norm(cls))

        return cls


class CLSPooling(nn.Module):
    """
    Two-block CLS attention pooling module.

    This is closer to the original ParT class-attention setup:
    particle tokens are first updated by particle attention, then a learnable
    CLS token reads them through multiple class attention blocks.

    Default:
        num_class_layers = 2
        dropout = 0.0
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        num_class_layers: int = 2,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.layers = nn.ModuleList(
            [
                ClassAttentionBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    dropout=dropout,
                )
                for _ in range(num_class_layers)
            ]
        )

    def forward(
        self,
        h: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = h.size(0)
        cls = self.cls_token.expand(batch_size, -1, -1)

        for layer in self.layers:
            cls = layer(
                cls=cls,
                h=h,
                padding_mask=padding_mask,
            )

        return cls.squeeze(dim=1)


class RepresentationHead(nn.Module):
    """
    Projection head for representation learning.

    Submodule stage:
        Final stage of the main model.

    Input:
        cls: (B, embed_dim)

    Output:
        z: (B, representation_dim)

    This replaces the original Linear classifier head.

    We can use the output representation for:
        contrastive learning,
        anomaly detection,
        downstream classifiers,
        clustering,
        latent-space visualization,
        or reconstruction/auxiliary objectives.
    """

    def __init__(
        self,
        embed_dim: int,
        representation_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, representation_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ------------------------------------------------------------
# Main model
# ------------------------------------------------------------

class MinimalParticleTransformer(nn.Module):
    """
    Minimal ParticleTransformer for representation learning.

    Main stages:
        1. NodeEmbedding
        2. PairwiseAttentionBias
        3. ParticleTransformerEncoder
        4. CLSPooling

    Input:
        x: (B, N, F)

    Optional input:
        padding_mask: (B, N)
            Boolean tensor.
            True means this node is padding and should be ignored.

    Output:
        z: (B, hidden_dim), final CLS hidden states
    """

    def __init__(self, config: ParticleTransformerConfig):
        super().__init__()

        self.config = config

        self.node_embedding = NodeEmbedding(
            input_dim=config.node_input_dim,
            embed_dim=config.embed_dim,
            dropout=config.dropout,
        )

        if config.use_pairwise_bias:
            self.pairwise_bias = PairwiseAttentionBias(
                num_heads=config.num_heads,
                hidden_dim=config.pairwise_hidden_dim,
                num_pair_features=config.pairwise_num_features,
                dropout=config.dropout,
                eps=config.eps,
            )
        else:
            self.pairwise_bias = None

        if self.pairwise_bias is not None:
            self.pairwise_bias.feature_config = config

        node_index_tensor = torch.tensor(
            config.node_feature_indices,
            dtype=torch.long,
        )
        standardized_index_tensor = torch.tensor(
            config.standardized_node_feature_indices,
            dtype=torch.long,
        )
        self.register_buffer(
            "_node_feature_indices",
            node_index_tensor,
            persistent=False,
        )
        self.register_buffer(
            "_standardized_node_feature_indices",
            standardized_index_tensor,
            persistent=False,
        )
        ### Buffers for batch-normalizing node features
        ### Needed for consistent behavior between training and evaluation
        self.register_buffer(
            "_feature_running_mean",
            torch.zeros(len(standardized_index_tensor), dtype=torch.float32),
        )
        self.register_buffer(
            "_feature_running_var",
            torch.zeros(len(standardized_index_tensor), dtype=torch.float32),
        )
        self.register_buffer(
            "_feature_num_batches_tracked",
            torch.zeros(0, dtype=torch.long),
        )

        self.encoder = ParticleTransformerEncoder(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            ffn_mult=config.ffn_mult,
            dropout=config.dropout,
        )

        self.cls_pooling = CLSPooling(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            ffn_mult=config.ffn_mult,
            dropout=config.class_dropout,
            num_class_layers=config.num_class_layers,
        )

    def _prepare_node_features(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Drop raw four-vectors and standardize selected node features.

        During training:
            - Compute statistics over all valid nodes in the current global DDP batch.
            - Normalize with the current batch statistics.
            - Update running mean and variance.

        During evaluation:
            - Normalize with frozen running statistics.
            - Evaluation batch composition therefore does not affect the output.
        """

        node_x = torch.index_select(
            x,
            dim=-1,
            index=self._node_feature_indices,
        ).clone()
        # node_x: (B, N, num_selected_features)

        if padding_mask is None:
            valid_mask = torch.ones(
                x.shape[:2],
                dtype=torch.bool,
                device=x.device,
            )
        else:
            valid_mask = ~padding_mask
        # valid_mask: (B, N)

        standardized_indices = self._standardized_node_feature_indices

        if standardized_indices.numel() > 0:
            selected = torch.index_select(
                node_x.float(),
                dim=-1,
                index=standardized_indices,
            )
            # selected: (B, N, num_standardized_features)

            if self.training:
                valid_values = selected[valid_mask]
                # valid_values:
                # (num_valid_nodes_on_this_rank, num_standardized_features)

                num_features = selected.shape[-1]

                if valid_values.numel() > 0:
                    local_sum = valid_values.sum(dim=0)
                    local_squared_sum = valid_values.square().sum(dim=0)
                    local_count = torch.tensor(
                        valid_values.shape[0],
                        dtype=torch.float32,
                        device=selected.device,
                    )
                else:
                    local_sum = torch.zeros(
                        num_features,
                        dtype=torch.float32,
                        device=selected.device,
                    )
                    local_squared_sum = torch.zeros_like(local_sum)
                    local_count = torch.zeros(
                        (),
                        dtype=torch.float32,
                        device=selected.device,
                    )

                # Synchronize statistics across DDP ranks.
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    torch.distributed.all_reduce(
                        local_sum,
                        op=torch.distributed.ReduceOp.SUM,
                    )
                    torch.distributed.all_reduce(
                        local_squared_sum,
                        op=torch.distributed.ReduceOp.SUM,
                    )
                    torch.distributed.all_reduce(
                        local_count,
                        op=torch.distributed.ReduceOp.SUM,
                    )

                if local_count.item() > 0:
                    batch_mean = local_sum / local_count
                    batch_var = (
                        local_squared_sum / local_count
                        - batch_mean.square()
                    ).clamp_min(0.0)

                    mean = batch_mean
                    var = batch_var

                    # Updating running buffers must not enter autograd.
                    with torch.no_grad():
                        momentum = self.feature_norm_momentum

                        self._feature_running_mean.mul_(
                            1.0 - momentum
                        ).add_(
                            momentum * batch_mean.detach()
                        )

                        self._feature_running_var.mul_(
                            1.0 - momentum
                        ).add_(
                            momentum * batch_var.detach()
                        )

                        self._feature_num_batches_tracked.add_(1)
                else:
                    # Extremely defensive fallback: no valid node on any rank.
                    mean = self._feature_running_mean
                    var = self._feature_running_var

            else: # Eval mode
                # Fixed statistics in eval mode.
                mean = self._feature_running_mean
                var = self._feature_running_var

            std = torch.sqrt(var).clamp_min(self.config.eps)

            standardized = (
                selected - mean.view(1, 1, -1)
            ) / std.view(1, 1, -1)

            selected = torch.where(
                valid_mask.unsqueeze(-1),
                standardized,
                torch.zeros_like(standardized),
            )

            node_x[..., standardized_indices] = selected.to(
                dtype=node_x.dtype
            )

        if padding_mask is not None:
            node_x = node_x.masked_fill(
                padding_mask.unsqueeze(-1),
                0.0,
            )

        return node_x

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode a full JetClass feature tensor into a CLS representation."""

        if x.ndim != 3:
            raise ValueError(f"Expected x to have shape (B, N, F), got {tuple(x.shape)}.")
        if x.size(-1) != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, "
                f"but got x.size(-1)={x.size(-1)}."
            )

        if padding_mask is not None:
            if padding_mask.shape != x.shape[:2]:
                raise ValueError(
                    f"Expected padding_mask shape {(x.shape[0], x.shape[1])}, "
                    f"got {tuple(padding_mask.shape)}."
                )
            padding_mask = padding_mask.bool()

        # Pairwise bias always sees the untouched raw four-vector entries.
        attn_bias = None
        if self.pairwise_bias is not None:
            attn_bias = self.pairwise_bias(x, padding_mask=padding_mask)

        # The learned node embedding never receives px, py, pz, or energy.
        node_x = self._prepare_node_features(x, padding_mask)

        use_autocast = (
            self.config.use_internal_autocast
            and x.device.type in {"cuda", "cpu"}
            and self.config.compute_dtype in {torch.float16, torch.bfloat16}
        )

        with torch.autocast(
            device_type=x.device.type,
            dtype=self.config.compute_dtype,
            enabled=use_autocast,
        ):
            h = self.node_embedding(node_x)
            if padding_mask is not None:
                h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

            h = self.encoder(
                h,
                attn_bias=attn_bias,
                padding_mask=padding_mask,
            )
            cls = self.cls_pooling(h, padding_mask=padding_mask)

        return cls

class ClassificationHead(nn.Module):
    """
    Binary classification head for supervised PART upper-bound training.

    Input:
        cls: (B, embed_dim)

    Output:
        logits: (B,)
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ------------------------------------------------------------
# Categorical classification head for JetClass
# ------------------------------------------------------------

class CategoricalClassificationHead(nn.Module):
    """JetClass categorical head producing one logit per truth label."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            RMSNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParticleTransformerClassifier(MinimalParticleTransformer):
    """
    Supervised binary classifier sharing the PART backbone with LeJEPA models.

    Main stages:
        1. NodeEmbedding
        2. PairwiseAttentionBias
        3. ParticleTransformerEncoder
        4. CLSPooling
        5. ClassificationHead

    Input:
        x: (B, N, F)

    Optional input:
        padding_mask: (B, N), True means padded node.

    Output:
        logits: (B,), unbounded signal logits.
    """

    def __init__(self, config: ParticleTransformerConfig):
        super().__init__(config)
        self.classifier_head = ClassificationHead(
            embed_dim=config.embed_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        cls = super().forward(x, padding_mask=padding_mask)

        use_autocast = (
            self.config.use_internal_autocast
            and x.device.type in {"cuda", "cpu"}
            and self.config.compute_dtype in {torch.float16, torch.bfloat16}
        )

        with torch.autocast(
            device_type=x.device.type,
            dtype=self.config.compute_dtype,
            enabled=use_autocast,
        ):
            logits = self.classifier_head(cls)

        return logits


# ------------------------------------------------------------
# Augmentation config
# ------------------------------------------------------------

@dataclass
class MultiViewAugmentationConfig:
    """
    Configuration for random pt-drop multi-view augmentation.

    This is the particle/node analogue of LeJEPA's multi-crop image
    augmentation.

    Instead of image crops, we generate multiple graph/node views by
    randomly dropping nodes until the dropped cumulative pt reaches a
    sampled threshold.

    Default view setup:
        - 2 global views
        - 6 local views

    Global views:
        Drop 0% to 70% of total valid pt.

    Local views:
        Drop 70% to 95% of total valid pt.

    Important:
        The dropping is stochastic and biased toward low-pt nodes.
        High-pt nodes can still be dropped, but with lower probability.

    Feature convention:
        By default, x[..., 4] is JetClass part_pt.
    """

    num_global_views: int = 2
    num_local_views: int = 6

    global_drop_pt_frac_range: Tuple[float, float] = (0.0, 0.70)
    local_drop_pt_frac_range: Tuple[float, float] = (0.70, 0.95)

    min_nodes: int = 4

    px_index: int = 0
    py_index: int = 1
    pz_index: int = 2
    energy_index: int = 3
    pt_index: int = 4
    log_pt_fraction_index: int = 5
    eps: float = 1e-8

    # Controls how strongly low-pt nodes are preferred for dropping.
    #
    # drop_weight_i = 1 / (pt_i + eps) ** pt_drop_power
    #
    # Larger value means low-pt nodes are much more likely to be dropped.
    pt_drop_power: float = 1.0

    # If True, dropped node features are zeroed out.
    # The corresponding padding_mask position is also set to True.
    zero_dropped_features: bool = True


# ------------------------------------------------------------
# Corrupted negative view augmentation config
# ------------------------------------------------------------

@dataclass
class CorruptedNegativeAugmentationConfig:
    """Configuration for corrupted negative views.

    Absolute constituent coordinates are intentionally absent. Angular
    corruptions operate only on the jet-relative ``part_deta`` and
    ``part_dphi`` entries. Whenever a corruption changes ``part_pt``, the
    derived ``log_pt_fraction`` entry is recomputed automatically.
    """

    num_negative_views: int = 4
    batch_mix_prob: float = 0.45
    pt_resample_prob: float = 0.25
    node_deta_dphi_rotation_prob: float = 0.20
    deta_dphi_shuffle_prob: float = 0.05
    identity_shuffle_prob: float = 0.05

    min_nodes: int = 4
    eps: float = 1e-8

    deta_index: int = 6
    dphi_index: int = 7
    pt_index: int = 4
    log_pt_fraction_index: int = 5
    d0_sig_index: int = 8
    dz_sig_index: int = 9
    charge_index: int = 10
    identity_start_index: int = 11
    identity_end_index: int = 16

    corrupt_node_frac: float = 1.0
    batch_mix_anchor_frac_min: float = 0.1
    batch_mix_anchor_frac_max: float = 0.9
    renormalize_pt_sum: bool = True



# ------------------------------------------------------------
# Random pt-drop + rotation multi-view augmentation
# ------------------------------------------------------------

class MultiViewAugmentation(nn.Module):
    """
    Vectorized multi-view random node dropping augmentation.

    Submodule role:
        Used inside LeJEPAParticleTransformerRepresentation as:

            self.augmentation

    Input:
        x: (B, N, F)
        padding_mask: optional bool tensor of shape (B, N)
            True means padded node.

    Output:
        views:
            List of tensors, each with shape (B, N, F)

        view_padding_masks:
            List of bool masks, each with shape (B, N)

        view_types:
            List[str], e.g. ["global", "global", "local", ...]

    Algorithm:
        For each view and each event, sample a target dropped cumulative-pt
        fraction, then produce a low-pt-biased random drop order in one
        vectorized pass.

        The old implementation sampled one node at a time with a Python
        while-loop and repeated torch.multinomial calls. That is faithful but
        slow on MPS/CUDA because it creates many small dynamic operations.

        This implementation uses an exponential-race / Gumbel-style weighted
        random permutation:

            score_i = -log(u_i) * pt_i ** pt_drop_power

        Nodes with larger weights, i.e. lower pt, tend to get smaller scores
        and therefore appear earlier in the drop order. We then drop nodes in
        that order until the cumulative dropped pt reaches the sampled target,
        while preserving at least min_nodes valid nodes.
        
    Important:
        Previously, the augmentation also includes a random rotation of jet, 
        which corresponds to a random shift along the phi-axis. However, since 
        during preprocessing stage (preprocessing.py) we already subtracted the 
        eta and phi coordinates of each PFCand by the fat jet center, the 
        random rotation is cancelled out. Therefore, no rotation is applied 
        here.

    This preserves the original sequence length N. It does not physically
    shrink the tensor. Dropped nodes are masked out.
    """

    def __init__(self, config: MultiViewAugmentationConfig):
        super().__init__()
        self.config = config

    def _drop_nodes(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
        drop_frac_range: Tuple[float, float],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create one augmented view for the whole batch.

        Input:
            x: (B, N, F)
            padding_mask: (B, N), True means padded node

        Output:
            view_x: (B, N, F)
            view_mask: (B, N), True means padded or dropped node
        """

        low, high = drop_frac_range
        if not (0.0 <= low <= high <= 1.0):
            raise ValueError(
                f"Invalid drop_frac_range={drop_frac_range}. "
                "Expected 0 <= low <= high <= 1."
            )

        batch_size, seq_len, _ = x.shape
        device = x.device

        valid_mask = ~padding_mask
        num_valid = valid_mask.sum(dim=1) # shape (B,)

        pt = x[..., self.config.pt_index].clamp(min=self.config.eps)
        pt = pt.masked_fill(~valid_mask, 0.0)
        total_pt = pt.sum(dim=1)

        # Events with too few valid nodes or non-positive total pt should not
        # receive additional augmentation drops.
        can_drop_event = (num_valid > self.config.min_nodes) & torch.isfinite(total_pt) & (total_pt > 0)

        # Sample a target dropped cumulative-pt fraction for each event in the batch.
        drop_frac = torch.empty(batch_size, device=device, dtype=pt.dtype).uniform_(low, high) # shape (B,)
        target_drop_pt = drop_frac * total_pt

        # Low-pt nodes get smaller weights and therefore earlier drop order.
        weights = pt.pow(self.config.pt_drop_power).clamp(min=self.config.eps)
        weights = weights.masked_fill(~valid_mask, 0.0)

        # Exponential race for weighted random permutation.
        # Smaller score means earlier in the drop order.
        u = torch.rand(batch_size, seq_len, device=device, dtype=pt.dtype).clamp(min=self.config.eps) # shape (B, N)
        scores = -torch.log(u) * weights.clamp(min=self.config.eps)
        scores = scores.masked_fill(~valid_mask, torch.inf)

        order = torch.argsort(scores, dim=1) # shape (B, N)
        sorted_pt = torch.gather(pt, dim=1, index=order)
        sorted_valid = torch.gather(valid_mask, dim=1, index=order)

        cum_pt = sorted_pt.cumsum(dim=1) # shape (B, N)
        cum_pt_before = cum_pt - sorted_pt

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        max_drop_nodes = (num_valid - self.config.min_nodes).clamp(min=0).unsqueeze(1) # shape (B, 1)

        # Include the node that crosses the threshold, matching the old
        # while-loop behavior: sample a node, add its pt, then stop if the
        # cumulative pt has exceeded the target.
        drop_sorted = (
            sorted_valid
            & can_drop_event.unsqueeze(1)
            & (positions < max_drop_nodes)
            & (cum_pt_before < target_drop_pt.unsqueeze(1))
        )

        drop_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        drop_mask.scatter_(dim=1, index=order, src=drop_sorted)

        view_mask = padding_mask | drop_mask

        if self.config.zero_dropped_features:
            view_x = x.masked_fill(drop_mask.unsqueeze(-1), 0.0)
        else:
            view_x = x.clone()

        return view_x, view_mask
    
    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_types: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[str]]:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B, N, F), got {tuple(x.shape)}.")

        batch_size, seq_len, _ = x.shape
        device = x.device

        if padding_mask is None:
            padding_mask = torch.zeros(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=device,
            )
        else:
            padding_mask = padding_mask.bool()

        views: List[torch.Tensor] = []
        view_padding_masks: List[torch.Tensor] = []
        view_types: List[str] = []

        for _ in range(self.config.num_global_views):
            view_x, view_mask = self._drop_nodes(
                x=x,
                padding_mask=padding_mask,
                drop_frac_range=self.config.global_drop_pt_frac_range,
            )
            views.append(view_x)
            view_padding_masks.append(view_mask)
            if return_types:
                view_types.append("global")

        for _ in range(self.config.num_local_views):
            view_x, view_mask = self._drop_nodes(
                x=x,
                padding_mask=padding_mask,
                drop_frac_range=self.config.local_drop_pt_frac_range,
            )
            views.append(view_x)
            view_padding_masks.append(view_mask)
            if return_types:
                view_types.append("local")

        return views, view_padding_masks, view_types

    
# ------------------------------------------------------------
# Corrupted negative view augmentation
# ------------------------------------------------------------

class CorruptedNegativeAugmentation(nn.Module):
    """
    Generate invalid/corrupted negative views for triplet training.

    Input:
        x: (B, N, F)
        padding_mask: optional bool tensor of shape (B, N)

    Output:
        negative_views:
            List of tensors, each with shape (B, N, F)

        negative_padding_masks:
            List of bool masks, each with shape (B, N)

        negative_types:
            List[List[str]]. Outer length is num_negative_views; each inner
            list has length B and stores the independently sampled corruption
            mode for every event in that negative view.

    Implemented corruption modes:
        identity_shuffle:
            Within each event, jointly shuffle charge and the five particle
            identity indicators.

        pt_resample:
            Within each event, resample part_pt values with replacement from
            valid nodes, then optionally renormalize the valid-node pt sum to
            the original event's pt sum.

        node_deta_dphi_rotation:
            Treat each valid node's (part_deta, part_dphi) as a 2D vector and
            independently rotate it while preserving its radial distance.

        deta_dphi_shuffle:
            Within each event, jointly shuffle the (part_deta, part_dphi) block
            across valid nodes.

        batch_mix:
            Build a fake jet by concatenating valid nodes from an anchor event
            and a donor event. Relative coordinates are copied with each node,
            and the resulting pt sum can be renormalized to the anchor event.
    """

    def __init__(self, config: CorruptedNegativeAugmentationConfig):
        super().__init__()
        self.config = config

        self._mode_names = (
            "batch_mix",
            "pt_resample",
            "node_deta_dphi_rotation",
            "deta_dphi_shuffle",
            "identity_shuffle",
        )

        mode_probabilities = torch.tensor(
            [
                config.batch_mix_prob,
                config.pt_resample_prob,
                config.node_deta_dphi_rotation_prob,
                config.deta_dphi_shuffle_prob,
                config.identity_shuffle_prob,
            ],
            dtype=torch.float32,
        )

        if torch.any(mode_probabilities < 0):
            raise ValueError(
                f"Corruption probabilities must be non-negative, got {mode_probabilities.tolist()}."
            )

        probability_sum = float(mode_probabilities.sum().item())
        if not math.isclose(probability_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            mode_probabilities /= probability_sum

        self.register_buffer(
            "_mode_probabilities",
            mode_probabilities,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_types: bool = False,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[List[str]]]:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape (B, N, F), got {tuple(x.shape)}.")

        batch_size, seq_len, _ = x.shape
        device = x.device

        if padding_mask is None:
            padding_mask = torch.zeros(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=device,
            )
        else:
            padding_mask = padding_mask.bool()

        negative_views: List[torch.Tensor] = []
        negative_padding_masks: List[torch.Tensor] = []
        negative_types: List[List[str]] = []

        # Independently sample one corruption mode for every event and every
        # negative view. Shape: (B, K), where K = num_negative_views.
        sampled_mode_indices = torch.multinomial(
            self._mode_probabilities,
            num_samples=batch_size * self.config.num_negative_views,
            replacement=True,
        ).view(batch_size, self.config.num_negative_views)

        for view_index in range(self.config.num_negative_views):
            event_mode_indices = sampled_mode_indices[:, view_index]  # (B,)

            # Start from the uncorrupted batch, then overwrite only the rows
            # assigned to each corruption mode. This keeps the mode sampling
            # event-specific while avoiding Python loops over individual events.
            neg_x = x.clone()
            neg_mask = padding_mask.clone()

            for mode_index, mode in enumerate(self._mode_names):
                event_selector = event_mode_indices == mode_index  # (B,)
                if not torch.any(event_selector):
                    continue
                
                if mode == "batch_mix":
                    candidate_x, candidate_mask = self._batch_mix(x, padding_mask)
                    neg_x = torch.where(
                        event_selector.view(batch_size, 1, 1),
                        candidate_x,
                        neg_x,
                    )
                    neg_mask = torch.where(
                        event_selector.view(batch_size, 1),
                        candidate_mask,
                        neg_mask,
                    )
                else:
                    selected_x = x[event_selector]
                    selected_mask = padding_mask[event_selector]
                    if mode == "identity_shuffle":
                        candidate_x, candidate_mask = self._identity_shuffle(selected_x, selected_mask)
                    elif mode == "pt_resample":
                        candidate_x, candidate_mask = self._pt_resample(selected_x, selected_mask)
                    elif mode == "node_deta_dphi_rotation":
                        candidate_x, candidate_mask = self._node_deta_dphi_rotation(selected_x, selected_mask)
                    elif mode == "deta_dphi_shuffle":
                        candidate_x, candidate_mask = self._deta_dphi_shuffle(selected_x, selected_mask)
                    else:
                        raise RuntimeError(f"Unexpected negative corruption mode: {mode}")

                    neg_x[event_selector] = candidate_x
                    neg_mask[event_selector] = candidate_mask

            negative_views.append(neg_x)
            negative_padding_masks.append(neg_mask)
            if return_types:
                negative_types.append(
                    [self._mode_names[index] for index in event_mode_indices.tolist()]
                )

        return negative_views, negative_padding_masks, negative_types

    def _valid_mask(self, padding_mask: torch.Tensor) -> torch.Tensor:
        return ~padding_mask.bool()

    def _valid_order(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Return valid node indices first, padded/invalid positions last.

        Output:
            order: (B, N)
        """

        batch_size, seq_len = valid_mask.shape
        device = valid_mask.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        sortable = torch.where(valid_mask, positions, seq_len + positions)
        return torch.argsort(sortable, dim=1)

    def _random_valid_order(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Return a random permutation of valid node indices for every event.
        Invalid/padded positions are sorted to the end.

        Output:
            order: (B, N)
        """

        scores = torch.rand(valid_mask.shape, device=valid_mask.device)
        scores = scores.masked_fill(~valid_mask, torch.inf)
        return torch.argsort(scores, dim=1)

    def _target_mask(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Vectorized replacement for per-event target-node sampling.

        Selects roughly corrupt_node_frac of valid nodes in each event. Events
        with <=1 valid node are left unmodified.

        Output:
            target_mask: (B, N), bool
        """

        batch_size, seq_len = valid_mask.shape
        device = valid_mask.device

        num_valid = valid_mask.sum(dim=1)
        num_target = torch.round(self.config.corrupt_node_frac * num_valid.float()).long()
        num_target = num_target.clamp(min=1)
        num_target = torch.minimum(num_target, num_valid)
        num_target = torch.where(num_valid > 1, num_target, torch.zeros_like(num_target))

        order = self._random_valid_order(valid_mask)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        target_sorted = positions < num_target.unsqueeze(1)

        target_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        target_mask.scatter_(dim=1, index=order, src=target_sorted)
        return target_mask & valid_mask

    def _permuted_source_index_for_each_valid_node(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        For every valid target node, return a source node index from a random
        same-event valid-node permutation.

        Output:
            source_idx: (B, N)
        """

        random_order = self._random_valid_order(valid_mask)
        valid_rank = valid_mask.long().cumsum(dim=1) - 1
        valid_rank = valid_rank.clamp(min=0)
        source_idx = torch.gather(random_order, dim=1, index=valid_rank)
        return source_idx

    def _sample_valid_source_index_with_replacement(self, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        For every target node, sample a same-event valid source node with
        replacement. This is used by pt_resample.

        Output:
            source_idx: (B, N)
        """

        batch_size, seq_len = valid_mask.shape
        device = valid_mask.device

        valid_order = self._valid_order(valid_mask)
        num_valid = valid_mask.sum(dim=1).clamp(min=1)

        random_rank = torch.floor(
            torch.rand(batch_size, seq_len, device=device) * num_valid.unsqueeze(1).float()
        ).long()
        random_rank = random_rank.clamp(min=0, max=seq_len - 1)

        source_idx = torch.gather(valid_order, dim=1, index=random_rank)
        return source_idx

    def _valid_mean_std(
        self,
        values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-event mean/std over valid nodes.

        Input:
            values: (B, N)
            valid_mask: (B, N)

        Output:
            mean: (B, 1)
            std: (B, 1), population std
        """

        count = valid_mask.sum(dim=1, keepdim=True).clamp(min=1).to(values.dtype)
        masked_values = values.masked_fill(~valid_mask, 0.0)
        mean = masked_values.sum(dim=1, keepdim=True) / count

        centered = (values - mean).masked_fill(~valid_mask, 0.0)
        var = centered.square().sum(dim=1, keepdim=True) / count
        std = torch.sqrt(var.clamp(min=0.0))

        return mean, std

    def _sync_log_pt_fraction(
        self,
        view_x: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute log(pt fraction) after a pt-changing corruption."""

        view_x = view_x.clone()
        pt = view_x[..., self.config.pt_index].clamp(min=self.config.eps)
        log_pt = torch.log(pt)
        view_x[..., self.config.log_pt_fraction_index] = torch.where(
            valid_mask,
            log_pt.to(dtype=view_x.dtype),
            torch.zeros_like(view_x[..., self.config.log_pt_fraction_index]),
        )
        return view_x

    def _renormalize_pt(
        self,
        view_x: torch.Tensor,
        valid_mask: torch.Tensor,
        target_pt_sum: torch.Tensor,
    ) -> torch.Tensor:
        """Match each event's valid-node pt sum to the requested target sum."""

        if not self.config.renormalize_pt_sum:
            return self._sync_log_pt_fraction(view_x, valid_mask)

        view_x = view_x.clone()
        valid_mask = valid_mask.bool()

        pt = view_x[..., self.config.pt_index].clamp(min=self.config.eps)
        current_sum = pt.masked_fill(~valid_mask, 0.0).sum(dim=1, keepdim=True)
        scale = target_pt_sum / current_sum.clamp(min=self.config.eps)
        matched_pt = pt * scale

        view_x[..., self.config.pt_index] = torch.where(
            valid_mask,
            matched_pt,
            view_x[..., self.config.pt_index],
        )

        return self._sync_log_pt_fraction(view_x, valid_mask)

    def _identity_shuffle(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        view_x = x.clone()
        view_mask = padding_mask.clone()

        valid_mask = self._valid_mask(padding_mask)
        target_mask = self._target_mask(valid_mask)
        source_idx = self._permuted_source_index_for_each_valid_node(valid_mask)

        source_x = torch.gather(
            x,
            dim=1,
            index=source_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)),
        )

        block_idx = torch.tensor(
            [self.config.charge_index]
            + list(
                range(
                    self.config.identity_start_index,
                    self.config.identity_end_index,
                )
            ),
            device=x.device,
            dtype=torch.long,
        )

        updated_block = torch.where(
            target_mask.unsqueeze(-1),
            source_x[..., block_idx],
            view_x[..., block_idx],
        )
        view_x[..., block_idx] = updated_block

        return view_x, view_mask

    def _pt_resample(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        view_x = x.clone()
        view_mask = padding_mask.clone()

        valid_mask = self._valid_mask(padding_mask)
        target_mask = self._target_mask(valid_mask)
        source_idx = self._sample_valid_source_index_with_replacement(valid_mask)

        original_pt_sum = x[..., self.config.pt_index].clamp(min=0.0).masked_fill(
            ~valid_mask,
            0.0,
        ).sum(dim=1, keepdim=True)

        source_x = torch.gather(
            x,
            dim=1,
            index=source_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)),
        )

        # Resample only part_pt.
        sampled_pt = torch.where(
            target_mask,
            source_x[..., self.config.pt_index],
            view_x[..., self.config.pt_index],
        )
        view_x[..., self.config.pt_index] = sampled_pt

        can_renormalize = torch.isfinite(original_pt_sum) & (original_pt_sum > 0)
        view_x = self._renormalize_pt(
            view_x=view_x,
            valid_mask=valid_mask & can_renormalize,
            target_pt_sum=original_pt_sum.clamp(min=self.config.eps),
        )

        return view_x, view_mask

    def _deta_dphi_shuffle(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Jointly shuffle deta and dphi across valid same-event nodes.
        """

        view_x = x.clone()
        view_mask = padding_mask.clone()

        valid_mask = self._valid_mask(padding_mask)
        target_mask = self._target_mask(valid_mask)
        source_idx = self._permuted_source_index_for_each_valid_node(valid_mask)

        source_x = torch.gather(
            x,
            dim=1,
            index=source_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)),
        )

        block_idx = torch.tensor(
            [
                self.config.deta_index,
                self.config.dphi_index,
            ],
            device=x.device,
            dtype=torch.long,
        )

        updated_block = torch.where(
            target_mask.unsqueeze(-1),
            source_x[..., block_idx],
            view_x[..., block_idx],
        )
        view_x[..., block_idx] = updated_block

        return view_x, view_mask

    def _batch_mix(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, feature_dim = x.shape
        device = x.device

        if batch_size <= 1:
            return x.clone(), padding_mask.clone()

        valid_mask = self._valid_mask(padding_mask)
        num_valid = valid_mask.sum(dim=1) # shape: (B,)

        donor_b = torch.randint(
            low=0,
            high=batch_size - 1,
            size=(batch_size,),
            device=device,
        ) # shape: (B,)
        batch_indices = torch.arange(batch_size, device=device)
        donor_b = donor_b + (donor_b >= batch_indices).long() # index-shift to make sure donor is not myself

        donor_valid_mask = valid_mask[donor_b]
        donor_num_valid = donor_valid_mask.sum(dim=1)

        original_pt_sum = x[..., self.config.pt_index].clamp(min=0.0).masked_fill(
            ~valid_mask,
            0.0,
        ).sum(dim=1, keepdim=True)

        # Randomly sample anchor frac 
        batch_mix_anchor_frac = torch.rand(batch_size, device=device, dtype=x.dtype)
        batch_mix_anchor_frac = (self.config.batch_mix_anchor_frac_max - self.config.batch_mix_anchor_frac_min) * batch_mix_anchor_frac + self.config.batch_mix_anchor_frac_min
        num_anchor_keep = torch.round(
            batch_mix_anchor_frac * num_valid.to(x.dtype)
        ).long()
        num_anchor_keep = num_anchor_keep.clamp(min=1)
        num_anchor_keep = torch.minimum(num_anchor_keep, num_valid)

        num_donor_keep = (seq_len - num_anchor_keep).clamp(min=0)
        num_donor_keep = torch.minimum(num_donor_keep, donor_num_valid)

        total_out = num_anchor_keep + num_donor_keep
        fallback = (
            (num_valid < self.config.min_nodes)
            | (donor_num_valid == 0)
            | (total_out < self.config.min_nodes)
            | (~torch.isfinite(original_pt_sum.squeeze(1)))
            | (original_pt_sum.squeeze(1) <= 0)
        ) # shape: (B,)

        anchor_order = self._random_valid_order(valid_mask)
        donor_order_all = self._random_valid_order(valid_mask)
        donor_order = donor_order_all[donor_b]

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        is_anchor = positions < num_anchor_keep.unsqueeze(1)
        is_valid_out = positions < total_out.unsqueeze(1)

        anchor_rel = positions
        donor_rel = (positions - num_anchor_keep.unsqueeze(1)).clamp(min=0, max=seq_len - 1)

        anchor_source_idx = torch.gather(anchor_order, dim=1, index=anchor_rel)
        donor_source_idx = torch.gather(donor_order, dim=1, index=donor_rel)

        source_batch = torch.where(
            is_anchor,
            batch_indices.unsqueeze(1).expand(batch_size, seq_len),
            donor_b.unsqueeze(1).expand(batch_size, seq_len),
        )
        source_idx = torch.where(is_anchor, anchor_source_idx, donor_source_idx)

        mixed_x = x[source_batch, source_idx]
        mixed_x = torch.where(is_valid_out.unsqueeze(-1), mixed_x, torch.zeros_like(mixed_x))
        mixed_mask = ~is_valid_out

        # For rows that cannot be mixed safely, keep the original event.
        view_x = torch.where(fallback.view(batch_size, 1, 1), x, mixed_x)
        view_mask = torch.where(fallback.view(batch_size, 1), padding_mask, mixed_mask)

        view_valid_mask = ~view_mask
        active_mixed_mask = view_valid_mask & (~fallback).unsqueeze(1)


        view_x = self._renormalize_pt(
            view_x=view_x,
            valid_mask=active_mixed_mask,
            target_pt_sum=original_pt_sum.clamp(min=self.config.eps),
        )

        return view_x, view_mask
    
    def _node_deta_dphi_rotation(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Independently rotate every valid node in the relative deta-dphi plane.

        The radius sqrt(deta^2 + dphi^2) is preserved while the polar angle is
        resampled independently for each valid node.
        """

        view_x = x.clone()
        view_mask = padding_mask.clone()

        valid_mask = self._valid_mask(padding_mask)

        deta = x[..., self.config.deta_index].float()
        dphi = x[..., self.config.dphi_index].float()

        radius = torch.sqrt(deta.square() + dphi.square())
        angle = torch.atan2(dphi, deta)

        theta = torch.rand(
            deta.shape,
            device=x.device,
            dtype=torch.float32,
        ) * (2.0 * math.pi)

        rotated_angle = angle + theta
        rotated_deta = radius * torch.cos(rotated_angle)
        rotated_dphi = radius * torch.sin(rotated_angle)

        view_x[..., self.config.deta_index] = torch.where(
            valid_mask,
            rotated_deta.to(dtype=x.dtype),
            x[..., self.config.deta_index],
        )
        view_x[..., self.config.dphi_index] = torch.where(
            valid_mask,
            rotated_dphi.to(dtype=x.dtype),
            x[..., self.config.dphi_index],
        )


        return view_x, view_mask


# ------------------------------------------------------------
# LeJEPA-style loss
# ------------------------------------------------------------

@dataclass
class LeJEPALossConfig:
    """
    Configuration for LeJEPA-style representation loss.

    The total loss is:

        total_loss = invariant_weight * invariant_loss
                   + sigreg_weight * sigreg_loss

    The default SIGReg weight follows the LeJEPA lambda setting:

        sigreg_weight = 0.02

    invariant_loss:
        Encourages different views of the same event to have similar
        representations.

    sigreg_loss:
        Applies LeJEPA SIGReg to all representations from all views mixed
        together.

    The SIGReg part follows the LeJEPA usage pattern:

        import lejepa

        univariate_test = lejepa.univariate.EppsPulley(n_points=17)

        loss_fn = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=1024,
        )

        sigreg_loss = loss_fn(embeddings)

    where embeddings has shape:

        (num_samples, representation_dim)
    """

    invariant_weight: float = 1.0
    sigreg_weight: float = 0.02

    epps_pulley_num_points: int = 17
    num_slices: int = 1024


# ------------------------------------------------------------
# Triplet loss config
# ------------------------------------------------------------

@dataclass
class TripletLossConfig:
    """
    Configuration for adding a multi-negative triplet objective on top of
    LeJEPA invariant loss and SIGReg.

    Anchor:
        Mean of global-view representations, same as LeJEPA invariant anchor.

    Positives:
        By default, all global views.

    Negatives:
        Corrupted negative views generated by CorruptedNegativeAugmentation.

    The triplet loss uses all positive-negative pairs:

        mean ReLU(d(anchor, positive) - d(anchor, negative) + margin)
    """

    triplet_weight: float = 0.1
    triplet_margin: float = 1.0
    normalize_representations_for_triplet: bool = False
    use_global_views_as_positives: bool = True

# ------------------------------------------------------------
# Mahalanobis negative loss config
# ------------------------------------------------------------

@dataclass
class MahalanobisNegativeLossConfig:
    """
    Configuration for LeJEPA + SIGReg + distributional negative exclusion.

    Normal reference distribution:
        Estimated online from the per-event global-view mean representations:

            anchor_b = mean_{v in global views} z_{v,b}

        These anchors are detached before updating running statistics.

    Running statistics:
        EMA mean:

            mu_t = beta * mu_{t-1}
                 + (1 - beta) * batch_mean_t

        EMA raw second moment:

            M_t = beta * M_{t-1}
                + (1 - beta) * E_batch[z z^T]

        Covariance:

            Sigma_t = M_t - mu_t mu_t^T

    Negative objective:
        Corrupted negative representations are assigned a Mahalanobis radius:

            D_M(z^-)
                = sqrt(
                    (z^- - mu)^T
                    Sigma^+
                    (z^- - mu)
                  )

        The loss pushes negatives outside a target Mahalanobis radius R:

            L_maha
                = mean(
                    ReLU(R - D_M(z^-))^2
                  )

        Therefore:
            D_M(z^-) < R:
                negative is still too close to the normal distribution and
                receives an outward gradient.

            D_M(z^-) >= R:
                loss is zero and the negative is no longer pushed farther.

    Important:
        Running statistics are detached state. Gradients flow only through
        negative representations, not through the EMA mean/covariance.
    """

    mahalanobis_weight: float = 0.1

    # EMA decay for both first and raw second moments.
    ema_decay: float = 0.99

    # Target Mahalanobis radius for corrupted negative samples.
    # Negatives beyond this radius receive zero Mahalanobis loss.
    target_radius: float = 5.0

class LeJEPASIGRegLoss(nn.Module):
    """
    LeJEPA-style loss module.

    Submodule role:
        Used inside LeJEPAParticleTransformerRepresentation as:

            self.loss

    Input:
        z_views:
            Tensor of shape (V, B, D)

            V: number of views
            B: batch size
            D: representation dimension

    Output:
        Dictionary containing:
            total_loss
            invariant_loss
            sigreg_loss

    Invariant loss formula:
        For event b and view v, let z_{v,b} be its representation.

        The anchor is the mean representation of the global views only:

            zbar_global_b = (1 / G) sum_{v in global} z_{v,b}

        The invariant loss pulls every view, including both global and local
        views, toward this global-view anchor:

            L_inv = mean_{v,b} || z_{v,b} - zbar_global_b ||_2^2

        In code, this is implemented as mean squared error over all
        dimensions, views, and batch samples.

    SIGReg:
        All global and local view representations are mixed into one matrix:

            embeddings = z_views.reshape(V * B, D)

        Then SIGReg is computed on this full mixed set.
    """

    def __init__(self, config: LeJEPALossConfig):
        super().__init__()
        self.config = config

        univariate_test = lejepa.univariate.EppsPulley(
            n_points=config.epps_pulley_num_points
        )

        self.sigreg_loss_fn = lejepa.multivariate.SlicingUnivariateTest(
            univariate_test=univariate_test,
            num_slices=config.num_slices,
        )

    def forward(self, z_views: torch.Tensor, num_global_views: int) -> Dict[str, torch.Tensor]:
        if z_views.ndim != 3:
            raise ValueError(
                f"Expected z_views shape (V, B, D), got {tuple(z_views.shape)}."
            )

        if not (1 <= num_global_views <= z_views.size(0)):
            raise ValueError(
                f"Expected num_global_views to be in [1, {z_views.size(0)}], "
                f"got {num_global_views}."
            )
            
        # Compute SSL/statistical losses in fp32. bf16/fp16 is useful for the encoder
        # forward pass, but SIGReg and distance losses are numerically sensitive.
        z_views = z_views.float()

        invariant_z = z_views

        global_anchor = invariant_z[:num_global_views].mean(dim=0, keepdim=True)
        invariant_loss = (invariant_z - global_anchor).pow(2).mean()

        sigreg_z = z_views

        num_views, batch_size, dim = sigreg_z.shape

        # Mix every representation from every global/local view together.
        sigreg_embeddings = sigreg_z.reshape(num_views * batch_size, dim)

        sigreg_loss = self.sigreg_loss_fn(sigreg_embeddings)

        total_loss = (
            self.config.invariant_weight * invariant_loss
            + self.config.sigreg_weight * sigreg_loss
        )

        return {
            "total_loss": total_loss,
            "invariant_loss": invariant_loss,
            "sigreg_loss": sigreg_loss,
        }


# ------------------------------------------------------------
# LeJEPA + auxiliary classification loss
# ------------------------------------------------------------

class LeJEPASIGRegClassificationLoss(LeJEPASIGRegLoss):
    """LeJEPA invariant + SIGReg with an auxiliary categorical classifier."""

    def __init__(
        self,
        lejepa_config: LeJEPALossConfig,
        semi_supervised_config,
    ):
        super().__init__(lejepa_config)
        self.semi_supervised_config = semi_supervised_config

    def forward(
        self,
        z_views: torch.Tensor,
        classification_logits: torch.Tensor,
        y: torch.Tensor,
        num_global_views: int,
    ) -> Dict[str, torch.Tensor]:
        base_loss = super().forward(
            z_views=z_views,
            num_global_views=num_global_views,
        )

        if classification_logits.ndim != 2:
            raise ValueError(
                "Expected classification_logits shape (B, C), got "
                f"{tuple(classification_logits.shape)}."
            )
        if y.ndim != 2:
            raise ValueError(
                f"Expected one-hot y shape (B, C), got {tuple(y.shape)}."
            )
        if classification_logits.shape != y.shape:
            raise ValueError(
                "classification_logits and y must have the same shape, got "
                f"{tuple(classification_logits.shape)} and {tuple(y.shape)}."
            )

        # JetClass labels are mutually exclusive one-hot targets. Convert them
        # to class indices and use categorical cross entropy.
        target_class = y.argmax(dim=-1).long()
        classification_loss = F.cross_entropy(
            classification_logits.float(),
            target_class,
        )

        total_loss = (
            base_loss["total_loss"]
            + self.semi_supervised_config.classification_weight
            * classification_loss
        )

        return {
            **base_loss,
            "classification_loss": classification_loss,
            "total_loss": total_loss,
        }


# ------------------------------------------------------------
# LeJEPA + SIGReg + triplet loss
# ------------------------------------------------------------

class LeJEPASIGRegTripletLoss(nn.Module):
    """
    LeJEPA invariant + SIGReg loss with an additional multi-negative triplet loss.

    Anchor:
        Mean of global views, identical to the LeJEPA invariant anchor.

    Positives:
        Global views by default.

    Negatives:
        Corrupted views generated from the same input batch.

    Input:
        z_views: (V, B, D)
        z_negatives: (K, B, D)

    Output:
        Dictionary containing total_loss, invariant_loss, sigreg_loss,
        triplet_loss, and diagnostic positive/negative distances.
    """

    def __init__(
        self,
        lejepa_config: LeJEPALossConfig,
        triplet_config: TripletLossConfig,
    ):
        super().__init__()
        self.lejepa_loss = LeJEPASIGRegLoss(lejepa_config)
        self.triplet_config = triplet_config

    def forward(
        self,
        z_views: torch.Tensor,
        z_negatives: torch.Tensor,
        num_global_views: int,
    ) -> Dict[str, torch.Tensor]:
        if z_views.ndim != 3:
            raise ValueError(
                f"Expected z_views shape (V, B, D), got {tuple(z_views.shape)}."
            )
        if z_negatives.ndim != 3:
            raise ValueError(
                f"Expected z_negatives shape (K, B, D), got {tuple(z_negatives.shape)}."
            )
        if z_views.size(1) != z_negatives.size(1):
            raise ValueError(
                f"Batch size mismatch: z_views has B={z_views.size(1)}, "
                f"z_negatives has B={z_negatives.size(1)}."
            )
        if z_views.size(2) != z_negatives.size(2):
            raise ValueError(
                f"Representation dimension mismatch: z_views has D={z_views.size(2)}, "
                f"z_negatives has D={z_negatives.size(2)}."
            )
        
        # Compute loss terms in fp32 even when the encoder forward pass uses autocast.
        # This avoids reduced-precision quantization in SIGReg and triplet distances.
        z_views = z_views.float()
        z_negatives = z_negatives.float()

        base_loss = self.lejepa_loss(
            z_views=z_views,
            num_global_views=num_global_views,
        )

        triplet_views = z_views
        triplet_negatives = z_negatives
        if self.triplet_config.normalize_representations_for_triplet:
            triplet_views = F.normalize(triplet_views, p=2, dim=-1)
            triplet_negatives = F.normalize(triplet_negatives, p=2, dim=-1)

        anchor = triplet_views[:num_global_views].mean(dim=0)

        if self.triplet_config.use_global_views_as_positives:
            positives = triplet_views[:num_global_views]
        else:
            positives = triplet_views

        negatives = triplet_negatives

        # Shapes:
        #   anchor:    (B, D)
        #   positives: (P, B, D)
        #   negatives: (K, B, D)
        #
        # Distances are mean squared distances over representation dimension.
        d_pos = (positives - anchor.unsqueeze(0)).pow(2).mean(dim=-1)  # (P, B)
        d_neg = (negatives - anchor.unsqueeze(0)).pow(2).mean(dim=-1)  # (K, B)

        triplet_terms = F.relu(
            d_pos.unsqueeze(0)
            - d_neg.unsqueeze(1)
            + self.triplet_config.triplet_margin
        )  # (K, P, B)
        triplet_loss = triplet_terms.mean()

        total_loss = base_loss["total_loss"] + self.triplet_config.triplet_weight * triplet_loss

        output = {
            **base_loss,
            "triplet_loss": triplet_loss,
            "triplet_pos_distance": d_pos.mean(),
            "triplet_neg_distance": d_neg.mean(),
            "total_loss": total_loss,
        }

        return output

# ------------------------------------------------------------
# Main SSL model
# ------------------------------------------------------------

class LeJEPAParticleTransformerRepresentation(MinimalParticleTransformer):
    """
    Minimal ParticleTransformer with LeJEPA-style SSL training wrapper.

    Parent class:
        MinimalParticleTransformer

    Added members:
        self.augmentation:
            MultiViewAugmentation

        self.loss:
            LeJEPASIGRegLoss

    Main use:
        During pretraining, call:

            output = model.forward_pretrain(x, padding_mask)
            loss = output["total_loss"]

    During representation extraction, call the inherited forward method:

            z = model(x, padding_mask)

    This keeps the representation model and SSL pretraining logic together,
    while preserving the clean single-view encoder API from the parent class.
    """

    def __init__(
        self,
        model_config: ParticleTransformerConfig,
        augmentation_config: Optional[MultiViewAugmentationConfig] = None,
        loss_config: Optional[LeJEPALossConfig] = None,
    ):
        super().__init__(model_config)

        if augmentation_config is None:
            augmentation_config = MultiViewAugmentationConfig()

        if loss_config is None:
            loss_config = LeJEPALossConfig()

        self.augmentation = MultiViewAugmentation(augmentation_config)
        self.loss = LeJEPASIGRegLoss(loss_config)
        self.representation_head = RepresentationHead(
            embed_dim=model_config.embed_dim,
            representation_dim=model_config.representation_dim,
            dropout=model_config.dropout,
        )

    def forward_pretrain(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_views: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Pretraining forward pass.

        Input:
            x:
                Node tensor with shape (B, N, F).
            
            y:
                Label tensor with shape (B, C) (not used).

            padding_mask:
                Optional bool tensor with shape (B, N).
                True means padded node.

            return_views:
                If True, also return augmented views and masks.
                This is useful for debugging augmentation behavior.

        Output:
            Dictionary with at least:
                total_loss
                invariant_loss
                sigreg_loss
                z_views

            z_views has shape:
                (num_views, B, representation_dim)

            The encoder is applied once to the concatenated view batch with
            shape (num_views * B, N, F), then reshaped back to separate views.
        """

        views, view_padding_masks, view_types = self.augmentation(
            x=x,
            padding_mask=padding_mask,
        )

        num_views = len(views)
        batch_size = x.size(0)

        batched_views = torch.cat(views, dim=0) # shape: (num_views * B, N, F)
        batched_view_masks = torch.cat(view_padding_masks, dim=0)

        batched_z = super().forward( # final CLS hidden states
            batched_views,
            padding_mask=batched_view_masks,
        )
        
        batched_z = self.representation_head(batched_z) # get representations

        z_views = batched_z.view(num_views, batch_size, -1)

        loss_dict = self.loss(
            z_views,
            num_global_views=self.augmentation.config.num_global_views,
        )

        output: Dict[str, torch.Tensor] = {
            **loss_dict,
            "z_views": z_views,
        }

        # This is not a tensor, so keep it separate if requested.
        # The return type annotation is kept simple for training usage.
        if return_views:
            output["views"] = views
            output["view_padding_masks"] = view_padding_masks
            output["view_types"] = view_types

        return output

# ------------------------------------------------------------
# LeJEPA + triplet SSL model
# ------------------------------------------------------------

class LeJEPATripletParticleTransformerRepresentation(MinimalParticleTransformer):
    """
    ParticleTransformer representation model with LeJEPA + corrupted-negative triplet training.

    This model uses the same backbone as LeJEPAParticleTransformerRepresentation:
        NodeEmbedding -> PairwiseAttentionBias -> ParticleTransformerEncoder
        -> CLSPooling -> RepresentationHead

    Pretraining objective:
        1. LeJEPA invariant loss:
            global-view mean anchor matched against all global/local views.

        2. SIGReg:
            all global/local view representations mixed together.

        3. Multi-negative triplet loss:
            anchor = global-view mean;
            positives = global views;
            negatives = corrupted views.

    During representation extraction, use the inherited forward method.
    """

    def __init__(
        self,
        model_config: ParticleTransformerConfig,
        augmentation_config: Optional[MultiViewAugmentationConfig] = None,
        negative_augmentation_config: Optional[CorruptedNegativeAugmentationConfig] = None,
        loss_config: Optional[LeJEPALossConfig] = None,
        triplet_loss_config: Optional[TripletLossConfig] = None,
    ):
        super().__init__(model_config)

        if augmentation_config is None:
            augmentation_config = MultiViewAugmentationConfig()
        if negative_augmentation_config is None:
            negative_augmentation_config = CorruptedNegativeAugmentationConfig()
        if loss_config is None:
            loss_config = LeJEPALossConfig()
        if triplet_loss_config is None:
            triplet_loss_config = TripletLossConfig()

        self.augmentation = MultiViewAugmentation(augmentation_config)
        self.negative_augmentation = CorruptedNegativeAugmentation(negative_augmentation_config)
        self.loss = LeJEPASIGRegTripletLoss(
            lejepa_config=loss_config,
            triplet_config=triplet_loss_config,
        )
        self.representation_head = RepresentationHead(
            embed_dim=model_config.embed_dim,
            representation_dim=model_config.representation_dim,
            dropout=model_config.dropout,
        )

    def forward_pretrain(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_views: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Pretraining forward pass with positive LeJEPA views and corrupted negatives.

        Input:
            x: (B, N, F)
            y: (B, C) labels for supervised loss (not used)
            padding_mask: optional bool tensor of shape (B, N)

        Output dictionary includes:
            total_loss
            invariant_loss
            sigreg_loss
            triplet_loss
            triplet_pos_distance
            triplet_neg_distance
            z_views: (V, B, D)
            z_negatives: (K, B, D)
        """
        with record_function("positive_augmentation"):
            views, view_padding_masks, view_types = self.augmentation(
                x=x,
                padding_mask=padding_mask,
                return_types=return_views,
            )
        
        with record_function("negative_augmentation"):
            negative_views, negative_padding_masks, negative_types = self.negative_augmentation(
                x=x,
                padding_mask=padding_mask,
                return_types=return_views,
            )

        num_views = len(views)
        num_negative_views = len(negative_views)
        batch_size = x.size(0)
        
        with record_function("batch_concat"):
            batched_views = torch.cat(views, dim=0)
            batched_view_masks = torch.cat(view_padding_masks, dim=0)

            batched_negatives = torch.cat(negative_views, dim=0)
            batched_negative_masks = torch.cat(negative_padding_masks, dim=0)

            all_inputs = torch.cat([batched_views, batched_negatives], dim=0) # shape: (num_views * B + num_negative_views * B, N, F)
            all_masks = torch.cat([batched_view_masks, batched_negative_masks], dim=0) # shape: (num_views * B + num_negative_views * B, N)
        
        with record_function("backbone_forward"):
            all_z = super().forward(
                all_inputs,
                padding_mask=all_masks,
            )
            all_z = self.representation_head(all_z) # get representations

        with record_function("reshape_representations"):
            z_views_flat = all_z[: num_views * batch_size]
            z_negatives_flat = all_z[num_views * batch_size :]

            z_views = z_views_flat.view(num_views, batch_size, -1)
            z_negatives = z_negatives_flat.view(num_negative_views, batch_size, -1)

        with record_function("ssl_loss"):
            loss_dict = self.loss(
                z_views=z_views,
                z_negatives=z_negatives,
                num_global_views=self.augmentation.config.num_global_views,
            )

        output: Dict[str, torch.Tensor] = {
            **loss_dict,
            "z_views": z_views,
            "z_negatives": z_negatives,
        }

        if return_views:
            output["views"] = views
            output["view_padding_masks"] = view_padding_masks
            output["view_types"] = view_types
            output["negative_views"] = negative_views
            output["negative_padding_masks"] = negative_padding_masks
            output["negative_types"] = negative_types

        return output


@dataclass
class SemiSupervisedLossConfig:
    classification_weight: float = 1.0
    num_classes: int = 10
    
# ------------------------------------------------------------
# LeJEPA + SIGReg + classification objective (semi-supervised)
# ------------------------------------------------------------

class LeJEPASemiSupervisedParticleTransformerRepresentation(
    LeJEPAParticleTransformerRepresentation
):
    """LeJEPA + SIGReg with an auxiliary JetClass classification objective."""

    def __init__(
        self,
        model_config: ParticleTransformerConfig,
        augmentation_config: MultiViewAugmentationConfig,
        loss_config: LeJEPALossConfig,
        semi_supervised_config: SemiSupervisedLossConfig,
    ):
        super().__init__(
            model_config=model_config,
            augmentation_config=augmentation_config,
            loss_config=loss_config,
        )
        self.semi_supervised_config = semi_supervised_config
        self.classification_head = CategoricalClassificationHead(
            embed_dim=model_config.embed_dim,
            num_classes=semi_supervised_config.num_classes,
            dropout=model_config.dropout,
        )
        self.loss = LeJEPASIGRegClassificationLoss(
            lejepa_config=loss_config,
            semi_supervised_config=semi_supervised_config,
        )
        self.representation_head = RepresentationHead(
            embed_dim=model_config.embed_dim,
            representation_dim=model_config.representation_dim,
            dropout=model_config.dropout,
        )

    def forward_pretrain(
        self,
        x_particles: torch.Tensor,
        y: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        expected_shape = (
            x_particles.size(0),
            self.semi_supervised_config.num_classes,
        )
        if tuple(y.shape) != expected_shape:
            raise ValueError(
                f"Expected one-hot y shape {expected_shape}, got {tuple(y.shape)}."
            )

        views, view_padding_masks, view_types = self.augmentation(
            x=x_particles,
            padding_mask=padding_mask,
            return_types=True,
        )
        z_views = torch.stack( # get final CLS hidden states
            [
                self(
                    view_x,
                    padding_mask=view_mask,
                )
                for view_x, view_mask in zip(views, view_padding_masks)
            ],
            dim=0,
        )
        
        # compute classification logits from final CLS hidden states of globals
        classification_logits = self.classification_head(z_views[: self.augmentation.config.num_global_views].mean(dim=0))
        # compute representations from final CLS hidden states of all views
        z_views = self.representation_head(z_views)
        
        losses = self.loss(
            z_views=z_views,
            classification_logits=classification_logits,
            y=y,
            num_global_views=self.augmentation.config.num_global_views,
        )

        return {
            **losses,
            "classification_logits": classification_logits,
            "z_views": z_views,
            "view_types": view_types,
        }

# ------------------------------------------------------------
# LeJEPA + SIGReg + triplet + classification objective
# ------------------------------------------------------------

class LeJEPASIGRegTripletClassificationLoss(nn.Module):
    """
    Combine LeJEPA invariant loss, SIGReg, corrupted-negative triplet loss,
    and auxiliary categorical background classification.
    """

    def __init__(
        self,
        lejepa_config: LeJEPALossConfig,
        triplet_config: TripletLossConfig,
        semi_supervised_config: SemiSupervisedLossConfig,
    ):
        super().__init__()
        self.triplet_loss = LeJEPASIGRegTripletLoss(
            lejepa_config=lejepa_config,
            triplet_config=triplet_config,
        )
        self.semi_supervised_config = semi_supervised_config

    def forward(
        self,
        z_views: torch.Tensor,
        z_negatives: torch.Tensor,
        classification_logits: torch.Tensor,
        y: torch.Tensor,
        num_global_views: int,
    ) -> Dict[str, torch.Tensor]:
        base_loss = self.triplet_loss(
            z_views=z_views,
            z_negatives=z_negatives,
            num_global_views=num_global_views,
        )

        if classification_logits.ndim != 2:
            raise ValueError(
                "Expected classification_logits shape (B, C), got "
                f"{tuple(classification_logits.shape)}."
            )
        if y.ndim != 2:
            raise ValueError(
                f"Expected one-hot y shape (B, C), got {tuple(y.shape)}."
            )
        if classification_logits.shape != y.shape:
            raise ValueError(
                "classification_logits and y must have the same shape, got "
                f"{tuple(classification_logits.shape)} and {tuple(y.shape)}."
            )

        target_class = y.argmax(dim=-1).long()
        classification_loss = F.cross_entropy(
            classification_logits.float(),
            target_class,
        )

        total_loss = (
            base_loss["total_loss"]
            + self.semi_supervised_config.classification_weight
            * classification_loss
        )

        return {
            **base_loss,
            "classification_loss": classification_loss,
            "total_loss": total_loss,
        }


class LeJEPASemiSupervisedTripletParticleTransformerRepresentation(
    LeJEPATripletParticleTransformerRepresentation
):
    """
    ParticleTransformer trained jointly with four objectives:

        1. LeJEPA invariant loss on positive global/local views;
        2. SIGReg on all positive-view representations;
        3. triplet loss against corrupted negative views;
        4. categorical classification of the selected background classes.

    The classification head reads the mean pre-projection CLS state of the
    global positive views. The triplet and LeJEPA losses use the shared
    representation head output, matching the existing semi-supervised and
    triplet model conventions.
    """

    def __init__(
        self,
        model_config: ParticleTransformerConfig,
        augmentation_config: Optional[MultiViewAugmentationConfig] = None,
        negative_augmentation_config: Optional[
            CorruptedNegativeAugmentationConfig
        ] = None,
        loss_config: Optional[LeJEPALossConfig] = None,
        triplet_loss_config: Optional[TripletLossConfig] = None,
        semi_supervised_config: Optional[SemiSupervisedLossConfig] = None,
    ):
        if semi_supervised_config is None:
            semi_supervised_config = SemiSupervisedLossConfig()

        super().__init__(
            model_config=model_config,
            augmentation_config=augmentation_config,
            negative_augmentation_config=negative_augmentation_config,
            loss_config=loss_config,
            triplet_loss_config=triplet_loss_config,
        )

        # The parent constructor has already resolved optional configs.
        resolved_loss_config = (
            loss_config if loss_config is not None else LeJEPALossConfig()
        )
        resolved_triplet_config = (
            triplet_loss_config
            if triplet_loss_config is not None
            else TripletLossConfig()
        )

        self.semi_supervised_config = semi_supervised_config
        self.classification_head = CategoricalClassificationHead(
            embed_dim=model_config.embed_dim,
            num_classes=semi_supervised_config.num_classes,
            dropout=model_config.dropout,
        )
        self.loss = LeJEPASIGRegTripletClassificationLoss(
            lejepa_config=resolved_loss_config,
            triplet_config=resolved_triplet_config,
            semi_supervised_config=semi_supervised_config,
        )

    def forward_pretrain(
        self,
        x_particles: torch.Tensor,
        y: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        return_views: bool = False,
    ) -> Dict[str, torch.Tensor]:
        expected_shape = (
            x_particles.size(0),
            self.semi_supervised_config.num_classes,
        )
        if tuple(y.shape) != expected_shape:
            raise ValueError(
                f"Expected one-hot y shape {expected_shape}, got {tuple(y.shape)}."
            )

        with record_function("positive_augmentation"):
            views, view_padding_masks, view_types = self.augmentation(
                x=x_particles,
                padding_mask=padding_mask,
                return_types=return_views,
            )

        with record_function("negative_augmentation"):
            (
                negative_views,
                negative_padding_masks,
                negative_types,
            ) = self.negative_augmentation(
                x=x_particles,
                padding_mask=padding_mask,
                return_types=return_views,
            )

        num_views = len(views)
        num_negative_views = len(negative_views)
        batch_size = x_particles.size(0)
        num_global_views = self.augmentation.config.num_global_views

        with record_function("batch_concat"):
            batched_views = torch.cat(views, dim=0)
            batched_view_masks = torch.cat(view_padding_masks, dim=0)
            batched_negatives = torch.cat(negative_views, dim=0)
            batched_negative_masks = torch.cat(
                negative_padding_masks,
                dim=0,
            )

            all_inputs = torch.cat(
                [batched_views, batched_negatives],
                dim=0,
            )
            all_masks = torch.cat(
                [batched_view_masks, batched_negative_masks],
                dim=0,
            )

        with record_function("backbone_forward"):
            all_cls = MinimalParticleTransformer.forward(
                self,
                all_inputs,
                padding_mask=all_masks,
            )

        with record_function("classification_forward"):
            positive_cls_flat = all_cls[: num_views * batch_size]
            positive_cls = positive_cls_flat.view(
                num_views,
                batch_size,
                -1,
            )
            global_cls_anchor = positive_cls[:num_global_views].mean(dim=0)
            classification_logits = self.classification_head(
                global_cls_anchor
            )

        with record_function("representation_projection"):
            all_z = self.representation_head(all_cls)

        with record_function("reshape_representations"):
            z_views_flat = all_z[: num_views * batch_size]
            z_negatives_flat = all_z[num_views * batch_size :]

            z_views = z_views_flat.view(num_views, batch_size, -1)
            z_negatives = z_negatives_flat.view(
                num_negative_views,
                batch_size,
                -1,
            )

        with record_function("joint_ssl_loss"):
            loss_dict = self.loss(
                z_views=z_views,
                z_negatives=z_negatives,
                classification_logits=classification_logits,
                y=y,
                num_global_views=num_global_views,
            )

        output: Dict[str, torch.Tensor] = {
            **loss_dict,
            "classification_logits": classification_logits,
            "z_views": z_views,
            "z_negatives": z_negatives,
        }

        if return_views:
            output["views"] = views
            output["view_padding_masks"] = view_padding_masks
            output["view_types"] = view_types
            output["negative_views"] = negative_views
            output["negative_padding_masks"] = negative_padding_masks
            output["negative_types"] = negative_types

        return output

