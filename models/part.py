import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
import lejepa

import torch
import torch.nn as nn
import torch.nn.functional as F

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

augmentation_config = PtDropAugmentationConfig(
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
    """
    Configuration object for the minimal particle transformer.

    This model assumes the input tensor has shape:

        x: (B, N, F)

    where each node has features ordered as:

        [eta, phi, pt, d0/d0Err, dz/dzErr, mass, charge]

    The model produces a representation vector instead of class logits.
    """

    input_dim: int = 7
    embed_dim: int = 128
    num_heads: int = 8
    num_layers: int = 4
    ffn_mult: int = 4
    dropout: float = 0.1
    representation_dim: int = 128

    use_pairwise_bias: bool = True
    pairwise_hidden_dim: int = 64
    pairwise_num_features: int = 4

    compute_dtype: torch.dtype = torch.bfloat16
    use_internal_autocast: bool = False

    eps: float = 1e-8


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


def build_four_vector_from_eta_phi_pt_mass(
    x: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Build four-vectors from node features.

    Input:
        x: (B, N, F)
        feature order:
            0: eta
            1: phi
            2: pt
            5: mass

    Output:
        p4: (B, N, 4)
        order:
            [px, py, pz, E]

    This is different from the original implementation, where the code
    expected the last four dimensions to already represent a four-vector
    or something convertible to one.
    """

    eta = x[..., 0]
    phi = x[..., 1]
    pt = x[..., 2].clamp(min=eps)
    mass = x[..., 5].clamp(min=0.0)

    px = pt * torch.cos(phi)
    py = pt * torch.sin(phi)
    pz = pt * torch.sinh(eta)

    p2 = px.square() + py.square() + pz.square()
    energy = torch.sqrt(p2 + mass.square() + eps)

    return torch.stack([px, py, pz, energy], dim=-1)


def pairwise_physics_features(
    x: torch.Tensor,
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

    eta = x[..., 0]
    phi = x[..., 1]
    pt = x[..., 2].clamp(min=eps)

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

    p4 = build_four_vector_from_eta_phi_pt_mass(x, eps=eps)
    p4_i = p4.unsqueeze(2)
    p4_j = p4.unsqueeze(1)
    p4_ij = p4_i + p4_j

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
    Node feature embedding.

    Submodule stage:
        Stage 1 of the main model.

    Input:
        x: (B, N, input_dim)

    Output:
        h: (B, N, embed_dim)

    This replaces the original implementation's CPF/NPF/VTX/LT-specific
    InputProcess modules.

    Original code:
        The existing implementation separately embedded cpf, npf, vtx, lt
        using Conv1d or Linear layers, then concatenated them along the
        sequence dimension.

    This minimal version:
        There is only one type of node, so a single shared Linear embedding
        is applied to every node.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.norm = RMSNorm(input_dim)
        self.proj = nn.Linear(input_dim, embed_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.proj(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


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
        pair_features = pairwise_physics_features(
            x,
            padding_mask=padding_mask,
            eps=self.eps,
        )

        bias = self.mlp(pair_features)
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

class CLSPooling(nn.Module):
    """
    CLS attention pooling module.

    Submodule stage:
        Stage after the Transformer encoder.

    This module uses a learnable CLS token as a query. The encoded particle
    nodes serve as keys and values.

    This is close to the original implementation's logic:
        ordinary nodes are first updated by self-attention;
        then a CLS token reads the final node representations.

    Input:
        h: (B, N, embed_dim)
        padding_mask: (B, N), optional

    Output:
        cls: (B, embed_dim)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.query_norm = RMSNorm(embed_dim)
        self.context_norm = RMSNorm(embed_dim)

        self.attn = nn.MultiheadAttention(
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
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = h.size(0)

        cls = self.cls_token.expand(batch_size, -1, -1)

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

        residual = cls
        cls = residual + self.ffn(self.ffn_norm(cls))

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

class MinimalParticleTransformerRepresentation(nn.Module):
    """
    Minimal ParticleTransformer for representation learning.

    Main stages:
        1. NodeEmbedding
        2. PairwiseAttentionBias
        3. ParticleTransformerEncoder
        4. CLSPooling
        5. RepresentationHead

    Input:
        x: (B, N, F)

    Optional input:
        padding_mask: (B, N)
            Boolean tensor.
            True means this node is padding and should be ignored.

    Output:
        z: (B, representation_dim)

    This model is not a classifier. It does not output class logits.
    """

    def __init__(self, config: ParticleTransformerConfig):
        super().__init__()

        self.config = config

        self.node_embedding = NodeEmbedding(
            input_dim=config.input_dim,
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
            dropout=config.dropout,
        )

        self.representation_head = RepresentationHead(
            embed_dim=config.embed_dim,
            representation_dim=config.representation_dim,
            dropout=config.dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        normalize_output: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x:
                Tensor of shape (B, N, F).

            padding_mask:
                Optional bool tensor of shape (B, N).
                True means padded node.
                False means valid node.

            normalize_output:
                If True, L2-normalize the representation vector.
                This is often useful for contrastive learning.

        Returns:
            z:
                Tensor of shape (B, representation_dim).
        """

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
            h = self.node_embedding(x)

            if padding_mask is not None:
                h = h.masked_fill(padding_mask.unsqueeze(-1), 0.0)

            attn_bias = None
            if self.pairwise_bias is not None:
                attn_bias = self.pairwise_bias(
                    x,
                    padding_mask=padding_mask,
                )

            h = self.encoder(
                h,
                attn_bias=attn_bias,
                padding_mask=padding_mask,
            )

            cls = self.cls_pooling(
                h,
                padding_mask=padding_mask,
            )

            z = self.representation_head(cls)

            if normalize_output:
                z = F.normalize(z, p=2, dim=-1)

        return z


# ------------------------------------------------------------
# Augmentation config
# ------------------------------------------------------------

@dataclass
class PtDropAugmentationConfig:
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
        x[..., 2] is pt.
    """

    num_global_views: int = 2
    num_local_views: int = 6

    global_drop_pt_frac_range: Tuple[float, float] = (0.0, 0.70)
    local_drop_pt_frac_range: Tuple[float, float] = (0.70, 0.95)

    min_nodes: int = 4

    pt_index: int = 2
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
# Random pt-drop multi-view augmentation
# ------------------------------------------------------------

class PtDropMultiViewAugmentation(nn.Module):
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

            score_i = -log(u_i) / weight_i
            weight_i = 1 / pt_i ** pt_drop_power

        Nodes with larger weights, i.e. lower pt, tend to get smaller scores
        and therefore appear earlier in the drop order. We then drop nodes in
        that order until the cumulative dropped pt reaches the sampled target,
        while preserving at least min_nodes valid nodes.

    Important:
        This is not bitwise equivalent to repeated multinomial sampling, but it
        implements the same intended stochastic policy: weighted sampling
        without replacement, biased toward dropping low-pt nodes first.

    This preserves the original sequence length N. It does not physically
    shrink the tensor. Dropped nodes are masked out.
    """

    def __init__(self, config: PtDropAugmentationConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
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
            view_x, view_mask = self._make_view_vectorized(
                x=x,
                padding_mask=padding_mask,
                drop_frac_range=self.config.global_drop_pt_frac_range,
            )
            views.append(view_x)
            view_padding_masks.append(view_mask)
            view_types.append("global")

        for _ in range(self.config.num_local_views):
            view_x, view_mask = self._make_view_vectorized(
                x=x,
                padding_mask=padding_mask,
                drop_frac_range=self.config.local_drop_pt_frac_range,
            )
            views.append(view_x)
            view_padding_masks.append(view_mask)
            view_types.append("local")

        return views, view_padding_masks, view_types

    def _make_view_vectorized(
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

    normalize_representations_for_invariant: bool = False
    normalize_representations_for_sigreg: bool = False


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

        invariant_z = z_views
        if self.config.normalize_representations_for_invariant:
            invariant_z = F.normalize(invariant_z, p=2, dim=-1)

        global_anchor = invariant_z[:num_global_views].mean(dim=0, keepdim=True)
        invariant_loss = (invariant_z - global_anchor).pow(2).mean()

        sigreg_z = z_views
        if self.config.normalize_representations_for_sigreg:
            sigreg_z = F.normalize(sigreg_z, p=2, dim=-1)

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
# Main SSL model
# ------------------------------------------------------------

class LeJEPAParticleTransformerRepresentation(MinimalParticleTransformerRepresentation):
    """
    Minimal ParticleTransformer with LeJEPA-style SSL training wrapper.

    Parent class:
        MinimalParticleTransformerRepresentation

    Added members:
        self.augmentation:
            PtDropMultiViewAugmentation

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
        augmentation_config: Optional[PtDropAugmentationConfig] = None,
        loss_config: Optional[LeJEPALossConfig] = None,
    ):
        super().__init__(model_config)

        if augmentation_config is None:
            augmentation_config = PtDropAugmentationConfig()

        if loss_config is None:
            loss_config = LeJEPALossConfig()

        self.augmentation = PtDropMultiViewAugmentation(augmentation_config)
        self.loss = LeJEPASIGRegLoss(loss_config)

    def forward_pretrain(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        normalize_output: bool = False,
        return_views: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Pretraining forward pass.

        Input:
            x:
                Node tensor with shape (B, N, F).

            padding_mask:
                Optional bool tensor with shape (B, N).
                True means padded node.

            normalize_output:
                If True, L2-normalize each view representation before it
                is returned from the encoder.

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

        batched_views = torch.cat(views, dim=0)
        batched_view_masks = torch.cat(view_padding_masks, dim=0)

        batched_z = super().forward(
            batched_views,
            padding_mask=batched_view_masks,
            normalize_output=normalize_output,
        )

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