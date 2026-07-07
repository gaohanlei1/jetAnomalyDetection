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
    """
    Configuration object for the minimal particle transformer.

    This model assumes the input tensor has shape:

        x: (B, N, F)

    where each node has features ordered as:

        "eta,phi,pt,d0/d0Err,dz/dzErr,charge,mass,log_pt,pdgId_-211,pdgId_-13,pdgId_-11,pdgId_11,pdgId_13,pdgId_22,pdgId_130,pdgId_211"

    The model produces a representation vector instead of class logits.
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
            6: mass

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
    mass = x[..., 6].clamp(min=0.0)

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
            dropout=config.class_dropout,
            num_class_layers=config.num_class_layers,
        )

        self.representation_head = RepresentationHead(
            embed_dim=config.embed_dim,
            representation_dim=config.representation_dim,
            dropout=config.dropout,
        )

    def encode_cls(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode node features into a CLS token before the output head.

        Returns:
            cls: Tensor of shape (B, embed_dim).
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

        return cls

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

        cls = self.encode_cls(x, padding_mask=padding_mask)

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
            z = self.representation_head(cls)

            if normalize_output:
                z = F.normalize(z, p=2, dim=-1)

        return z


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


class ParticleTransformerClassifier(MinimalParticleTransformerRepresentation):
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
        normalize_output: bool = False,
    ) -> torch.Tensor:
        if normalize_output:
            raise ValueError(
                "ParticleTransformerClassifier does not support normalize_output."
            )

        cls = self.encode_cls(x, padding_mask=padding_mask)

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
# Corrupted negative view augmentation config
# ------------------------------------------------------------

@dataclass
class CorruptedNegativeAugmentationConfig:
    """
    Configuration for generating corrupted negative views for triplet training.

    These views are intentionally not guaranteed to be physically valid jets.
    The goal is to keep individual feature values mostly realistic while
    breaking event-level or node-feature joint structure.

    Feature convention for the current node feature list:
        0: eta
        1: phi
        2: pt
        3: d0/d0Err
        4: dz/dzErr
        5: charge
        6: mass
        7: log_pt
        8:15: pdgId one-hot block

    Default corruption sampling probabilities:
        45% batch_mix
        25% pt_resample
        20% node_eta_phi_rotation
         5% eta_phi_shuffle
         5% identity_shuffle
    """

    num_negative_views: int = 4
    # Each generated negative view independently samples one corruption mode
    # according to these probabilities. The probabilities must sum to 1.
    batch_mix_prob: float = 0.45
    pt_resample_prob: float = 0.25
    node_eta_phi_rotation_prob: float = 0.20
    eta_phi_shuffle_prob: float = 0.05
    identity_shuffle_prob: float = 0.05

    min_nodes: int = 4
    eps: float = 1e-8

    eta_index: int = 0
    phi_index: int = 1
    pt_index: int = 2
    d0_index: int = 3
    dz_index: int = 4
    charge_index: int = 5
    mass_index: int = 6
    log_pt_index: int = 7
    pdg_start_index: int = 8
    pdg_end_index: int = 16

    # Fraction of valid nodes whose selected feature block is corrupted for
    # within-event shuffles. 1.0 means every valid node receives a shuffled
    # block from another node in the same event.
    corrupt_node_frac: float = 1.0

    # In the batch-mix corruption, keep this fraction of nodes from the anchor
    # event and fill the remaining available slots with nodes from a donor
    # event in the same batch.
    batch_mix_anchor_frac: float = 0.5

    # After pt-changing corruptions, renormalize the valid-node pt sum to match
    # the original event's valid-node pt sum. This does not assume the sum is 1.
    renormalize_pt_sum: bool = True

    # For corrupted pt/log_pt values, keep the event-level log_pt distribution
    # on the same scale as the original event by matching the original valid-node
    # mean and standard deviation.
    renormalize_log_pt_stats: bool = True


# ------------------------------------------------------------
# Random pt-drop + rotation multi-view augmentation
# ------------------------------------------------------------

class MultiViewAugmentation(nn.Module):
    """
    Vectorized multi-view random node dropping + rotation augmentation.

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
        
        Additionally, each event is randomly rotated by a angle in [0, 2pi), 
        treating the eta-phi plane as a 2D Cartesian plane. 

    This preserves the original sequence length N. It does not physically
    shrink the tensor. Dropped nodes are masked out.
    """

    def __init__(self, config: MultiViewAugmentationConfig):
        super().__init__()
        self.config = config

    def _random_rotation(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device
        valid_mask = ~padding_mask  # shape: (B, N)
        
        # Sample random rotation angles for each event in the batch
        theta = torch.rand(batch_size, 1, device=device, dtype=torch.float32) * 2 * math.pi # shape: (B, 1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Extract eta and phi coordinates
        eta = x[..., 0].float() # shape: (B, N)
        phi = x[..., 1].float() # shape: (B, N)

        # Apply rotation
        rotated_eta = cos_theta * eta - sin_theta * phi
        rotated_phi = sin_theta * eta + cos_theta * phi

        # Combine rotated coordinates with other features
        rotated_x = x.clone()
        
        rotated_x[..., 0] = torch.where(valid_mask, rotated_eta, x[..., 0])
        rotated_x[..., 1] = torch.where(valid_mask, rotated_phi, x[..., 1])

        return rotated_x, padding_mask

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
            view_x, _ = self._random_rotation(view_x, view_mask)
            views.append(view_x)
            view_padding_masks.append(view_mask)
            view_types.append("global")

        for _ in range(self.config.num_local_views):
            view_x, view_mask = self._drop_nodes(
                x=x,
                padding_mask=padding_mask,
                drop_frac_range=self.config.local_drop_pt_frac_range,
            )
            view_x, _ = self._random_rotation(view_x, view_mask)
            views.append(view_x)
            view_padding_masks.append(view_mask)
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
            Within each event, shuffle the identity block jointly:
                charge, mass, pdgId one-hot.

        pt_resample:
            Within each event, resample pt values with replacement from that
            event's valid pt/log_pt set, keeping pt and scaled log_pt bound
            together. Then renormalize the pt sum to the original event pt sum
            and match the original event-level log_pt mean/std.

        node_eta_phi_rotation:
            Treat each valid node's eta-phi pair as a 2D Cartesian vector,
            convert it to polar form, and independently rotate every valid node
            by its own random angle sampled uniformly from [0, 2pi). The radial
            distance sqrt(eta^2 + phi^2) is preserved for each node.

        eta_phi_shuffle:
            Within each event, shuffle the eta-phi pair jointly across nodes.

        batch_mix:
            Build a fake jet by concatenating valid nodes from the anchor event
            and another donor event in the same batch. If the fake jet exceeds
            the current padded sequence length N, extra nodes are dropped.
            The resulting pt sum is renormalized to the original anchor event's
            pt sum, and log_pt is matched to the original anchor event's mean/std.

    Important:
        Every negative view is also rotated (event-level) randomly by an angle 
        in [0, 2pi), treating the eta-phi plane as a 2D Cartesian plane. This is
        done after the corruption mode is applied. This follows the same rotation 
        convention as the multi-view augmentation.
    """

    def __init__(self, config: CorruptedNegativeAugmentationConfig):
        super().__init__()
        self.config = config

        self._mode_names = (
            "batch_mix",
            "pt_resample",
            "node_eta_phi_rotation",
            "eta_phi_shuffle",
            "identity_shuffle",
        )

        mode_probabilities = torch.tensor(
            [
                config.batch_mix_prob,
                config.pt_resample_prob,
                config.node_eta_phi_rotation_prob,
                config.eta_phi_shuffle_prob,
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
            raise ValueError(
                f"Corruption probabilities must sum to 1.0, got {probability_sum:.8f}."
            )

        self.register_buffer(
            "_mode_probabilities",
            mode_probabilities,
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
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

                if mode == "identity_shuffle":
                    candidate_x, candidate_mask = self._identity_shuffle(x, padding_mask)
                elif mode == "pt_resample":
                    candidate_x, candidate_mask = self._pt_resample(x, padding_mask)
                elif mode == "node_eta_phi_rotation":
                    candidate_x, candidate_mask = self._node_eta_phi_rotation(x, padding_mask)
                elif mode == "eta_phi_shuffle":
                    candidate_x, candidate_mask = self._eta_phi_shuffle(x, padding_mask)
                elif mode == "batch_mix":
                    candidate_x, candidate_mask = self._batch_mix(x, padding_mask)
                else:
                    raise RuntimeError(f"Unexpected negative corruption mode: {mode}")

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

            # Apply one additional event-level rigid rotation to every event in
            # the completed negative view, matching the positive-view convention.
            neg_x, _ = self._random_rotation(neg_x, neg_mask)

            negative_views.append(neg_x)
            negative_padding_masks.append(neg_mask)
            negative_types.append(
                [self._mode_names[index] for index in event_mode_indices.tolist()]
            )

        return negative_views, negative_padding_masks, negative_types

    def _random_rotation( # Copied from MultiViewAugmentation
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.size(0)
        device = x.device
        valid_mask = ~padding_mask  # shape: (B, N)
        
        # Sample random rotation angles for each event in the batch
        theta = torch.rand(batch_size, 1, device=device, dtype=torch.float32) * 2 * math.pi # shape: (B, 1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        # Extract eta and phi coordinates
        eta = x[..., 0].float() # shape: (B, N)
        phi = x[..., 1].float() # shape: (B, N)

        # Apply rotation
        rotated_eta = cos_theta * eta - sin_theta * phi
        rotated_phi = sin_theta * eta + cos_theta * phi

        # Combine rotated coordinates with other features
        rotated_x = x.clone()
        
        rotated_x[..., 0] = torch.where(valid_mask, rotated_eta, x[..., 0])
        rotated_x[..., 1] = torch.where(valid_mask, rotated_phi, x[..., 1])

        return rotated_x, padding_mask
    
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

    def _renormalize_pt_and_log_pt(
        self,
        view_x: torch.Tensor,
        valid_mask: torch.Tensor,
        target_pt_sum: torch.Tensor,
        target_log_pt_mean: torch.Tensor,
        target_log_pt_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        Vectorized pt/log_pt post-processing for corrupted views.

        pt:
            Match each event's valid-node pt sum to target_pt_sum.

        log_pt:
            Match each event's valid-node log_pt mean/std to the original
            event statistics, without hard-coding the upstream scaler constants.
        """

        view_x = view_x.clone()
        valid_mask = valid_mask.bool()

        pt = view_x[..., self.config.pt_index].clamp(min=self.config.eps)

        if self.config.renormalize_pt_sum:
            current_sum = pt.masked_fill(~valid_mask, 0.0).sum(dim=1, keepdim=True)
            scale = target_pt_sum / current_sum.clamp(min=self.config.eps)
            pt = pt * scale
            view_x[..., self.config.pt_index] = torch.where(
                valid_mask,
                pt,
                view_x[..., self.config.pt_index],
            )

        if self.config.renormalize_log_pt_stats:
            log_pt = view_x[..., self.config.log_pt_index]
            current_mean, current_std = self._valid_mean_std(log_pt, valid_mask)

            normalized = (log_pt - current_mean) / current_std.clamp(min=self.config.eps)
            matched = normalized * target_log_pt_std.clamp(min=self.config.eps) + target_log_pt_mean

            # If an event has only one valid node, std matching is undefined.
            # In that case, set valid log_pt values to the original mean.
            valid_count = valid_mask.sum(dim=1, keepdim=True)
            matched = torch.where(valid_count > 1, matched, target_log_pt_mean)

            view_x[..., self.config.log_pt_index] = torch.where(
                valid_mask,
                matched,
                view_x[..., self.config.log_pt_index],
            )

        return view_x

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
            [self.config.charge_index, self.config.mass_index]
            + list(range(self.config.pdg_start_index, self.config.pdg_end_index)),
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
        original_log_pt_mean, original_log_pt_std = self._valid_mean_std(
            x[..., self.config.log_pt_index],
            valid_mask,
        )

        source_x = torch.gather(
            x,
            dim=1,
            index=source_idx.unsqueeze(-1).expand(-1, -1, x.size(-1)),
        )

        # Keep pt and scaled log_pt bound together when resampling.
        pt_log_pt_idx = torch.tensor(
            [self.config.pt_index, self.config.log_pt_index],
            device=x.device,
            dtype=torch.long,
        )
        updated_block = torch.where(
            target_mask.unsqueeze(-1),
            source_x[..., pt_log_pt_idx],
            view_x[..., pt_log_pt_idx],
        )
        view_x[..., pt_log_pt_idx] = updated_block

        can_renormalize = torch.isfinite(original_pt_sum) & (original_pt_sum > 0)
        view_x = self._renormalize_pt_and_log_pt(
            view_x=view_x,
            valid_mask=valid_mask & can_renormalize,
            target_pt_sum=original_pt_sum.clamp(min=self.config.eps),
            target_log_pt_mean=original_log_pt_mean,
            target_log_pt_std=original_log_pt_std,
        )

        return view_x, view_mask

    def _eta_phi_shuffle(
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
            [self.config.eta_index, self.config.phi_index],
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
        num_valid = valid_mask.sum(dim=1)

        donor_b = torch.randint(
            low=0,
            high=batch_size - 1,
            size=(batch_size,),
            device=device,
        )
        batch_indices = torch.arange(batch_size, device=device)
        donor_b = donor_b + (donor_b >= batch_indices).long() # index-shift to make sure donor is not myself

        donor_valid_mask = valid_mask[donor_b]
        donor_num_valid = donor_valid_mask.sum(dim=1)

        original_pt_sum = x[..., self.config.pt_index].clamp(min=0.0).masked_fill(
            ~valid_mask,
            0.0,
        ).sum(dim=1, keepdim=True)
        original_log_pt_mean, original_log_pt_std = self._valid_mean_std(
            x[..., self.config.log_pt_index],
            valid_mask,
        )

        num_anchor_keep = torch.round(
            self.config.batch_mix_anchor_frac * num_valid.float()
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
        view_x = self._renormalize_pt_and_log_pt(
            view_x=view_x,
            valid_mask=view_valid_mask & (~fallback).unsqueeze(1),
            target_pt_sum=original_pt_sum.clamp(min=self.config.eps),
            target_log_pt_mean=original_log_pt_mean,
            target_log_pt_std=original_log_pt_std,
        )

        return view_x, view_mask
    
    def _node_eta_phi_rotation(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Independently rotate every valid node in the eta-phi plane.

        Each node is treated as a 2D Cartesian vector:

            eta = r cos(alpha)
            phi = r sin(alpha)

        and receives its own independent random angle theta ~ Uniform(0, 2pi):

            alpha' = alpha + theta

        This preserves each node's radius sqrt(eta^2 + phi^2), while changing
        its angular position independently of every other node in the event.
        Padded nodes are left unchanged.
        """

        view_x = x.clone()
        view_mask = padding_mask.clone()

        valid_mask = self._valid_mask(padding_mask)

        eta = x[..., self.config.eta_index].float()
        phi = x[..., self.config.phi_index].float()

        radius = torch.sqrt(eta.square() + phi.square())
        angle = torch.atan2(phi, eta)

        theta = torch.rand(
            eta.shape,
            device=x.device,
            dtype=torch.float32,
        ) * (2.0 * math.pi)

        rotated_angle = angle + theta
        rotated_eta = radius * torch.cos(rotated_angle)
        rotated_phi = radius * torch.sin(rotated_angle)

        view_x[..., self.config.eta_index] = torch.where(
            valid_mask,
            rotated_eta.to(dtype=x.dtype),
            x[..., self.config.eta_index],
        )
        view_x[..., self.config.phi_index] = torch.where(
            valid_mask,
            rotated_phi.to(dtype=x.dtype),
            x[..., self.config.phi_index],
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

    normalize_representations_for_invariant: bool = False
    normalize_representations_for_sigreg: bool = False


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

class LeJEPAParticleTransformerRepresentation(MinimalParticleTransformerRepresentation):
    """
    Minimal ParticleTransformer with LeJEPA-style SSL training wrapper.

    Parent class:
        MinimalParticleTransformerRepresentation

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

        batched_views = torch.cat(views, dim=0) # shape: (num_views * B, N, F)
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


# ------------------------------------------------------------
# LeJEPA + triplet SSL model
# ------------------------------------------------------------

class LeJEPATripletParticleTransformerRepresentation(MinimalParticleTransformerRepresentation):
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

    def forward_pretrain(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        normalize_output: bool = False,
        return_views: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Pretraining forward pass with positive LeJEPA views and corrupted negatives.

        Input:
            x: (B, N, F)
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

        views, view_padding_masks, view_types = self.augmentation(
            x=x,
            padding_mask=padding_mask,
        )

        negative_views, negative_padding_masks, negative_types = self.negative_augmentation(
            x=x,
            padding_mask=padding_mask,
        )

        num_views = len(views)
        num_negative_views = len(negative_views)
        batch_size = x.size(0)

        batched_views = torch.cat(views, dim=0)
        batched_view_masks = torch.cat(view_padding_masks, dim=0)

        batched_negatives = torch.cat(negative_views, dim=0)
        batched_negative_masks = torch.cat(negative_padding_masks, dim=0)

        all_inputs = torch.cat([batched_views, batched_negatives], dim=0) # shape: (num_views * B + num_negative_views * B, N, F)
        all_masks = torch.cat([batched_view_masks, batched_negative_masks], dim=0) # shape: (num_views * B + num_negative_views * B, N)
        
        all_z = super().forward(
            all_inputs,
            padding_mask=all_masks,
            normalize_output=normalize_output,
        )

        z_views_flat = all_z[: num_views * batch_size]
        z_negatives_flat = all_z[num_views * batch_size :]

        z_views = z_views_flat.view(num_views, batch_size, -1)
        z_negatives = z_negatives_flat.view(num_negative_views, batch_size, -1)

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