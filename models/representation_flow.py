"""Affine flow for frozen CLS states, following arXiv:2412.15129, §§2.1–2.4.

Densities are continuous densities of CLS states (no image dequantization).
Loss is negative log likelihood in nats per dimension, including constants.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class CouplingMLP(nn.Module):
    """d -> Linear -> SwiGLU -> Linear -> 2d (scale logits and bias)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.input1 = nn.Linear(dim, 2 * hidden_dim)
        self.output1 = nn.Linear(hidden_dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.input2 = nn.Linear(dim, 2 * hidden_dim)
        self.output2 = nn.Linear(hidden_dim, 2 * dim)
        # Paper §2.4: zero the FINAL projection.
        nn.init.zeros_(self.output2.weight)
        nn.init.zeros_(self.output2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.input1(self.ln1(x)).chunk(2, dim=-1)
        x = self.output1(F.silu(gate) * value)
        gate, value = self.input2(self.ln2(x)).chunk(2, dim=-1)
        return self.output2(F.silu(gate) * value)


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, permutation: torch.Tensor):
        super().__init__()
        self.register_buffer("permutation", permutation)
        self.register_buffer("inverse_permutation", permutation.argsort())
        self.mlp = CouplingMLP(dim // 2, hidden_dim)

    def _transform(self, x: torch.Tensor):
        x1, x2 = x[..., self.permutation].chunk(2, dim=-1)
        scale_logits, bias = self.mlp(x1).chunk(2, dim=-1)
        # Keep affine arithmetic and log-Jacobians at least float32.
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x1, x2 = x1.to(dtype), x2.to(dtype)
        scale_logits, bias = scale_logits.to(dtype), bias.to(dtype)
        y2 = (x2 + bias) * (2.0 * torch.sigmoid(scale_logits))
        y = torch.cat((x1, y2), dim=-1)[..., self.inverse_permutation]
        return y, scale_logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._transform(x)[0]

    def forward_with_log_det(self, x: torch.Tensor):
        y, scale_logits = self._transform(x)
        log_scales = F.logsigmoid(scale_logits) + math.log(2.0)
        return y, log_scales


class RepresentationFlow(nn.Module):
    """Map a batch [B, D] of CLS states to a standard Gaussian.

    Fixed random partitions alternate with their complements, guaranteeing
    that every dimension is transformed in each pair of coupling layers.
    Permutations are checkpoint buffers; concatenation restores original order.
    """

    def __init__(self, dim: int, num_layers: int = 8, hidden_dim: int = 128,
                 seed: int = 42):
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError("CLS dimension must be positive and even.")
        if num_layers < 2 or hidden_dim < 1:
            raise ValueError("Use at least two coupling layers and a positive hidden_dim.")
        self.dim = dim
        generator = torch.Generator().manual_seed(seed)
        layers = []
        for index in range(num_layers):
            if index % 2 == 0:
                permutation = torch.randperm(dim, generator=generator)
            else:
                permutation = permutation.roll(dim // 2)
            layers.append(AffineCoupling(dim, hidden_dim, permutation))
        self.layers = nn.ModuleList(layers)

    def _validate(self, x: torch.Tensor) -> None:
        if x.ndim != 2 or x.shape[-1] != self.dim:
            raise ValueError(f"Expected a CLS batch [B, {self.dim}], got {tuple(x.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Only transform CLS states; no density, loss, or mutable cache."""
        self._validate(x)
        for layer in self.layers:
            x = layer(x)
        return x

    def loss(self, z: torch.Tensor, log_det: torch.Tensor) -> torch.Tensor:
        """Per-event NLL in nats/dim; -D * loss is log p(x)."""
        gaussian_nll = 0.5 * (z.square() + math.log(2.0 * math.pi)).sum(-1)
        return (gaussian_nll - log_det) / self.dim

    def forward_pretrain(self, x: torch.Tensor):
        self._validate(x)
        log_scales = []
        for layer in self.layers:
            x, cached_log_scales = layer.forward_with_log_det(x)
            log_scales.append(cached_log_scales)
        # Cache within this call/output, retaining autograd without stale graphs.
        log_scales = torch.stack(log_scales, dim=1)
        log_det = log_scales.sum(dim=(1, 2))
        nll = self.loss(x, log_det)
        return {"z": x, "log_scales": log_scales, "log_det": log_det,
                "nll": nll, "log_prob": -self.dim * nll,
                "loss": nll.mean(), "total_loss": nll.mean()}

    pretrain_forward = forward_pretrain
