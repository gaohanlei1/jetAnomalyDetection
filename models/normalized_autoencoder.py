"""Normalized autoencoder (NAE) for density estimation in representation space.

The network is an ordinary symmetric MLP autoencoder.  The NAE behavior comes
from the training objective used by the training script:

    E_theta(z) = ||z - AE_theta(z)||_2^2

During ordinary AE pretraining, only E_data is minimized.  During NAE training,
contrastive divergence approximately follows the maximum-likelihood gradient:

    grad L ~= E_data[grad E_theta(z)] - E_model[grad E_theta(z)]

where model samples are obtained with short Langevin chains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import torch
from torch import nn


@dataclass(frozen=True)
class NormalizedAutoencoderConfig:
    """Configuration for the representation-space MLP autoencoder.

    The DarkCLR architecture for a 128-dimensional representation is:

        128 -> 64 -> 32 -> 16 -> 8 -> 3
        3 -> 8 -> 16 -> 32 -> 64 -> 128

    ``hidden_dims`` excludes the input dimension and bottleneck dimension.
    """

    input_dim: int = 128
    hidden_dims: Sequence[int] = (64, 32, 16, 8)
    bottleneck_dim: int = 3
    activation: str = "relu"
    output_activation: str = "identity"


def _make_activation(name: str) -> nn.Module:
    normalized = name.strip().lower()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.2)
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "gelu":
        return nn.GELU()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name!r}")


def _build_mlp(
    dims: Iterable[int],
    activation: str,
    final_activation: str,
) -> nn.Sequential:
    dims = list(dims)
    if len(dims) < 2:
        raise ValueError("An MLP requires at least an input and output dimension.")

    layers: list[nn.Module] = []
    for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim))
        is_last = index == len(dims) - 2
        layers.append(
            _make_activation(final_activation if is_last else activation)
        )
    return nn.Sequential(*layers)


class NormalizedAutoencoder(nn.Module):
    """Symmetric MLP autoencoder whose reconstruction error is an energy.

    The module itself is deliberately simple.  Ordinary AE pretraining and NAE
    contrastive-divergence training are implemented by the training driver.
    """

    def __init__(self, config: NormalizedAutoencoderConfig):
        super().__init__()
        self.config = config

        hidden_dims = [int(value) for value in config.hidden_dims]
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one value.")
        if config.input_dim <= 0 or config.bottleneck_dim <= 0:
            raise ValueError("input_dim and bottleneck_dim must be positive.")

        encoder_dims = [
            int(config.input_dim),
            *hidden_dims,
            int(config.bottleneck_dim),
        ]
        decoder_dims = [
            int(config.bottleneck_dim),
            *reversed(hidden_dims),
            int(config.input_dim),
        ]

        self.encoder = _build_mlp(
            encoder_dims,
            activation=config.activation,
            final_activation="identity",
        )
        self.decoder = _build_mlp(
            decoder_dims,
            activation=config.activation,
            final_activation=config.output_activation,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        self._validate_input(z)
        return self.encoder(z)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 2:
            raise ValueError(
                f"Expected bottleneck tensor with shape (B, D), got {tuple(h.shape)}."
            )
        return self.decoder(h)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(z))

    def energy_per_sample(self, z: torch.Tensor) -> torch.Tensor:
        """Squared L2 reconstruction energy for each event, shape ``(B,)``."""
        reconstruction = self(z)
        return (reconstruction - z).pow(2).sum(dim=-1)

    def mse_per_sample(self, z: torch.Tensor) -> torch.Tensor:
        """Dimension-averaged reconstruction error for reporting/ROC."""
        reconstruction = self(z)
        return (reconstruction - z).pow(2).mean(dim=-1)

    def forward_with_energy(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstruction = self(z)
        squared_error = (reconstruction - z).pow(2)
        return {
            "reconstruction": reconstruction,
            "energy": squared_error.sum(dim=-1),
            "mse": squared_error.mean(dim=-1),
            "bottleneck": self.encode(z),
        }

    def _validate_input(self, z: torch.Tensor) -> None:
        if z.ndim != 2:
            raise ValueError(
                f"Expected representation tensor with shape (B, D), got {tuple(z.shape)}."
            )
        if z.size(-1) != self.config.input_dim:
            raise ValueError(
                f"Expected input_dim={self.config.input_dim}, got {z.size(-1)}."
            )
