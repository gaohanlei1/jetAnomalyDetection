"""Frozen ZCA whitening fitted to background representations only."""
import math

import torch
from torch import nn


class BackgroundWhitening(nn.Module):
    def __init__(self, mean, matrix):
        super().__init__()
        mean = torch.as_tensor(mean).detach().float()
        matrix = torch.as_tensor(matrix).detach().float()
        if mean.ndim != 1 or matrix.shape != (len(mean), len(mean)):
            raise ValueError("Invalid whitening mean/matrix dimensions.")
        if not torch.isfinite(mean).all() or not torch.isfinite(matrix).all():
            raise ValueError("Whitening parameters contain NaN/Inf.")
        self.register_buffer("mean", mean)
        self.register_buffer("matrix", matrix)

    def forward(self, representation):
        if representation.ndim != 2 or representation.shape[1] != len(self.mean):
            raise ValueError("Representation dimension differs from fitted whitening.")
        result = (representation.float() - self.mean) @ self.matrix
        if not torch.isfinite(result).all():
            raise FloatingPointError("Whitened representations contain NaN/Inf.")
        return result

    @classmethod
    @torch.no_grad()
    def fit(cls, batches, *, eps=1e-5):
        """Streaming parallel-Welford covariance on CPU in float64.

        Sample covariance uses N-1. Eigenvalues are clipped to
        eps * max(largest_eigenvalue, 1), so singular directions stay finite.
        No dimensionality reduction or trainable parameters are introduced.
        """
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("whitening-eps must be finite and positive.")
        count, mean, m2 = 0, None, None
        for batch in batches:
            x = batch.detach().to(device="cpu", dtype=torch.float64)
            if x.ndim != 2 or not torch.isfinite(x).all():
                raise ValueError("Whitening fit needs finite 2D representations.")
            n = len(x)
            if n == 0:
                continue
            batch_mean = x.mean(dim=0)
            centered = x - batch_mean
            batch_m2 = centered.T @ centered
            if mean is None:
                mean, m2 = batch_mean, batch_m2
            else:
                if x.shape[1] != len(mean):
                    raise ValueError("Whitening fit representation dimensions changed.")
                delta = batch_mean - mean
                m2 += batch_m2 + torch.outer(delta, delta) * (count * n / (count + n))
                mean += delta * (n / (count + n))
            count += n
        if count < 2:
            raise ValueError("Whitening requires at least two training background events.")
        covariance = m2 / (count - 1)
        eigenvalues, vectors = torch.linalg.eigh((covariance + covariance.T) / 2)
        floor = eps * max(float(eigenvalues[-1]), 1.0)
        matrix = (vectors * eigenvalues.clamp_min(floor).rsqrt()) @ vectors.T
        transform = cls(mean, matrix).eval()
        metadata = dict(method="zca", num_events=count, input_dim=len(mean), eps=eps,
                        eigenvalue_floor=floor, min_eigenvalue=float(eigenvalues[0]),
                        max_eigenvalue=float(eigenvalues[-1]),
                        clipped_eigenvalues=int((eigenvalues < floor).sum()),
                        covariance_normalization="N-1", statistics_dtype="float64",
                        transform_dtype="float32")
        return transform, metadata

    def save(self, path, metadata):
        state = {key: value.detach().cpu() for key, value in self.state_dict().items()}
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(dict(version=1, state_dict=state, metadata=metadata), temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, *, input_dim, device):
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("version") != 1:
            raise ValueError("Unsupported whitening artifact version.")
        transform = cls(**saved["state_dict"])
        if len(transform.mean) != input_dim:
            raise ValueError("Saved whitening dimension does not match the backbone.")
        return transform.to(device).eval()
