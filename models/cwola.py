"""Small binary CWoLa classifier for frozen event representations."""
import torch
from torch import nn


class CWoLaMLP(nn.Module):
    def __init__(self, input_dim=128, dropout=0.1):
        super().__init__()
        if input_dim < 1 or not 0 <= dropout < 1:
            raise ValueError("input_dim must be positive and dropout must be in [0, 1).")
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward_logits(self, representation):
        """Stable BCE training; larger logits mean more mixture-like."""
        return self.layers(representation).squeeze(-1)

    def forward(self, representation):
        return torch.sigmoid(self.forward_logits(representation))
