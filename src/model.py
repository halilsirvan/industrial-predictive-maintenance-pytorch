"""PyTorch model definition."""

from __future__ import annotations

import torch
from torch import nn


class FailureMLP(nn.Module):
    """Compact MLP for imbalanced binary equipment-failure prediction."""

    def __init__(self, input_dim: int, dropout: float = 0.20) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(16, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(1)

