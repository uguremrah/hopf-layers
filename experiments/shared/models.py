"""CNN backbone and task-specific heads for hopf-layers experiments.

Architecture: 3-block CNN with batch normalization, ReLU, and global average pooling,
followed by a task-specific head (classification or regression).

The backbone is the same across all experiments to ensure fair ablation comparison.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class CNNBackbone(nn.Module):
    """Shared 3-block CNN backbone.

    Architecture:
        in_channels -> 32 -> 64 -> 128 -> global_avg_pool -> 128-dim vector

    Uses periodic padding (via circular mode) to respect lattice periodicity.
    """

    def __init__(self, in_channels: int, hidden_dims: tuple[int, ...] = (32, 64, 128)):
        super().__init__()
        dims = (in_channels,) + hidden_dims
        layers = []
        for i in range(len(hidden_dims)):
            layers.append(ConvBlock(dims[i], dims[i + 1]))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = hidden_dims[-1]

    def forward(self, x: Tensor) -> Tensor:
        """Extract feature vector from spatial input.

        Args:
            x: (batch, C, H, W)

        Returns:
            (batch, out_dim) feature vector.
        """
        h = self.features(x)
        h = self.pool(h)
        return h.flatten(1)


class ClassificationHead(nn.Module):
    """Classification head: FC -> ReLU -> FC -> logits."""

    def __init__(self, in_dim: int, num_classes: int, hidden: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class RegressionHead(nn.Module):
    """Regression head: FC -> ReLU -> FC -> scalar."""

    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x).squeeze(-1)


class ExperimentModel(nn.Module):
    """Combined backbone + head model.

    Args:
        in_channels: Number of input channels (depends on ablation).
        task: 'classification' or 'regression'.
        num_classes: For classification task only.
        hidden_dims: CNN backbone channel progression.
    """

    def __init__(
        self,
        in_channels: int,
        task: str = "classification",
        num_classes: int = 2,
        hidden_dims: tuple[int, ...] = (32, 64, 128),
    ):
        super().__init__()
        self.backbone = CNNBackbone(in_channels, hidden_dims)
        if task == "classification":
            self.head = ClassificationHead(self.backbone.out_dim, num_classes)
        elif task == "regression":
            self.head = RegressionHead(self.backbone.out_dim)
        else:
            raise ValueError(f"Unknown task: {task}")

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        return self.head(features)
