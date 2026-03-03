"""Ablation configurations for HopfLayer experiments.

Defines the 4 standard ablation configs that are consistent across all experiments:
  1. raw       — Raw quaternion link field (baseline)
  2. base_only — HopfLayer base (S² coordinates) only
  3. base_fiber — HopfLayer base + fiber phase
  4. full_hopf  — HopfLayer base + fiber + transition signals
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor

from hopf_layers import ClassicalHopfLayer


class AblationMode(str, Enum):
    RAW = "raw"
    BASE_ONLY = "base_only"
    BASE_FIBER = "base_fiber"
    FULL_HOPF = "full_hopf"


# Input channel counts for each ablation (link field with 2 directions)
ABLATION_CHANNELS = {
    AblationMode.RAW: 8,          # 4 quaternion × 2 directions
    AblationMode.BASE_ONLY: 6,    # 3 S² × 2 directions
    AblationMode.BASE_FIBER: 8,   # (3 S² + 1 fiber) × 2 directions
    AblationMode.FULL_HOPF: 12,   # (3 S² + 1 fiber + 1 trans_x + 1 trans_y) × 2 directions
}


@dataclass
class AblationConfig:
    """Configuration for an ablation variant."""
    mode: AblationMode
    in_channels: int
    label: str

    @staticmethod
    def all_configs() -> list[AblationConfig]:
        return [
            AblationConfig(AblationMode.RAW, ABLATION_CHANNELS[AblationMode.RAW], "Raw quaternion"),
            AblationConfig(AblationMode.BASE_ONLY, ABLATION_CHANNELS[AblationMode.BASE_ONLY], "Base only (S²)"),
            AblationConfig(AblationMode.BASE_FIBER, ABLATION_CHANNELS[AblationMode.BASE_FIBER], "Base + fiber"),
            AblationConfig(AblationMode.FULL_HOPF, ABLATION_CHANNELS[AblationMode.FULL_HOPF], "Full HopfLayer"),
        ]


class HopfFeatureExtractor:
    """Extract features from gauge link fields according to ablation mode.

    Input: gauge link field of shape (batch, 4, 2, Lx, Ly) — quaternion link field.
    Output: feature tensor of shape (batch, C, Lx, Ly) where C depends on ablation.

    The 2 link directions are flattened into the channel dimension.
    """

    def __init__(self, mode: AblationMode):
        self.mode = mode
        if mode != AblationMode.RAW:
            self.hopf = ClassicalHopfLayer()

    def __call__(self, links: Tensor) -> Tensor:
        """Extract features from gauge links.

        Args:
            links: (batch, 4, 2, Lx, Ly) quaternion link field.

        Returns:
            (batch, C, Lx, Ly) feature channels.
        """
        B, _, _, Lx, Ly = links.shape

        if self.mode == AblationMode.RAW:
            # Flatten directions into channels: (B, 4, 2, Lx, Ly) -> (B, 8, Lx, Ly)
            return links.reshape(B, 8, Lx, Ly)

        out = self.hopf(links)
        # out.base: (B, 3, 2, Lx, Ly), out.fiber: (B, 2, Lx, Ly)
        # out.transitions_x: (B, 2, Lx, Ly), out.transitions_y: (B, 2, Lx, Ly)

        if self.mode == AblationMode.BASE_ONLY:
            # (B, 3, 2, Lx, Ly) -> (B, 6, Lx, Ly)
            return out.base.reshape(B, 6, Lx, Ly)

        elif self.mode == AblationMode.BASE_FIBER:
            base_flat = out.base.reshape(B, 6, Lx, Ly)
            # fiber: (B, 2, Lx, Ly) — sin/cos encode for continuity
            fiber = out.fiber  # (B, 2, Lx, Ly) already has 2 directions
            return torch.cat([base_flat, fiber], dim=1)  # (B, 8, Lx, Ly)

        else:  # FULL_HOPF
            base_flat = out.base.reshape(B, 6, Lx, Ly)
            fiber = out.fiber  # (B, 2, Lx, Ly)
            tx = out.transitions_x  # (B, 2, Lx, Ly)
            ty = out.transitions_y  # (B, 2, Lx, Ly)
            return torch.cat([base_flat, fiber, tx, ty], dim=1)  # (B, 12, Lx, Ly)
