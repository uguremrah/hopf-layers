"""Dataset classes for loading MC-generated gauge configurations.

Supports both pure SU(2) gauge configs and SU(2)+Higgs configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .ablations import AblationMode, HopfFeatureExtractor


class GaugeDataset(Dataset):
    """Dataset of pure SU(2) gauge configurations with scalar labels.

    Each sample is a gauge link field (4, 2, Lx, Ly) with an associated label.
    """

    def __init__(
        self,
        configs: list[np.ndarray] | np.ndarray,
        labels: list[float] | np.ndarray,
    ):
        if isinstance(configs, list):
            configs = np.stack(configs)
        if isinstance(labels, list):
            labels = np.array(labels, dtype=np.float32)
        self.configs = torch.from_numpy(configs).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.configs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.configs[idx], self.labels[idx]


class HiggsDataset(Dataset):
    """Dataset of SU(2)+Higgs configurations with phase labels.

    Each sample contains gauge links (4, 2, Lx, Ly) and optionally Higgs field (3, Lx, Ly).
    Labels are integer phase indices.
    """

    def __init__(
        self,
        gauge_configs: list[np.ndarray] | np.ndarray,
        labels: list[int] | np.ndarray,
        higgs_configs: Optional[list[np.ndarray] | np.ndarray] = None,
    ):
        if isinstance(gauge_configs, list):
            gauge_configs = np.stack(gauge_configs)
        if isinstance(labels, list):
            labels = np.array(labels, dtype=np.int64)
        self.gauge = torch.from_numpy(gauge_configs).float()
        self.labels = torch.from_numpy(labels).long()
        self.higgs = None
        if higgs_configs is not None:
            if isinstance(higgs_configs, list):
                higgs_configs = np.stack(higgs_configs)
            self.higgs = torch.from_numpy(higgs_configs).float()

    def __len__(self) -> int:
        return len(self.gauge)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.gauge[idx], self.labels[idx]


class PrecomputedFeatureDataset(Dataset):
    """Dataset with pre-extracted HopfLayer features.

    For efficiency, run HopfFeatureExtractor once and cache the results.
    """

    def __init__(
        self,
        features: Tensor,
        labels: Tensor,
    ):
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.features[idx], self.labels[idx]


def precompute_features(
    dataset: GaugeDataset | HiggsDataset,
    mode: AblationMode,
    batch_size: int = 64,
    device: str = "auto",
) -> PrecomputedFeatureDataset:
    """Pre-extract HopfLayer features from a gauge dataset.

    Runs the HopfFeatureExtractor on the entire dataset in batches
    and returns a new dataset with cached features.

    Args:
        dataset: Source dataset with gauge configs.
        mode: Ablation mode determining which features to extract.
        batch_size: Batch size for feature extraction.
        device: Device for computation.

    Returns:
        PrecomputedFeatureDataset with extracted features.
    """
    resolved_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if device == "auto" else torch.device(device)

    extractor = HopfFeatureExtractor(mode)
    all_features = []
    all_labels = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(resolved_device)
            feats = extractor(batch_x)
            all_features.append(feats.cpu())
            all_labels.append(batch_y)

    return PrecomputedFeatureDataset(
        features=torch.cat(all_features, dim=0),
        labels=torch.cat(all_labels, dim=0),
    )
