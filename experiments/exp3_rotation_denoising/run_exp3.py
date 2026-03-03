#!/usr/bin/env python
"""Experiment 3: Rotation Field Denoising via HopfLayer Ablation.

Generates smooth quaternion rotation fields on 2D grids, adds noise, and
trains per-pixel CNNs to recover clean rotation matrices. Compares 4 ablation
modes of the HopfLayer decomposition using geodesic distance on SO(3).

Usage:
    python run_exp3.py [--verbose]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup -- allow imports from parent experiment dirs and src/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from shared.ablations import AblationConfig, AblationMode, HopfFeatureExtractor, ABLATION_CHANNELS
from exp3_rotation_denoising.rotation_utils import (
    generate_denoising_dataset,
    quaternion_to_rotation_matrix,
    geodesic_distance,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

L = 16
N_SAMPLES = 5000
N_MODES = 3
SIGMAS = [0.1, 0.3, 0.5]
TRAIN_FRACTION = 0.7
SEEDS = [42, 123]
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
DEVICE = "auto"

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------------------
# Per-pixel model
# ---------------------------------------------------------------------------

class PixelwiseCNN(nn.Module):
    """Per-pixel rotation prediction from features."""

    def __init__(self, in_channels: int, out_channels: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, out_channels, 1),  # 1x1 conv for per-pixel output
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)  # (B, 9, Lx, Ly)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DenoisingDataset(Dataset):
    """Dataset of noisy quaternion fields and clean rotation matrix targets."""

    def __init__(self, noisy_fields: np.ndarray, clean_rotmats: np.ndarray):
        self.fields = torch.from_numpy(noisy_fields).float()
        self.targets = torch.from_numpy(clean_rotmats).float()

    def __len__(self) -> int:
        return len(self.fields)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return self.fields[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Feature pre-extraction
# ---------------------------------------------------------------------------

def _resolve_device() -> torch.device:
    """Resolve the global DEVICE setting to an actual torch.device."""
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


def extract_features(dataset: DenoisingDataset, mode: AblationMode, batch_size: int = 64):
    """Pre-extract HopfLayer features from a denoising dataset.

    Args:
        dataset: DenoisingDataset with (4, 2, Lx, Ly) fields.
        mode: Ablation mode.
        batch_size: Batch size for extraction.

    Returns:
        (features, targets) tuple of tensors.
    """
    device = _resolve_device()
    extractor = HopfFeatureExtractor(mode)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_feats = []
    all_targets = []
    with torch.no_grad():
        for fields, targets in loader:
            fields = fields.to(device)
            feats = extractor(fields)
            all_feats.append(feats.cpu())
            all_targets.append(targets)
    return torch.cat(all_feats, dim=0), torch.cat(all_targets, dim=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_pixelwise(
    model: PixelwiseCNN,
    features: Tensor,
    targets: Tensor,
    seed: int = 42,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    verbose: bool = False,
) -> dict:
    """Train per-pixel rotation predictor and evaluate.

    Args:
        model: PixelwiseCNN model.
        features: (N, C, Lx, Ly) input features.
        targets: (N, Lx, Ly, 9) clean rotation matrices.
        seed: Random seed.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        lr: Learning rate.
        verbose: Print training progress.

    Returns:
        Dict with train_losses, test_mse, test_geodesic_mean.
    """
    torch.manual_seed(seed)
    device = _resolve_device()
    model = model.to(device)

    # Split train/test (70/30)
    n_total = len(features)
    n_train = int(TRAIN_FRACTION * n_total)
    n_test = n_total - n_train
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(seed))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_x = features[train_idx].to(device)
    train_y = targets[train_idx].to(device)
    test_x = features[test_idx].to(device)
    test_y = targets[test_idx].to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    train_losses = []

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed + epoch))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            batch_idx = perm[i : i + batch_size]
            bx = train_x[batch_idx]
            by = train_y[batch_idx]

            pred = model(bx)  # (B, 9, Lx, Ly)
            pred = pred.permute(0, 2, 3, 1)  # (B, Lx, Ly, 9)
            loss = criterion(pred, by)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1:3d}: loss={avg_loss:.6f}")

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        pred_test = model(test_x).permute(0, 2, 3, 1)  # (B, Lx, Ly, 9)
        test_mse = criterion(pred_test, test_y).item()

        # Geodesic distance — move to CPU for numpy compatibility
        pred_R = pred_test.cpu().reshape(-1, 3, 3)
        true_R = test_y.cpu().reshape(-1, 3, 3)
        geo_dist = geodesic_distance(pred_R, true_R)
        geo_mean = geo_dist.mean().item()
        geo_std = geo_dist.std().item()

    return {
        "train_losses": [float(x) for x in train_losses],
        "test_mse": float(test_mse),
        "test_geodesic_mean": float(geo_mean),
        "test_geodesic_std": float(geo_std),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(verbose: bool = False) -> dict:
    """Run the full experiment and return results."""
    print("=" * 60)
    print("Experiment 3: Rotation Field Denoising")
    print("=" * 60)

    all_results = {}

    # -----------------------------------------------------------------------
    # Part 1: Ablation study at sigma=0.3
    # -----------------------------------------------------------------------
    sigma_ablation = 0.3
    print(f"\n--- Part 1: Ablation Study (sigma={sigma_ablation}) ---")
    print(f"Generating {N_SAMPLES} samples, L={L}...")

    t0 = time.time()
    noisy_fields, clean_rotmats = generate_denoising_dataset(
        N_SAMPLES, L, L, sigma_ablation, n_modes=N_MODES, seed_base=42
    )
    gen_time = time.time() - t0
    print(f"Data generation took {gen_time:.1f}s")

    dataset = DenoisingDataset(noisy_fields, clean_rotmats)

    ablation_configs = AblationConfig.all_configs()
    ablation_results = {}

    for ac in ablation_configs:
        print(f"\n  Ablation: {ac.label} (in_channels={ac.in_channels})")

        # Pre-extract features
        feats, tgts = extract_features(dataset, ac.mode)
        print(f"    Feature shape: {feats.shape}")

        seed_results = {}
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = PixelwiseCNN(in_channels=ac.in_channels)
            result = train_pixelwise(
                model, feats, tgts, seed=seed, verbose=verbose
            )
            seed_results[str(seed)] = result
            print(f"    seed={seed}: MSE={result['test_mse']:.6f}, "
                  f"geodesic={result['test_geodesic_mean']:.4f} +/- "
                  f"{result['test_geodesic_std']:.4f}")

        # Compute mean across seeds
        geo_means = [seed_results[str(s)]["test_geodesic_mean"] for s in SEEDS]
        mse_means = [seed_results[str(s)]["test_mse"] for s in SEEDS]

        ablation_results[ac.mode.value] = {
            "label": ac.label,
            "in_channels": ac.in_channels,
            "seeds": seed_results,
            "geodesic_mean": float(np.mean(geo_means)),
            "geodesic_std_across_seeds": float(np.std(geo_means)),
            "mse_mean": float(np.mean(mse_means)),
        }

    all_results["ablation_study"] = ablation_results

    # Print summary
    print(f"\n{'='*60}")
    print(f"{'Ablation':20s} | {'Geo Mean':>10s} | {'Geo Std':>10s} | {'MSE':>10s}")
    print("-" * 60)
    for ac in ablation_configs:
        r = ablation_results[ac.mode.value]
        print(f"{ac.label:20s} | {r['geodesic_mean']:>10.4f} | "
              f"{r['geodesic_std_across_seeds']:>10.4f} | {r['mse_mean']:>10.6f}")

    # -----------------------------------------------------------------------
    # Part 2: Noise level sweep with full_hopf
    # -----------------------------------------------------------------------
    print(f"\n--- Part 2: Noise Level Sweep (full_hopf) ---")

    noise_sweep_results = {}
    for sigma in SIGMAS:
        print(f"\n  sigma={sigma}")
        noisy_s, clean_s = generate_denoising_dataset(
            N_SAMPLES, L, L, sigma, n_modes=N_MODES, seed_base=42
        )
        ds_s = DenoisingDataset(noisy_s, clean_s)
        feats_s, tgts_s = extract_features(ds_s, AblationMode.FULL_HOPF)

        torch.manual_seed(42)
        model = PixelwiseCNN(in_channels=ABLATION_CHANNELS[AblationMode.FULL_HOPF])
        result = train_pixelwise(
            model, feats_s, tgts_s, seed=42, verbose=verbose
        )
        noise_sweep_results[str(sigma)] = result
        print(f"    geodesic={result['test_geodesic_mean']:.4f}, "
              f"MSE={result['test_mse']:.6f}")

    all_results["noise_sweep"] = noise_sweep_results

    # Print noise sweep summary
    print(f"\n{'='*60}")
    print(f"{'sigma':>8s} | {'Geodesic Mean':>14s} | {'MSE':>10s}")
    print("-" * 40)
    for sigma in SIGMAS:
        r = noise_sweep_results[str(sigma)]
        print(f"{sigma:>8.1f} | {r['test_geodesic_mean']:>14.4f} | "
              f"{r['test_mse']:>10.6f}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "exp3_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    run_experiment(verbose=verbose)
