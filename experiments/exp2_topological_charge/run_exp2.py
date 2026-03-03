"""Experiment 2: Topological Charge Detection via HopfLayer Ablation.

Generates 2D SU(2) gauge configurations at multiple beta values,
computes topological charge (vortex winding number), and trains
CNN regressors with 4 ablation modes to predict the charge.

Usage:
    python experiments/exp2_topological_charge/run_exp2.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Path setup: add experiments/ and src/ to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

from mc_generation.su2_metropolis import generate_configs
from exp2_topological_charge.charge_utils import compute_charge_batch
from shared.ablations import AblationMode, AblationConfig, ABLATION_CHANNELS
from shared.data import GaugeDataset, precompute_features
from shared.models import ExperimentModel
from shared.training import TrainConfig, train_regression, compute_r2, compute_mae


def main():
    t0 = time.time()

    # -------------------------------------------------------------------------
    # 1. Generate SU(2) gauge configs at multiple beta values
    # -------------------------------------------------------------------------
    betas = [1.0, 2.0, 4.0, 6.0]
    L = 16
    n_per_beta = 1250
    n_therm = 500
    n_skip = 5

    print("=" * 60)
    print("Experiment 2: Topological Charge Detection")
    print("=" * 60)
    print(f"\nGenerating {n_per_beta} configs per beta, L={L}")

    all_configs = []
    all_betas_label = []

    for beta in betas:
        print(f"\n  beta={beta}: generating {n_per_beta} configs...", end=" ", flush=True)
        cfgs = generate_configs(
            beta=beta, L=L, n_configs=n_per_beta,
            n_therm=n_therm, n_skip=n_skip, epsilon=0.5,
            start="hot", seed=int(beta * 1000), verbose=False,
        )
        all_configs.extend(cfgs)
        all_betas_label.extend([beta] * n_per_beta)
        print("done")

    print(f"\nTotal configs: {len(all_configs)}")

    # -------------------------------------------------------------------------
    # 2. Compute topological charge for each config
    # -------------------------------------------------------------------------
    print("\nComputing topological charges...")
    charges = compute_charge_batch(all_configs)

    for beta in betas:
        mask = np.array(all_betas_label) == beta
        q_beta = charges[mask]
        print(f"  beta={beta}: Q mean={q_beta.mean():.3f}, "
              f"std={q_beta.std():.3f}, "
              f"range=[{q_beta.min():.3f}, {q_beta.max():.3f}]")

    # -------------------------------------------------------------------------
    # 3. Train/test split (70/30)
    # -------------------------------------------------------------------------
    n_total = len(all_configs)
    n_train = int(0.7 * n_total)

    # Shuffle with fixed seed
    rng = np.random.RandomState(42)
    indices = rng.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_configs = [all_configs[i] for i in train_idx]
    test_configs = [all_configs[i] for i in test_idx]
    train_charges = charges[train_idx]
    test_charges = charges[test_idx]

    train_dataset = GaugeDataset(train_configs, train_charges)
    test_dataset = GaugeDataset(test_configs, test_charges)

    print(f"\nTrain: {len(train_dataset)}, Test: {len(test_dataset)}")

    # -------------------------------------------------------------------------
    # 4. Run ablation study: 4 modes x 2 seeds
    # -------------------------------------------------------------------------
    ablation_configs = AblationConfig.all_configs()
    seeds = [42, 123]
    train_cfg = TrainConfig(
        epochs=50, batch_size=64, lr=1e-3,
        device="auto", verbose=False,
    )

    results = {}

    for ab in ablation_configs:
        print(f"\n--- {ab.label} (in_channels={ab.in_channels}) ---")

        # Pre-compute features
        train_feat = precompute_features(train_dataset, ab.mode)
        test_feat = precompute_features(test_dataset, ab.mode)

        r2_list = []
        mae_list = []

        for seed in seeds:
            torch.manual_seed(seed)
            model = ExperimentModel(
                in_channels=ab.in_channels,
                task="regression",
            )
            result = train_regression(model, train_feat, test_feat, train_cfg, seed=seed)
            r2 = result.test_metric
            mae = compute_mae(result.test_predictions, result.test_labels)
            r2_list.append(r2)
            mae_list.append(mae)
            print(f"  seed={seed}: R2={r2:.4f}, MAE={mae:.4f}")

        results[ab.mode.value] = {
            "label": ab.label,
            "in_channels": ab.in_channels,
            "r2_mean": float(np.mean(r2_list)),
            "r2_std": float(np.std(r2_list)),
            "mae_mean": float(np.mean(mae_list)),
            "mae_std": float(np.std(mae_list)),
            "r2_per_seed": r2_list,
            "mae_per_seed": mae_list,
        }

    # -------------------------------------------------------------------------
    # 5. Print summary table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Mode':<20s} {'R2 mean':>10s} {'R2 std':>10s} {'MAE mean':>10s} {'MAE std':>10s}")
    print("-" * 60)
    for mode_val, r in results.items():
        print(f"{r['label']:<20s} {r['r2_mean']:>10.4f} {r['r2_std']:>10.4f} "
              f"{r['mae_mean']:>10.4f} {r['mae_std']:>10.4f}")

    # -------------------------------------------------------------------------
    # 6. Save results as JSON
    # -------------------------------------------------------------------------
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp2_results.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
