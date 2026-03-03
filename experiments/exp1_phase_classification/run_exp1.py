#!/usr/bin/env python
"""Experiment 1: SU(2) Phase Classification via HopfLayer Ablation.

Classifies lattice gauge theory configurations into confined vs Higgs phases
using a HopfLayer-augmented CNN. Compares 4 ablation modes across 2 seeds.

The SU(2)+adjoint Higgs model has a phase transition from confined (low kappa)
to Higgs (high kappa) phase. We train a simple CNN to distinguish these phases
using features extracted at different levels of the Hopf fibration decomposition.

Usage:
    python run_exp1.py [--verbose]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — allow imports from parent experiment dirs and src/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch

from mc_generation.su2_higgs import HiggsConfig, generate_higgs_configs
from shared.ablations import AblationConfig, AblationMode
from shared.data import HiggsDataset, precompute_features
from shared.models import ExperimentModel
from shared.training import TrainConfig, train_classification

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

L = 16
BETA = 2.0
KAPPA_CONFINED = [0.1, 0.2, 0.3]
KAPPA_HIGGS = [0.8, 1.0, 1.5]
N_CONFIGS_PER_KAPPA = 833
N_THERM = 500
N_SKIP = 5
TRAIN_FRACTION = 0.7
SEEDS = [42, 123]
TRAIN_CONFIG = TrainConfig(
    epochs=50, batch_size=64, lr=1e-3, device="auto", verbose=False
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def generate_data(verbose: bool = False) -> tuple[HiggsDataset, HiggsDataset]:
    """Generate MC configurations and split into train/test datasets."""
    all_gauge = []
    all_higgs = []
    all_labels = []

    all_kappas = KAPPA_CONFINED + KAPPA_HIGGS

    for kappa in all_kappas:
        label = 0 if kappa < 0.5 else 1
        phase_name = "confined" if label == 0 else "Higgs"
        if verbose:
            print(f"Generating kappa={kappa} ({phase_name})...")

        config = HiggsConfig(
            Lx=L, Ly=L, beta=BETA, kappa=kappa, m2=1.0, lam=0.5, seed=None
        )
        configs = generate_higgs_configs(
            config,
            n_configs=N_CONFIGS_PER_KAPPA,
            n_therm=N_THERM,
            n_skip=N_SKIP,
            start="hot",
            verbose=False,
        )
        for gauge, higgs in configs:
            all_gauge.append(gauge)
            all_higgs.append(higgs)
            all_labels.append(label)

    # Shuffle deterministically
    np.random.seed(42)
    n_total = len(all_labels)
    indices = np.random.permutation(n_total)

    all_gauge = [all_gauge[i] for i in indices]
    all_higgs = [all_higgs[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]

    # Split 70/30
    n_train = int(TRAIN_FRACTION * n_total)
    train_ds = HiggsDataset(
        all_gauge[:n_train], all_labels[:n_train], all_higgs[:n_train]
    )
    test_ds = HiggsDataset(
        all_gauge[n_train:], all_labels[n_train:], all_higgs[n_train:]
    )

    n_confined_train = sum(1 for l in all_labels[:n_train] if l == 0)
    n_higgs_train = n_train - n_confined_train
    n_confined_test = sum(1 for l in all_labels[n_train:] if l == 0)
    n_higgs_test = (n_total - n_train) - n_confined_test

    print(f"Total: {n_total} configs ({sum(1 for l in all_labels if l == 0)} confined, "
          f"{sum(1 for l in all_labels if l == 1)} Higgs)")
    print(f"Train: {n_train} (confined={n_confined_train}, Higgs={n_higgs_train})")
    print(f"Test:  {n_total - n_train} (confined={n_confined_test}, Higgs={n_higgs_test})")

    return train_ds, test_ds


def run_experiment(verbose: bool = False) -> dict:
    """Run the full experiment: generate data, train all ablations, return results."""
    print("=" * 60)
    print("Experiment 1: SU(2) Phase Classification")
    print("=" * 60)

    # --- Generate data ---
    print("\n--- Data Generation ---")
    t0 = time.time()
    np.random.seed(42)
    train_ds, test_ds = generate_data(verbose=verbose)
    gen_time = time.time() - t0
    print(f"Data generation took {gen_time:.1f}s")

    # --- Precompute features for each ablation ---
    print("\n--- Feature Extraction ---")
    ablation_configs = AblationConfig.all_configs()
    precomputed_train = {}
    precomputed_test = {}

    for ac in ablation_configs:
        t0 = time.time()
        precomputed_train[ac.mode] = precompute_features(train_ds, ac.mode)
        precomputed_test[ac.mode] = precompute_features(test_ds, ac.mode)
        dt = time.time() - t0
        feat_shape = precomputed_train[ac.mode].features.shape
        print(f"  {ac.label:20s}: {feat_shape} ({dt:.2f}s)")

    # --- Train all ablation x seed combinations ---
    print("\n--- Training ---")
    results = {}

    for ac in ablation_configs:
        results[ac.mode.value] = {
            "label": ac.label,
            "in_channels": ac.in_channels,
            "seeds": {},
        }

        for seed in SEEDS:
            torch.manual_seed(seed)
            model = ExperimentModel(
                in_channels=ac.in_channels, task="classification", num_classes=2
            )
            result = train_classification(
                model,
                precomputed_train[ac.mode],
                precomputed_test[ac.mode],
                TRAIN_CONFIG,
                seed=seed,
            )
            results[ac.mode.value]["seeds"][str(seed)] = {
                "test_accuracy": float(result.test_metric),
                "best_val_accuracy": float(result.best_val_metric),
                "best_epoch": int(result.best_epoch),
                "train_losses": [float(x) for x in result.train_losses],
                "val_metrics": [float(x) for x in result.val_metrics],
            }
            if verbose:
                print(f"  {ac.label:20s} seed={seed}: "
                      f"test_acc={result.test_metric:.4f}, "
                      f"val_acc={result.best_val_metric:.4f}")

    # --- Print summary table ---
    print("\n--- Results ---")
    print(f"{'Ablation':20s} | {'Seed':5s} | {'Test Acc':8s} | {'Val Acc':8s} | {'Best Ep':7s}")
    print("-" * 60)

    for ac in ablation_configs:
        mode_results = results[ac.mode.value]
        for seed in SEEDS:
            sr = mode_results["seeds"][str(seed)]
            print(f"{ac.label:20s} | {seed:5d} | {sr['test_accuracy']:8.4f} | "
                  f"{sr['best_val_accuracy']:8.4f} | {sr['best_epoch']:7d}")

    # Compute means
    print("-" * 60)
    print(f"\n{'Ablation':20s} | {'Mean Test Acc':12s} | {'Std':8s}")
    print("-" * 50)
    for ac in ablation_configs:
        mode_results = results[ac.mode.value]
        accs = [mode_results["seeds"][str(s)]["test_accuracy"] for s in SEEDS]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        results[ac.mode.value]["mean_test_accuracy"] = float(mean_acc)
        results[ac.mode.value]["std_test_accuracy"] = float(std_acc)
        print(f"{ac.label:20s} | {mean_acc:12.4f} | {std_acc:8.4f}")

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "exp1_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return results


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    run_experiment(verbose=verbose)
