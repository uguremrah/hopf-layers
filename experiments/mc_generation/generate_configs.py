"""Batch config generation from YAML scan files.

Usage:
    python generate_configs.py configs/pure_gauge_scan.yaml
    python generate_configs.py configs/higgs_phase_scan.yaml --dry-run
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from analytical import su2_plaquette_exact
from su2_metropolis import generate_configs, save_configs
from su2_higgs import HiggsConfig, generate_higgs_configs, save_higgs_configs


def run_pure_gauge_scan(spec: dict, out_root: Path, dry_run: bool = False) -> None:
    out_dir = out_root / spec["output_dir"]
    pattern = spec["file_pattern"]

    for L in spec["lattice_sizes"]:
        for i, beta in enumerate(spec["betas"]):
            fname = pattern.format(beta=beta, L=L)
            fpath = out_dir / fname
            seed = spec["seed_base"] + L * 1000 + i

            exact = su2_plaquette_exact(beta)
            print(f"[L={L:2d}, beta={beta:5.1f}] exact <P>={exact:.4f} -> {fname}")

            if dry_run:
                continue

            t0 = time.time()
            cfgs = generate_configs(
                beta=beta, L=L,
                n_configs=spec["n_configs"],
                n_therm=spec["n_therm"],
                n_skip=spec["n_skip"],
                epsilon=spec["epsilon"],
                seed=seed,
            )
            save_configs(cfgs, fpath)
            dt = time.time() - t0
            print(f"  -> {len(cfgs)} configs in {dt:.1f}s")


def run_higgs_scan(spec: dict, out_root: Path, dry_run: bool = False) -> None:
    out_dir = out_root / spec["output_dir"]
    pattern = spec["file_pattern"]

    for L in spec["lattice_sizes"]:
        for beta in spec["betas"]:
            for j, kappa in enumerate(spec["kappas"]):
                fname = pattern.format(beta=beta, kappa=kappa, L=L)
                fpath = out_dir / fname
                seed = spec["seed_base"] + L * 1000 + int(beta * 100) + j

                print(f"[L={L:2d}, beta={beta:.1f}, kappa={kappa:.2f}] -> {fname}")

                if dry_run:
                    continue

                cfg = HiggsConfig(
                    Lx=L, Ly=L, beta=beta, kappa=kappa,
                    m2=spec["m2"], lam=spec["lambda"], seed=seed,
                )
                t0 = time.time()
                cfgs = generate_higgs_configs(
                    cfg,
                    n_configs=spec["n_configs"],
                    n_therm=spec["n_therm"],
                    n_skip=spec["n_skip"],
                    gauge_epsilon=spec["gauge_epsilon"],
                    higgs_epsilon=spec["higgs_epsilon"],
                )
                save_higgs_configs(cfgs, fpath)
                dt = time.time() - t0
                print(f"  -> {len(cfgs)} configs in {dt:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Batch MC config generation")
    parser.add_argument("config", type=str, help="YAML config file")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument("--out-root", type=str, default=".",
                        help="Root directory for output")
    args = parser.parse_args()

    spec = yaml.safe_load(Path(args.config).read_text())
    out_root = Path(args.out_root)

    if "kappas" in spec:
        run_higgs_scan(spec, out_root, args.dry_run)
    else:
        run_pure_gauge_scan(spec, out_root, args.dry_run)


if __name__ == "__main__":
    main()
