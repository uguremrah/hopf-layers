"""SU(2) Metropolis Monte Carlo for 2D lattice gauge theory.

Generates gauge field configurations distributed according to the
Wilson action:  P[U] ~ exp(-S_W[U])  with  S_W = beta * sum(1 - (1/2) Tr U_P).

Conventions
-----------
- SU(2) elements are represented as unit quaternions q = (a0, a1, a2, a3).
- q and -q are DISTINCT SU(2) elements (SU(2) is NOT SO(3)).
  Do not force q[0] > 0 — that would break the group structure.
- Plaquette trace:  Tr(U) = 2 * a0  for quaternion  (a0, a1, a2, a3).
- Link storage:  links[4, 2, Lx, Ly]  (quaternion components, directions, spatial).
- The average plaquette  <(1/2) Tr U_P>  should match the exact Migdal-Rusakov
  result for 2D:  I_0(beta)/I_1(beta) - 2/beta.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------------
# Quaternion algebra (numpy, component-first layout: q[4, ...])
# ---------------------------------------------------------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion(s) to unit norm.  No sign flipping."""
    norm = np.sqrt(np.sum(q**2, axis=0, keepdims=True).clip(min=EPS))
    return q / norm


def quat_multiply(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (component-first layout)."""
    p0, p1, p2, p3 = p[0], p[1], p[2], p[3]
    q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
    return np.stack([
        p0*q0 - p1*q1 - p2*q2 - p3*q3,
        p0*q1 + p1*q0 + p2*q3 - p3*q2,
        p0*q2 - p1*q3 + p2*q0 + p3*q1,
        p0*q3 + p1*q2 - p2*q1 + p3*q0,
    ], axis=0)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate: (a0, -a1, -a2, -a3)."""
    result = q.copy()
    result[1:4] = -q[1:4]
    return result


def random_su2_near_identity(epsilon: float = 0.5) -> np.ndarray:
    """Generate a random SU(2) element close to the identity.

    Method: V = (cos(theta/2), sin(theta/2)*n) with theta ~ U[0, epsilon]
    and n uniform on S^2.

    For small epsilon, this is approximately identity + O(epsilon).
    """
    n = np.random.randn(3)
    n /= np.linalg.norm(n) + EPS
    theta = epsilon * np.random.uniform()
    half = theta / 2.0
    return np.array([np.cos(half), np.sin(half)*n[0],
                     np.sin(half)*n[1], np.sin(half)*n[2]])


# ---------------------------------------------------------------------------
# Lattice field
# ---------------------------------------------------------------------------

@dataclass
class LatticeConfig:
    Lx: int = 16
    Ly: int = 16
    beta: float = 2.0
    seed: Optional[int] = 42


class SU2Lattice:
    """SU(2) gauge field on a 2D square lattice with periodic BC.

    Storage: links[4, 2, Lx, Ly]
        - axis 0: quaternion components (a0, a1, a2, a3)
        - axis 1: direction (0 = x, 1 = y)
        - axis 2, 3: spatial site (x, y)
    """

    def __init__(self, config: LatticeConfig, start: str = "cold"):
        self.Lx = config.Lx
        self.Ly = config.Ly
        self.beta = config.beta

        if start == "cold":
            self.links = np.zeros((4, 2, self.Lx, self.Ly))
            self.links[0] = 1.0  # identity everywhere
        elif start == "hot":
            self.links = np.random.randn(4, 2, self.Lx, self.Ly)
            self.links = quat_normalize(self.links)
        else:
            raise ValueError(f"start must be 'cold' or 'hot', got {start!r}")

    # --- Link access (with periodic BC) ---

    def get_link(self, mu: int, x: int, y: int) -> np.ndarray:
        return self.links[:, mu, x % self.Lx, y % self.Ly]

    def set_link(self, mu: int, x: int, y: int, q: np.ndarray) -> None:
        self.links[:, mu, x % self.Lx, y % self.Ly] = q

    def get_link_dag(self, mu: int, x: int, y: int) -> np.ndarray:
        return quat_conjugate(self.get_link(mu, x, y))

    # --- Observables ---

    def compute_plaquette(self, x: int, y: int) -> np.ndarray:
        """Plaquette P(x,y) = U_x(x,y) U_y(x+1,y) U_x^dag(x,y+1) U_y^dag(x,y)."""
        u1 = self.get_link(0, x, y)
        u2 = self.get_link(1, x + 1, y)
        u3 = self.get_link_dag(0, x, y + 1)
        u4 = self.get_link_dag(1, x, y)
        return quat_multiply(quat_multiply(quat_multiply(u1, u2), u3), u4)

    def plaquette_trace(self, x: int, y: int) -> float:
        """Re Tr P(x,y) = 2 * a0 of the plaquette quaternion."""
        return float(2.0 * self.compute_plaquette(x, y)[0])

    def all_plaquette_traces(self) -> np.ndarray:
        """All plaquette traces as (Lx, Ly) array."""
        traces = np.empty((self.Lx, self.Ly))
        for x in range(self.Lx):
            for y in range(self.Ly):
                traces[x, y] = self.plaquette_trace(x, y)
        return traces

    def average_plaquette(self) -> float:
        """<(1/2) Re Tr U_P> averaged over all plaquettes."""
        return float(np.mean(self.all_plaquette_traces()) / 2.0)

    def polyakov_loop(self, mu: int = 1) -> np.ndarray:
        """Polyakov loop: product of links wrapping around direction mu.

        L_mu(x) = Tr(prod_{n=0}^{N-1} U_mu(x, n)) / 2

        Returns array of shape (L_perp,) where L_perp is the lattice
        extent in the perpendicular direction.
        """
        if mu == 1:
            # Wrap in y-direction, measure at each x
            loops = np.zeros(self.Lx)
            for x in range(self.Lx):
                q = np.array([1.0, 0.0, 0.0, 0.0])
                for y in range(self.Ly):
                    q = quat_multiply(q, self.get_link(1, x, y))
                loops[x] = q[0]  # (1/2) Tr = a0
        else:
            # Wrap in x-direction, measure at each y
            loops = np.zeros(self.Ly)
            for y in range(self.Ly):
                q = np.array([1.0, 0.0, 0.0, 0.0])
                for x in range(self.Lx):
                    q = quat_multiply(q, self.get_link(0, x, y))
                loops[y] = q[0]
        return loops

    def average_polyakov_loop(self, mu: int = 1) -> float:
        """Spatial average of |Polyakov loop|."""
        return float(np.mean(np.abs(self.polyakov_loop(mu))))

    def wilson_action(self) -> float:
        """S_W = beta * sum(1 - (1/2) Re Tr P)."""
        traces = self.all_plaquette_traces()
        return float(self.beta * np.sum(1.0 - traces / 2.0))

    def validate_unitarity(self, tol: float = 1e-6) -> tuple[bool, float]:
        """Check all links have unit norm."""
        norms = np.sqrt(np.sum(self.links**2, axis=0))
        max_dev = float(np.max(np.abs(norms - 1.0)))
        return max_dev < tol, max_dev

    # --- Snapshot ---

    def snapshot(self) -> np.ndarray:
        """Return a copy of the link field."""
        return self.links.copy()


# ---------------------------------------------------------------------------
# Metropolis algorithm
# ---------------------------------------------------------------------------

def _local_action(lattice: SU2Lattice, mu: int, x: int, y: int) -> float:
    """Action contribution from the 2 plaquettes sharing link U_mu(x,y).

    For a 2D lattice, each link touches exactly 2 plaquettes.
    """
    if mu == 0:  # x-link: plaquettes at (x,y) and (x,y-1)
        t1 = lattice.plaquette_trace(x, y)
        t2 = lattice.plaquette_trace(x, y - 1)
    else:  # y-link: plaquettes at (x,y) and (x-1,y)
        t1 = lattice.plaquette_trace(x, y)
        t2 = lattice.plaquette_trace(x - 1, y)
    return lattice.beta * (2.0 - (t1 + t2) / 2.0)


def metropolis_sweep(lattice: SU2Lattice, epsilon: float = 0.5) -> float:
    """One Metropolis sweep over all links.  Returns acceptance rate."""
    n_accepted = 0
    n_total = 2 * lattice.Lx * lattice.Ly

    for x in range(lattice.Lx):
        for y in range(lattice.Ly):
            for mu in range(2):
                u_old = lattice.get_link(mu, x, y).copy()
                s_old = _local_action(lattice, mu, x, y)

                # Propose: U_new = V * U_old
                v = random_su2_near_identity(epsilon)
                u_new = quat_multiply(v, u_old)
                u_new = quat_normalize(u_new)
                lattice.set_link(mu, x, y, u_new)

                s_new = _local_action(lattice, mu, x, y)
                delta_s = s_new - s_old

                if delta_s < 0 or np.random.random() < np.exp(-delta_s):
                    n_accepted += 1
                else:
                    lattice.set_link(mu, x, y, u_old)

    return n_accepted / n_total


# ---------------------------------------------------------------------------
# Thermalization & config generation
# ---------------------------------------------------------------------------

@dataclass
class ThermHistory:
    """Thermalization history."""
    plaquettes: list[float] = field(default_factory=list)
    acceptance: list[float] = field(default_factory=list)


def thermalize(
    lattice: SU2Lattice,
    n_sweeps: int = 200,
    epsilon: float = 0.5,
    verbose: bool = False,
) -> ThermHistory:
    """Run Metropolis sweeps to reach thermal equilibrium."""
    history = ThermHistory()
    for i in range(n_sweeps):
        acc = metropolis_sweep(lattice, epsilon)
        avg_p = lattice.average_plaquette()
        history.plaquettes.append(avg_p)
        history.acceptance.append(acc)
        if verbose and (i + 1) % 20 == 0:
            print(f"  Sweep {i+1:4d}: <P> = {avg_p:.4f}, acc = {acc:.2%}")
    return history


def generate_configs(
    beta: float,
    L: int,
    n_configs: int = 100,
    n_therm: int = 500,
    n_skip: int = 10,
    epsilon: float = 0.5,
    start: str = "hot",
    seed: Optional[int] = None,
    verbose: bool = False,
) -> list[np.ndarray]:
    """Generate decorrelated gauge field configurations.

    Args:
        beta: Inverse coupling.
        L: Lattice size (LxL square).
        n_configs: Number of configurations to produce.
        n_therm: Thermalization sweeps before first measurement.
        n_skip: Sweeps between successive measurements (decorrelation).
        epsilon: Metropolis step size.
        start: 'hot' or 'cold'.
        seed: Random seed.
        verbose: Print progress.

    Returns:
        List of link arrays, each of shape (4, 2, L, L).
    """
    if seed is not None:
        np.random.seed(seed)

    config = LatticeConfig(Lx=L, Ly=L, beta=beta, seed=None)
    lattice = SU2Lattice(config, start=start)

    if verbose:
        print(f"Thermalizing (beta={beta}, L={L}, {n_therm} sweeps)...")
    thermalize(lattice, n_therm, epsilon, verbose=verbose)

    configs = []
    for i in range(n_configs):
        for _ in range(n_skip):
            metropolis_sweep(lattice, epsilon)
        configs.append(lattice.snapshot())
        if verbose and (i + 1) % 20 == 0:
            avg_p = lattice.average_plaquette()
            print(f"  Config {i+1}/{n_configs}: <P> = {avg_p:.4f}")

    return configs


def save_configs(configs: list[np.ndarray], path: str | Path) -> None:
    """Save configs as a single .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **{f"cfg_{i}": c for i, c in enumerate(configs)})


def load_configs(path: str | Path) -> list[np.ndarray]:
    """Load configs from .npz file."""
    data = np.load(str(path))
    return [data[k] for k in sorted(data.files, key=lambda s: int(s.split("_")[1]))]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SU(2) MC config generation")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--L", type=int, default=16)
    parser.add_argument("--n-configs", type=int, default=100)
    parser.add_argument("--n-therm", type=int, default=500)
    parser.add_argument("--n-skip", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="configs.npz")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    cfgs = generate_configs(
        beta=args.beta, L=args.L, n_configs=args.n_configs,
        n_therm=args.n_therm, n_skip=args.n_skip, epsilon=args.epsilon,
        seed=args.seed, verbose=args.verbose,
    )
    save_configs(cfgs, args.out)
    print(f"Saved {len(cfgs)} configs to {args.out}")
