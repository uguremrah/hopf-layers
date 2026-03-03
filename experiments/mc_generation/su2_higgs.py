"""SU(2)+adjoint Higgs Monte Carlo on a 2D lattice.

This adds a scalar Higgs field in the adjoint representation of SU(2)
to the pure gauge Wilson action.  The adjoint Higgs field lives on lattice
SITES (not links) and is a 3-component real vector representing a traceless
Hermitian 2x2 matrix: Phi = phi^a sigma^a (a=1,2,3, Pauli matrices).

The full action is:

    S = S_gauge + S_higgs

    S_gauge = beta * sum_P (1 - (1/2) Tr U_P)   [Wilson action]

    S_higgs = sum_x [
        -kappa * sum_mu Tr(Phi(x) U_mu(x) Phi(x+mu) U_mu^dag(x))
        + m2 * Tr(Phi^2)
        + lambda * (Tr(Phi^2))^2
    ]

For the adjoint representation of SU(2), gauge transport of Phi is
conjugation:  Phi -> U Phi U^dag.  In quaternion language, this is
the rotation of the imaginary part by the SO(3) matrix corresponding
to the unit quaternion U.

Phase structure (kappa-beta plane):
- Confined phase:     small beta, small kappa
- Higgs phase:        large kappa (Phi acquires VEV, breaks SU(2)->U(1))
- Coulomb phase:      large beta (ordered gauge, disordered Higgs)

The confinement-Higgs transition is the target for Experiment 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .su2_metropolis import (
        EPS,
        SU2Lattice,
        LatticeConfig,
        metropolis_sweep as gauge_sweep,
        quat_conjugate,
        quat_multiply,
        quat_normalize,
    )
except ImportError:
    from su2_metropolis import (
        EPS,
        SU2Lattice,
        LatticeConfig,
        metropolis_sweep as gauge_sweep,
        quat_conjugate,
        quat_multiply,
        quat_normalize,
    )


# ---------------------------------------------------------------------------
# Adjoint rotation: rotate 3-vector phi by quaternion q (SO(3) action)
# ---------------------------------------------------------------------------

def adjoint_rotate(q: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Rotate a 3-vector phi by quaternion q via adjoint action.

    This computes  q * (0, phi) * q^dag  where (0, phi) is a pure
    imaginary quaternion.  The result is the rotated 3-vector.

    Args:
        q: Unit quaternion (4,).
        phi: 3-vector (3,).

    Returns:
        Rotated 3-vector (3,).
    """
    # Embed phi as pure imaginary quaternion
    p = np.array([0.0, phi[0], phi[1], phi[2]])
    # q * p * q^dag
    result = quat_multiply(quat_multiply(q, p), quat_conjugate(q))
    return result[1:4]


# ---------------------------------------------------------------------------
# Higgs field on the lattice
# ---------------------------------------------------------------------------

@dataclass
class HiggsConfig:
    """Parameters for the SU(2)+adjoint Higgs system."""
    Lx: int = 16
    Ly: int = 16
    beta: float = 2.0     # gauge coupling
    kappa: float = 0.5    # hopping parameter (Higgs-gauge coupling)
    m2: float = 1.0       # mass squared
    lam: float = 0.5      # quartic coupling
    seed: Optional[int] = 42


class HiggsLattice:
    """SU(2) gauge + adjoint Higgs field on a 2D lattice.

    Attributes:
        gauge: SU2Lattice for the gauge field.
        phi:   Higgs field, shape (3, Lx, Ly) — adjoint 3-vector at each site.
    """

    def __init__(self, config: HiggsConfig, start: str = "cold"):
        self.config = config
        self.Lx = config.Lx
        self.Ly = config.Ly

        # Gauge field
        gauge_cfg = LatticeConfig(Lx=config.Lx, Ly=config.Ly,
                                  beta=config.beta, seed=None)
        self.gauge = SU2Lattice(gauge_cfg, start=start)

        # Higgs field
        if start == "cold":
            # VEV in the 3-direction:  phi = (0, 0, v) with v=1
            self.phi = np.zeros((3, self.Lx, self.Ly))
            self.phi[2] = 1.0
        elif start == "hot":
            self.phi = np.random.randn(3, self.Lx, self.Ly)
        else:
            raise ValueError(f"start must be 'cold' or 'hot', got {start!r}")

    def get_phi(self, x: int, y: int) -> np.ndarray:
        return self.phi[:, x % self.Lx, y % self.Ly].copy()

    def set_phi(self, x: int, y: int, v: np.ndarray) -> None:
        self.phi[:, x % self.Lx, y % self.Ly] = v

    # --- Observables ---

    def higgs_hopping(self, x: int, y: int, mu: int) -> float:
        """Kinetic (hopping) term: Tr(Phi(x) U_mu(x) Phi(x+mu) U_mu^dag(x)).

        In the adjoint representation this is:
            phi(x) . R(U_mu) phi(x+mu)
        where R(U) is the SO(3) rotation corresponding to quaternion U.
        Using Tr(sigma_a sigma_b) = 2 delta_ab, we have:
            Tr(Phi(x) U Phi(x+mu) U^dag) = 2 * phi(x) . R(U) phi(x+mu)
        """
        phi_x = self.get_phi(x, y)
        q = self.gauge.get_link(mu, x, y)

        # Shifted site
        if mu == 0:
            phi_xmu = self.get_phi(x + 1, y)
        else:
            phi_xmu = self.get_phi(x, y + 1)

        # Rotate phi(x+mu) by U_mu
        rotated = adjoint_rotate(q, phi_xmu)
        return 2.0 * np.dot(phi_x, rotated)

    def higgs_potential(self, x: int, y: int) -> float:
        """V(phi) = m^2 |phi|^2 + lambda |phi|^4."""
        phi = self.get_phi(x, y)
        phi_sq = np.dot(phi, phi)
        return self.config.m2 * phi_sq + self.config.lam * phi_sq**2

    def higgs_action(self) -> float:
        """Total Higgs action: sum_x [-kappa * hopping + potential]."""
        total = 0.0
        for x in range(self.Lx):
            for y in range(self.Ly):
                # Hopping in both directions
                for mu in range(2):
                    total -= self.config.kappa * self.higgs_hopping(x, y, mu)
                # Potential
                total += self.higgs_potential(x, y)
        return total

    def total_action(self) -> float:
        """S_gauge + S_higgs."""
        return self.gauge.wilson_action() + self.higgs_action()

    def order_parameter(self) -> float:
        """Higgs condensate: <|phi|^2> averaged over sites.

        Large in Higgs phase, small in confined phase.
        """
        phi_sq = np.sum(self.phi**2, axis=0)  # (Lx, Ly)
        return float(np.mean(phi_sq))

    def gauge_order_parameter(self) -> float:
        """Average plaquette from gauge sector."""
        return self.gauge.average_plaquette()

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return copies of (gauge_links, higgs_field)."""
        return self.gauge.snapshot(), self.phi.copy()


# ---------------------------------------------------------------------------
# Local Higgs action for Metropolis
# ---------------------------------------------------------------------------

def _local_higgs_action_site(lat: HiggsLattice, x: int, y: int) -> float:
    """Higgs action contribution from site (x,y) and its neighbors.

    Includes:
    - Potential at (x,y)
    - Hopping terms connecting (x,y) to its 4 nearest neighbors
    """
    total = lat.higgs_potential(x, y)

    # Forward hopping: (x,y) -> (x+1,y) and (x,y) -> (x,y+1)
    for mu in range(2):
        total -= lat.config.kappa * lat.higgs_hopping(x, y, mu)

    # Backward hopping: (x-1,y) -> (x,y) and (x,y-1) -> (x,y)
    # Link (x-1,y) in direction 0 connects site (x-1,y) to site (x,y)
    for mu, (dx, dy) in enumerate([(1, 0), (0, 1)]):
        x_back = x - dx
        y_back = y - dy
        total -= lat.config.kappa * lat.higgs_hopping(x_back, y_back, mu)

    return total


def _local_higgs_action_link(lat: HiggsLattice, mu: int, x: int, y: int) -> float:
    """Higgs action contribution from link U_mu(x,y).

    The link enters one hopping term: phi(x) . R(U_mu) phi(x+mu).
    """
    return -lat.config.kappa * lat.higgs_hopping(x, y, mu)


# ---------------------------------------------------------------------------
# Metropolis sweeps
# ---------------------------------------------------------------------------

def higgs_site_sweep(lat: HiggsLattice, epsilon: float = 0.3) -> float:
    """Metropolis sweep over Higgs field sites."""
    n_accepted = 0
    n_total = lat.Lx * lat.Ly

    for x in range(lat.Lx):
        for y in range(lat.Ly):
            phi_old = lat.get_phi(x, y)  # already a copy
            s_old = _local_higgs_action_site(lat, x, y)

            # Propose: phi_new = phi_old + epsilon * random
            delta = epsilon * np.random.randn(3)
            phi_new = phi_old + delta
            lat.set_phi(x, y, phi_new)

            s_new = _local_higgs_action_site(lat, x, y)
            delta_s = s_new - s_old

            if delta_s < 0 or np.random.random() < np.exp(-delta_s):
                n_accepted += 1
            else:
                lat.set_phi(x, y, phi_old)

    return n_accepted / n_total


def combined_sweep(lat: HiggsLattice,
                   gauge_epsilon: float = 0.5,
                   higgs_epsilon: float = 0.3) -> tuple[float, float]:
    """One combined sweep: gauge links + Higgs sites.

    Returns (gauge_acceptance, higgs_acceptance).
    """
    gauge_acc = gauge_sweep(lat.gauge, gauge_epsilon)
    higgs_acc = higgs_site_sweep(lat, higgs_epsilon)
    return gauge_acc, higgs_acc


# ---------------------------------------------------------------------------
# Thermalization and config generation
# ---------------------------------------------------------------------------

@dataclass
class HiggsThermHistory:
    plaquettes: list[float] = field(default_factory=list)
    condensates: list[float] = field(default_factory=list)
    gauge_acc: list[float] = field(default_factory=list)
    higgs_acc: list[float] = field(default_factory=list)


def thermalize_higgs(
    lat: HiggsLattice,
    n_sweeps: int = 500,
    gauge_epsilon: float = 0.5,
    higgs_epsilon: float = 0.3,
    verbose: bool = False,
) -> HiggsThermHistory:
    """Thermalize the gauge+Higgs system."""
    history = HiggsThermHistory()
    for i in range(n_sweeps):
        ga, ha = combined_sweep(lat, gauge_epsilon, higgs_epsilon)
        history.gauge_acc.append(ga)
        history.higgs_acc.append(ha)
        if (i + 1) % 20 == 0 or i == 0:
            p = lat.gauge_order_parameter()
            c = lat.order_parameter()
            history.plaquettes.append(p)
            history.condensates.append(c)
            if verbose:
                print(f"  Sweep {i+1:4d}: <P>={p:.4f}, <|phi|^2>={c:.4f}, "
                      f"gacc={ga:.2%}, hacc={ha:.2%}")
    return history


def generate_higgs_configs(
    config: HiggsConfig,
    n_configs: int = 100,
    n_therm: int = 500,
    n_skip: int = 10,
    gauge_epsilon: float = 0.5,
    higgs_epsilon: float = 0.3,
    start: str = "hot",
    verbose: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate decorrelated gauge+Higgs configurations.

    Returns list of (gauge_links, higgs_phi) tuples.
    """
    if config.seed is not None:
        np.random.seed(config.seed)

    lat = HiggsLattice(config, start=start)

    if verbose:
        print(f"Thermalizing (beta={config.beta}, kappa={config.kappa})...")
    thermalize_higgs(lat, n_therm, gauge_epsilon, higgs_epsilon, verbose=verbose)

    configs = []
    for i in range(n_configs):
        for _ in range(n_skip):
            combined_sweep(lat, gauge_epsilon, higgs_epsilon)
        configs.append(lat.snapshot())
        if verbose and (i + 1) % 20 == 0:
            print(f"  Config {i+1}/{n_configs}: <P>={lat.gauge_order_parameter():.4f}, "
                  f"<|phi|^2>={lat.order_parameter():.4f}")

    return configs


def save_higgs_configs(configs: list[tuple[np.ndarray, np.ndarray]],
                       path: str | Path) -> None:
    """Save gauge+Higgs configs as .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for i, (gauge, higgs) in enumerate(configs):
        data[f"gauge_{i}"] = gauge
        data[f"higgs_{i}"] = higgs
    np.savez_compressed(str(path), **data)


def load_higgs_configs(path: str | Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Load gauge+Higgs configs from .npz."""
    data = np.load(str(path))
    n = len([k for k in data.files if k.startswith("gauge_")])
    return [(data[f"gauge_{i}"], data[f"higgs_{i}"]) for i in range(n)]
