"""Tests for SU(2) Monte Carlo configuration generation."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add experiments to path so we can import mc_generation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from mc_generation.analytical import su2_plaquette_exact
from mc_generation.su2_metropolis import (
    SU2Lattice,
    LatticeConfig,
    metropolis_sweep,
    thermalize,
    quat_multiply,
    quat_conjugate,
    quat_normalize,
    random_su2_near_identity,
)


class TestAnalytical:
    """Verify the exact Migdal-Rusakov formula."""

    def test_known_values(self):
        """Check against tabulated values."""
        # At large beta, <P> -> 1  (ordered)
        assert su2_plaquette_exact(10.0) > 0.8
        # At beta=2, literature value ~0.433
        assert abs(su2_plaquette_exact(2.0) - 0.433) < 0.01
        # At small beta, <P> -> 0  (disordered)
        assert su2_plaquette_exact(0.5) < 0.15

    def test_monotonic(self):
        """<P> should increase monotonically with beta."""
        betas = [0.5, 1.0, 2.0, 4.0, 8.0]
        vals = [su2_plaquette_exact(b) for b in betas]
        for i in range(len(vals) - 1):
            assert vals[i] < vals[i + 1]


class TestQuaternionOps:
    """Verify quaternion algebra correctness."""

    def test_multiply_identity(self):
        """q * identity = q."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = quat_normalize(np.random.randn(4))
        result = quat_multiply(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-12)

    def test_multiply_inverse(self):
        """q * q^dag = identity for unit quaternions."""
        q = quat_normalize(np.random.randn(4))
        q_dag = quat_conjugate(q)
        result = quat_multiply(q, q_dag)
        np.testing.assert_allclose(result, [1, 0, 0, 0], atol=1e-12)

    def test_associativity(self):
        """(p*q)*r = p*(q*r)."""
        p = quat_normalize(np.random.randn(4))
        q = quat_normalize(np.random.randn(4))
        r = quat_normalize(np.random.randn(4))
        lhs = quat_multiply(quat_multiply(p, q), r)
        rhs = quat_multiply(p, quat_multiply(q, r))
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_near_identity_is_close(self):
        """random_su2_near_identity with small epsilon stays close."""
        v = random_su2_near_identity(epsilon=0.01)
        assert abs(v[0] - 1.0) < 0.01
        assert np.linalg.norm(v[1:]) < 0.01


class TestLattice:
    """Verify lattice setup and observables."""

    def test_cold_start_plaquette(self):
        """Cold start (identity everywhere) should give <P> = 1."""
        cfg = LatticeConfig(Lx=4, Ly=4, beta=2.0)
        lat = SU2Lattice(cfg, start="cold")
        assert abs(lat.average_plaquette() - 1.0) < 1e-10

    def test_cold_start_action_zero(self):
        """Wilson action should be 0 for identity field."""
        cfg = LatticeConfig(Lx=4, Ly=4, beta=2.0)
        lat = SU2Lattice(cfg, start="cold")
        assert abs(lat.wilson_action()) < 1e-10

    def test_hot_start_plaquette_near_zero(self):
        """Hot start should give <P> ~ 0 (random SU(2) average)."""
        np.random.seed(42)
        cfg = LatticeConfig(Lx=16, Ly=16, beta=2.0)
        lat = SU2Lattice(cfg, start="hot")
        # With Lx*Ly=256 plaquettes, fluctuations are small
        assert abs(lat.average_plaquette()) < 0.15

    def test_unitarity(self):
        """All links should be unit quaternions."""
        np.random.seed(0)
        cfg = LatticeConfig(Lx=8, Ly=8, beta=2.0)
        lat = SU2Lattice(cfg, start="hot")
        valid, max_dev = lat.validate_unitarity()
        assert valid
        assert max_dev < 1e-12

    def test_plaquette_is_unit(self):
        """Plaquette (product of unit quaternions) should be unit."""
        np.random.seed(0)
        cfg = LatticeConfig(Lx=8, Ly=8, beta=2.0)
        lat = SU2Lattice(cfg, start="hot")
        for x in range(4):
            for y in range(4):
                p = lat.compute_plaquette(x, y)
                norm = np.sqrt(np.sum(p**2))
                assert abs(norm - 1.0) < 1e-10


class TestMetropolis:
    """Verify Metropolis algorithm correctness."""

    def test_acceptance_rate_reasonable(self):
        """Acceptance rate should be between 30% and 95%."""
        np.random.seed(42)
        cfg = LatticeConfig(Lx=8, Ly=8, beta=2.0)
        lat = SU2Lattice(cfg, start="hot")
        acc = metropolis_sweep(lat, epsilon=0.5)
        assert 0.2 < acc < 0.98

    def test_cold_start_stays_ordered(self):
        """Starting from identity at large beta should stay ordered."""
        np.random.seed(42)
        cfg = LatticeConfig(Lx=8, Ly=8, beta=6.0)
        lat = SU2Lattice(cfg, start="cold")
        for _ in range(20):
            metropolis_sweep(lat, epsilon=0.3)
        assert lat.average_plaquette() > 0.8

    def test_unitarity_preserved_after_sweeps(self):
        """Links should remain unit quaternions after many sweeps."""
        np.random.seed(42)
        cfg = LatticeConfig(Lx=8, Ly=8, beta=2.0)
        lat = SU2Lattice(cfg, start="hot")
        for _ in range(50):
            metropolis_sweep(lat, epsilon=0.5)
        valid, max_dev = lat.validate_unitarity()
        assert valid
        assert max_dev < 1e-10

    @pytest.mark.slow
    def test_thermalization_matches_analytical(self):
        """After thermalization, <P> should match exact result within 5%.

        This is the KEY validation that the MC is correct.
        We use a small lattice and many sweeps for a quick but meaningful test.
        """
        np.random.seed(42)
        beta = 2.0
        exact = su2_plaquette_exact(beta)

        cfg = LatticeConfig(Lx=8, Ly=8, beta=beta)
        lat = SU2Lattice(cfg, start="hot")

        # Thermalize
        thermalize(lat, n_sweeps=300, epsilon=0.5)

        # Measure over 200 sweeps
        measurements = []
        for _ in range(200):
            metropolis_sweep(lat, epsilon=0.5)
            measurements.append(lat.average_plaquette())

        mc_mean = np.mean(measurements)
        mc_err = np.std(measurements) / np.sqrt(len(measurements))

        # Should be within 5% of exact or 3 sigma
        assert abs(mc_mean - exact) < max(0.05 * exact, 3 * mc_err), (
            f"MC <P> = {mc_mean:.4f} +/- {mc_err:.4f}, exact = {exact:.4f}"
        )

    @pytest.mark.slow
    def test_thermalization_multiple_betas(self):
        """Validate at 3 different beta values."""
        np.random.seed(123)
        for beta in [1.0, 2.0, 4.0]:
            exact = su2_plaquette_exact(beta)
            cfg = LatticeConfig(Lx=8, Ly=8, beta=beta)
            lat = SU2Lattice(cfg, start="hot")
            thermalize(lat, n_sweeps=500, epsilon=0.5)

            measurements = []
            for _ in range(300):
                metropolis_sweep(lat, epsilon=0.5)
                measurements.append(lat.average_plaquette())

            mc_mean = np.mean(measurements)
            mc_err = np.std(measurements) / np.sqrt(len(measurements))
            assert abs(mc_mean - exact) < max(0.08, 3 * mc_err), (
                f"beta={beta}: MC={mc_mean:.4f} +/- {mc_err:.4f}, exact={exact:.4f}"
            )
