"""Tests for SU(2)+adjoint Higgs Monte Carlo."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from mc_generation.su2_higgs import (
    HiggsConfig,
    HiggsLattice,
    adjoint_rotate,
    combined_sweep,
    thermalize_higgs,
)
from mc_generation.su2_metropolis import quat_normalize, quat_conjugate, quat_multiply


class TestAdjointRotation:
    """Verify SO(3) rotation from quaternion."""

    def test_identity_rotation(self):
        """Identity quaternion should not rotate."""
        q = np.array([1.0, 0.0, 0.0, 0.0])
        phi = np.array([1.0, 2.0, 3.0])
        result = adjoint_rotate(q, phi)
        np.testing.assert_allclose(result, phi, atol=1e-12)

    def test_rotation_preserves_norm(self):
        """Adjoint rotation should preserve |phi|."""
        np.random.seed(42)
        q = quat_normalize(np.random.randn(4))
        phi = np.random.randn(3)
        result = adjoint_rotate(q, phi)
        np.testing.assert_allclose(np.linalg.norm(result),
                                   np.linalg.norm(phi), atol=1e-12)

    def test_z_rotation_by_pi(self):
        """90 degree rotation around z-axis: (1,0,0) -> (0,1,0)."""
        # q for 90° around z: (cos(pi/4), 0, 0, sin(pi/4))
        angle = np.pi / 2
        q = np.array([np.cos(angle/2), 0, 0, np.sin(angle/2)])
        phi = np.array([1.0, 0.0, 0.0])
        result = adjoint_rotate(q, phi)
        np.testing.assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-12)

    def test_composition(self):
        """R(q1) R(q2) phi = R(q1*q2) phi."""
        np.random.seed(0)
        q1 = quat_normalize(np.random.randn(4))
        q2 = quat_normalize(np.random.randn(4))
        phi = np.random.randn(3)
        # Sequential
        step = adjoint_rotate(q2, phi)
        lhs = adjoint_rotate(q1, step)
        # Combined
        rhs = adjoint_rotate(quat_multiply(q1, q2), phi)
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)


class TestHiggsLattice:
    """Verify Higgs lattice setup."""

    def test_cold_start(self):
        cfg = HiggsConfig(Lx=4, Ly=4, kappa=0.5, m2=1.0, lam=0.5)
        lat = HiggsLattice(cfg, start="cold")
        # Gauge should be identity
        assert abs(lat.gauge_order_parameter() - 1.0) < 1e-10
        # Higgs should be VEV = (0, 0, 1)
        phi = lat.get_phi(0, 0)
        np.testing.assert_allclose(phi, [0, 0, 1], atol=1e-12)

    def test_order_parameter_cold(self):
        cfg = HiggsConfig(Lx=4, Ly=4)
        lat = HiggsLattice(cfg, start="cold")
        # |phi|^2 = 1 at every site
        assert abs(lat.order_parameter() - 1.0) < 1e-10

    def test_hot_start_random(self):
        np.random.seed(42)
        cfg = HiggsConfig(Lx=8, Ly=8)
        lat = HiggsLattice(cfg, start="hot")
        # Gauge should be near zero
        assert abs(lat.gauge_order_parameter()) < 0.15
        # Higgs condensate should be O(1) from random init
        assert lat.order_parameter() > 0.5


class TestHiggsSweep:
    """Verify combined Metropolis sweep."""

    def test_acceptance_reasonable(self):
        np.random.seed(42)
        cfg = HiggsConfig(Lx=4, Ly=4, beta=2.0, kappa=0.5, m2=1.0, lam=0.5)
        lat = HiggsLattice(cfg, start="hot")
        ga, ha = combined_sweep(lat, gauge_epsilon=0.5, higgs_epsilon=0.3)
        assert 0.1 < ga < 0.99
        assert 0.1 < ha < 0.99

    def test_cold_start_stays_ordered(self):
        """Large beta+kappa should keep cold start ordered."""
        np.random.seed(42)
        cfg = HiggsConfig(Lx=4, Ly=4, beta=6.0, kappa=2.0, m2=0.1, lam=0.1)
        lat = HiggsLattice(cfg, start="cold")
        for _ in range(20):
            combined_sweep(lat, gauge_epsilon=0.3, higgs_epsilon=0.2)
        assert lat.gauge_order_parameter() > 0.7
        assert lat.order_parameter() > 0.5

    @pytest.mark.slow
    def test_phase_transition_signal(self):
        """Condensate should differ between confined and Higgs phases."""
        np.random.seed(42)

        # Confined phase: small kappa
        cfg_conf = HiggsConfig(Lx=8, Ly=8, beta=1.0, kappa=0.1,
                               m2=2.0, lam=1.0)
        lat_conf = HiggsLattice(cfg_conf, start="hot")
        thermalize_higgs(lat_conf, n_sweeps=200)
        cond_conf = lat_conf.order_parameter()

        # Higgs phase: large kappa, negative m^2
        cfg_higgs = HiggsConfig(Lx=8, Ly=8, beta=4.0, kappa=1.0,
                                m2=-1.0, lam=0.5)
        lat_higgs = HiggsLattice(cfg_higgs, start="hot")
        thermalize_higgs(lat_higgs, n_sweeps=200)
        cond_higgs = lat_higgs.order_parameter()

        # Higgs phase should have larger condensate
        assert cond_higgs > cond_conf, (
            f"Confined: {cond_conf:.4f}, Higgs: {cond_higgs:.4f}"
        )
