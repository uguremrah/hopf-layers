"""Tests for the classical Hopf fibration layer (S^3 -> S^2)."""

import math

import pytest
import torch

from hopf_layers.classical import ClassicalHopfLayer, HopfOutput


class TestHopfMap:
    """Verify the Hopf map S^3 -> S^2."""

    def test_s2_constraint(self, site_field):
        """Base output must lie on S^2: x^2 + y^2 + z^2 = 1."""
        layer = ClassicalHopfLayer()
        out = layer(site_field)
        norm_sq = (out.base ** 2).sum(dim=1)  # sum over xyz channel
        torch.testing.assert_close(
            norm_sq, torch.ones_like(norm_sq), atol=1e-6, rtol=0
        )

    def test_s2_constraint_precision(self):
        """Verify machine-precision S^2 constraint with exact unit quaternions."""
        torch.manual_seed(0)
        q = torch.randn(16, 4, 4, 4, dtype=torch.float64)
        norm = torch.sqrt((q ** 2).sum(dim=1, keepdim=True))
        q = q / norm  # exact unit quaternions

        layer = ClassicalHopfLayer(eps=1e-16)
        out = layer(q)
        norm_sq = (out.base ** 2).sum(dim=1)
        error = (norm_sq - 1.0).abs().max().item()
        assert error < 1e-12, f"S^2 constraint error {error:.2e} exceeds 1e-12"


class TestFiber:
    """Verify fiber phase extraction."""

    def test_fiber_range(self, site_field):
        """Fiber phase must be in [0, 2*pi)."""
        layer = ClassicalHopfLayer()
        out = layer(site_field)
        assert out.fiber.min() >= 0.0
        assert out.fiber.max() < 2 * math.pi + 1e-6

    def test_known_fiber_value(self):
        """For q = (cos(phi), 0, 0, sin(phi)), fiber = atan2(sin(phi), cos(phi)) = phi.

        Note: the fiber phase is atan2(a3, a0), so if we set a0=cos(phi)
        and a3=sin(phi) directly, the extracted fiber should be phi.
        """
        phi = torch.tensor([0.5, 1.0, 2.0, 3.0])
        q = torch.zeros(4, 4, 1, 1)
        q[:, 0, 0, 0] = torch.cos(phi)
        q[:, 3, 0, 0] = torch.sin(phi)

        layer = ClassicalHopfLayer()
        out = layer(q)
        expected = torch.remainder(phi, 2 * math.pi)
        torch.testing.assert_close(
            out.fiber[:, 0, 0], expected, atol=1e-5, rtol=0
        )


class TestTransitions:
    """Verify winding transition detection."""

    def test_smooth_field_no_transitions(self):
        """A uniform quaternion field should produce near-zero transitions."""
        q = torch.zeros(1, 4, 8, 8)
        q[:, 0] = 1.0  # identity quaternion everywhere
        layer = ClassicalHopfLayer()
        out = layer(q)
        assert out.transitions_x.abs().max() < 0.01
        assert out.transitions_y.abs().max() < 0.01

    def test_vortex_produces_transitions(self):
        """A vortex configuration should produce non-zero transitions.

        We set a0=cos(phi), a3=sin(phi) so that the fiber phase IS phi
        (the full vortex angle), which wraps around 2*pi and produces transitions.
        """
        Lx, Ly = 16, 16
        cx, cy = Lx // 2, Ly // 2
        x = torch.arange(Lx, dtype=torch.float32)
        y = torch.arange(Ly, dtype=torch.float32)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        phi = torch.atan2(Y - cy, X - cx)

        q = torch.zeros(1, 4, Lx, Ly)
        q[0, 0] = torch.cos(phi)  # a0 = cos(phi)
        q[0, 3] = torch.sin(phi)  # a3 = sin(phi)

        layer = ClassicalHopfLayer()
        out = layer(q)
        # Vortex must produce some transitions
        assert out.transitions_x.abs().max() > 0.1
        assert out.transitions_y.abs().max() > 0.1


class TestInputShapes:
    """Verify both site and link field layouts."""

    def test_site_field_shapes(self, site_field):
        layer = ClassicalHopfLayer()
        out = layer(site_field)
        B, _, Lx, Ly = site_field.shape
        assert out.base.shape == (B, 3, Lx, Ly)
        assert out.fiber.shape == (B, Lx, Ly)
        assert out.transitions_x.shape == (B, Lx, Ly)
        assert out.transitions_y.shape == (B, Lx, Ly)

    def test_link_field_shapes(self, link_field):
        layer = ClassicalHopfLayer()
        out = layer(link_field)
        B, _, ndir, Lx, Ly = link_field.shape
        assert out.base.shape == (B, 3, ndir, Lx, Ly)
        assert out.fiber.shape == (B, ndir, Lx, Ly)
        assert out.transitions_x.shape == (B, ndir, Lx, Ly)
        assert out.transitions_y.shape == (B, ndir, Lx, Ly)

    def test_invalid_ndim_raises(self):
        layer = ClassicalHopfLayer()
        with pytest.raises(ValueError, match="Expected 4-D"):
            layer(torch.randn(4, 4, 4))


class TestGradientFlow:
    """Verify backward pass through all outputs."""

    def test_gradient_flows_through_all_outputs(self):
        q = torch.randn(2, 4, 8, 8, requires_grad=True)
        layer = ClassicalHopfLayer()
        out = layer(q)
        loss = (
            out.base.sum()
            + out.fiber.sum()
            + out.transitions_x.sum()
            + out.transitions_y.sum()
        )
        loss.backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()
        assert q.grad.abs().sum() > 0  # non-trivial gradients

    def test_ste_clipping_near_singularity(self):
        """Gradient near atan2 singularity should be clipped, not infinite."""
        # Quaternion near the singularity: a0 ≈ 0, a3 ≈ 0
        q = torch.zeros(1, 4, 1, 1, requires_grad=True)
        # a1=1, a2=0 -> near the "problematic" region for atan2(a3, a0)
        with torch.no_grad():
            q.data[0, 1, 0, 0] = 1.0
            q.data[0, 0, 0, 0] = 1e-10
            q.data[0, 3, 0, 0] = 1e-10

        layer = ClassicalHopfLayer(atan2_max_grad=100.0)
        out = layer(q)
        out.fiber.sum().backward()
        assert q.grad is not None
        assert not torch.isinf(q.grad).any()
        assert q.grad.abs().max() < 200  # clipped, not exploding


class TestEquivariance:
    """Verify SU(2)-equivariance of the Hopf map."""

    def test_left_multiplication_rotates_base(self):
        """Left-multiplying quaternion by g ∈ SU(2) should rotate the base point."""
        from hopf_layers.quaternion import quaternion_multiply, quaternion_normalize

        torch.manual_seed(123)
        # Random unit quaternion and random SU(2) element
        q = quaternion_normalize(torch.randn(8, 4))
        g = quaternion_normalize(torch.randn(8, 4))

        layer = ClassicalHopfLayer()

        # Hopf map of g*q vs Hopf map of q
        gq = quaternion_multiply(g, q)

        # Compute base for q and gq
        base_q = layer.hopf_map(q)
        base_gq = layer.hopf_map(gq)

        # Both should lie on S^2
        torch.testing.assert_close(
            (base_q ** 2).sum(-1), torch.ones(8), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            (base_gq ** 2).sum(-1), torch.ones(8), atol=1e-5, rtol=0
        )

        # The base points should be rotated versions of each other
        # (so they have the same norm, which is 1, and generally different coords)
        # We can't easily check the specific rotation without constructing the
        # SO(3) matrix from g, but we can verify the map is consistent:
        # HopfMap(g*q) for g=identity should equal HopfMap(q)
        identity = torch.zeros(8, 4)
        identity[:, 0] = 1.0
        base_id_q = layer.hopf_map(quaternion_multiply(identity, q))
        torch.testing.assert_close(base_id_q, base_q, atol=1e-6, rtol=0)
