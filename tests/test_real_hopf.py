"""Tests for the real Hopf fibration: S^0 -> S^1 -> S^1."""

import math

import numpy as np
import pytest
import torch

from hopf_layers.real import RealHopfLayer, RealHopfOutput


class TestDoubleCovering:
    """z and -z should map to the same base point."""

    def test_antipodal_same_base(self):
        """Antipodal points on S^1 have the same base angle."""
        layer = RealHopfLayer()
        # z = (cos(theta), sin(theta)) and -z = (-cos(theta), -sin(theta))
        theta = torch.tensor([0.3, 1.0, 2.5, 4.0])
        z = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        neg_z = -z

        out_z = layer(z)
        out_neg_z = layer(neg_z)

        # Base angles should be the same (mod 2*pi)
        diff = torch.remainder(out_z.base - out_neg_z.base, 2 * math.pi)
        # diff should be 0 or 2*pi
        diff = torch.min(diff, 2 * math.pi - diff)
        assert diff.max().item() < 1e-5, f"Base difference: {diff}"

    def test_antipodal_opposite_fiber(self):
        """z and -z should have opposite fiber signs."""
        layer = RealHopfLayer()
        theta = torch.tensor([0.3, 1.0, 2.5])
        z = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        neg_z = -z

        out_z = layer(z)
        out_neg_z = layer(neg_z)

        # Fiber signs should be opposite (except at x=0)
        product = out_z.fiber * out_neg_z.fiber
        assert (product <= 0).all(), "Fibers should be opposite signs"


class TestBaseRange:
    """Base angle should be in [0, 2*pi)."""

    def test_range(self):
        layer = RealHopfLayer()
        # Uniform samples around the circle
        theta = torch.linspace(0, 2 * math.pi, 100)[:-1]
        z = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        out = layer(z)
        assert out.base.min() >= 0.0
        assert out.base.max() < 2 * math.pi + 1e-6


class TestFiber:
    """Fiber should be discrete sign in {-1, +1}."""

    def test_discrete_values(self):
        layer = RealHopfLayer()
        torch.manual_seed(0)
        z = torch.randn(100, 2)
        z = z / z.norm(dim=-1, keepdim=True)
        out = layer(z)
        # All fiber values should be +1 or -1
        assert ((out.fiber == 1.0) | (out.fiber == -1.0)).all()

    def test_fiber_sign_matches_x(self):
        """Fiber should equal sign(x) of the input vector."""
        layer = RealHopfLayer()
        theta = torch.tensor([0.3, 1.0, math.pi - 0.1, math.pi + 0.1, 5.0])
        z = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        out = layer(z)
        expected_sign = torch.sign(z[..., 0])
        torch.testing.assert_close(out.fiber, expected_sign)


class TestReconstruction:
    """inverse(base, fiber) should recover the original vector."""

    def test_round_trip(self):
        """Decompose -> reconstruct should recover original."""
        layer = RealHopfLayer()
        torch.manual_seed(42)
        z = torch.randn(50, 2)
        z = z / z.norm(dim=-1, keepdim=True)

        out = layer(z)
        z_rec = layer.inverse(out.base, out.fiber)

        # Should recover original up to floating point
        torch.testing.assert_close(z_rec, z, atol=1e-5, rtol=0)

    def test_reconstruction_unit_norm(self):
        """Reconstructed vectors should have unit norm."""
        layer = RealHopfLayer()
        base = torch.rand(30) * 2 * math.pi
        fiber = torch.sign(torch.randn(30))
        fiber = torch.where(fiber == 0, torch.ones_like(fiber), fiber)
        z = layer.inverse(base, fiber)
        norms = z.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=0)


class TestS1Constraint:
    """Output base should satisfy the S^1 embedding constraint."""

    def test_base_covers_full_circle(self):
        """As input goes around S^1 once, base goes around twice (double cover).

        With evenly spaced points from 0 to 2pi (exclusive), base wraps once
        at theta=pi (the halfway point).  The total angular distance traveled
        by the base should be approximately 4*pi (two full revolutions).
        """
        layer = RealHopfLayer()
        n = 200
        theta = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        z = torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        out = layer(z)

        # Unwrap the base angle and compute total angular travel
        base_np = out.base.numpy()
        base_unwrapped = np.unwrap(base_np)
        total_travel = abs(base_unwrapped[-1] - base_unwrapped[0])

        # Should be close to 4*pi (double cover) minus one step
        expected = 4 * math.pi * (n - 1) / n
        assert abs(total_travel - expected) < 0.5, (
            f"Total travel {total_travel:.2f}, expected ~{expected:.2f}"
        )


class TestShapes:
    """Verify input/output shapes."""

    def test_1d_input(self):
        layer = RealHopfLayer()
        z = torch.tensor([1.0, 0.0])  # shape (2,)
        out = layer(z.unsqueeze(0))
        assert out.base.shape == (1,)
        assert out.fiber.shape == (1,)

    def test_batch_input(self):
        layer = RealHopfLayer()
        z = torch.randn(16, 2)
        z = z / z.norm(dim=-1, keepdim=True)
        out = layer(z)
        assert out.base.shape == (16,)
        assert out.fiber.shape == (16,)

    def test_multidim_input(self):
        layer = RealHopfLayer()
        z = torch.randn(4, 8, 8, 2)
        z = z / z.norm(dim=-1, keepdim=True)
        out = layer(z)
        assert out.base.shape == (4, 8, 8)
        assert out.fiber.shape == (4, 8, 8)

    def test_invalid_last_dim_raises(self):
        layer = RealHopfLayer()
        with pytest.raises(ValueError, match="Expected last dimension = 2"):
            layer(torch.randn(4, 3))


class TestGradientFlow:
    """Verify gradients flow through the layer."""

    def test_gradient_through_base(self):
        layer = RealHopfLayer()
        z = torch.randn(8, 2, requires_grad=True)
        out = layer(z)
        out.base.sum().backward()
        assert z.grad is not None
        assert not torch.isnan(z.grad).any()

    def test_gradient_through_inverse(self):
        layer = RealHopfLayer()
        base = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        fiber = torch.tensor([1.0, -1.0, 1.0])  # no grad for discrete
        z = layer.inverse(base, fiber)
        z.sum().backward()
        assert base.grad is not None
        assert not torch.isnan(base.grad).any()
