"""Tests for the exact Hopf inverse: (S^2, S^1) -> S^3."""

import math

import pytest
import torch

from hopf_layers.classical import ClassicalHopfLayer
from hopf_layers.reconstruction import hopf_inverse


class TestRoundTrip:
    """Decompose -> reconstruct should recover the original quaternion (up to sign)."""

    def test_round_trip_site_field(self):
        """Decompose a site field, reconstruct, then re-decompose.

        The reconstructed quaternion may differ from the original by a global
        sign (q and -q map to the same point on S^2 with the same fiber),
        but re-applying the Hopf map should recover the same base and fiber.
        """
        torch.manual_seed(42)
        q_raw = torch.randn(8, 4, 8, 8)

        layer = ClassicalHopfLayer()
        out = layer(q_raw)

        # Reconstruct: need base in (..., 3) and fiber in (...)
        # base is (B, 3, Lx, Ly) -> permute to (B, Lx, Ly, 3)
        base_last = out.base.permute(0, 2, 3, 1)
        fiber = out.fiber  # (B, Lx, Ly)

        q_reconstructed = hopf_inverse(base_last, fiber)

        # Re-decompose the reconstructed quaternion
        base_check = layer.hopf_map(q_reconstructed)
        fiber_check = layer.extract_fiber(q_reconstructed)

        torch.testing.assert_close(base_check, base_last, atol=1e-3, rtol=0)
        torch.testing.assert_close(fiber_check, fiber, atol=1e-3, rtol=0)

    def test_round_trip_unit_norm(self):
        """Reconstructed quaternions must have unit norm."""
        torch.manual_seed(0)
        q_raw = torch.randn(16, 4, 4, 4)
        layer = ClassicalHopfLayer()
        out = layer(q_raw)

        base_last = out.base.permute(0, 2, 3, 1)
        q_rec = hopf_inverse(base_last, out.fiber)
        norms = torch.sqrt((q_rec ** 2).sum(dim=-1))
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)

    def test_known_values(self):
        """For identity quaternion (1,0,0,0), base=(0,0,1), fiber=0."""
        base = torch.tensor([[0.0, 0.0, 1.0]])  # north pole
        fiber = torch.tensor([0.0])
        q = hopf_inverse(base, fiber)
        # Should be close to (1, 0, 0, 0) or (-1, 0, 0, 0)
        expected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        # Check up to sign ambiguity
        error = min(
            (q - expected).abs().max().item(),
            (q + expected).abs().max().item(),
        )
        assert error < 1e-4, f"Known-value error {error:.2e}"


class TestGradientFlow:
    def test_gradient_through_reconstruction(self):
        base = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
        fiber = torch.tensor([1.0], requires_grad=True)
        q = hopf_inverse(base, fiber)
        q.sum().backward()
        assert base.grad is not None
        assert fiber.grad is not None
        assert not torch.isnan(base.grad).any()
        assert not torch.isnan(fiber.grad).any()
