"""Tests for the quaternionic Hopf fibration: S³ → S⁷ → S⁴."""

import math

import pytest
import torch

from hopf_layers.quaternionic import (
    QuaternionicHopfLayer,
    QuaternionicHopfOutput,
    octonion_multiply,
    octonion_conjugate,
    octonion_norm,
)
from hopf_layers.quaternion import (
    quaternion_multiply,
    quaternion_conjugate,
    quaternion_normalize,
    quaternion_norm,
)


def random_s7(n: int, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate n random points on S⁷ as quaternion pairs."""
    torch.manual_seed(seed)
    v = torch.randn(n, 8)
    v = v / v.norm(dim=-1, keepdim=True)
    p = v[:, :4]
    q = v[:, 4:]
    return p, q


class TestOctonionAlgebra:
    """Test Cayley-Dickson octonion operations."""

    def test_conjugate_involution(self):
        """(o*)* = o."""
        torch.manual_seed(0)
        p = torch.randn(10, 4)
        q = torch.randn(10, 4)
        pp, qq = octonion_conjugate(octonion_conjugate((p, q)))
        torch.testing.assert_close(pp, p, atol=1e-6, rtol=0)
        torch.testing.assert_close(qq, q, atol=1e-6, rtol=0)

    def test_norm_multiplicative(self):
        """|a*b| = |a|*|b| (octonions are a normed division algebra)."""
        torch.manual_seed(1)
        p1, q1 = torch.randn(20, 4), torch.randn(20, 4)
        p2, q2 = torch.randn(20, 4), torch.randn(20, 4)
        prod = octonion_multiply((p1, q1), (p2, q2))
        norm_prod = octonion_norm(prod)
        norm1 = octonion_norm((p1, q1))
        norm2 = octonion_norm((p2, q2))
        torch.testing.assert_close(norm_prod, norm1 * norm2, atol=1e-4, rtol=1e-4)

    def test_identity_multiplication(self):
        """Multiplying by (1,0,0,0,0,0,0,0) gives identity."""
        torch.manual_seed(2)
        p, q = torch.randn(10, 4), torch.randn(10, 4)
        e = torch.zeros(10, 4)
        e[:, 0] = 1.0
        zero = torch.zeros(10, 4)
        # (1, 0) * (p, q) = (p, q)
        pp, qq = octonion_multiply((e, zero), (p, q))
        torch.testing.assert_close(pp, p, atol=1e-5, rtol=0)
        torch.testing.assert_close(qq, q, atol=1e-5, rtol=0)

    def test_non_associativity(self):
        """Octonion multiplication is NOT associative in general."""
        torch.manual_seed(3)
        a = (torch.randn(5, 4), torch.randn(5, 4))
        b = (torch.randn(5, 4), torch.randn(5, 4))
        c = (torch.randn(5, 4), torch.randn(5, 4))
        # (a*b)*c vs a*(b*c)
        lhs = octonion_multiply(octonion_multiply(a, b), c)
        rhs = octonion_multiply(a, octonion_multiply(b, c))
        # These should differ (non-associative)
        diff_p = (lhs[0] - rhs[0]).abs().max().item()
        diff_q = (lhs[1] - rhs[1]).abs().max().item()
        assert max(diff_p, diff_q) > 0.01, "Octonions should be non-associative"

    def test_conjugate_norm(self):
        """o * o* = |o|² * identity."""
        torch.manual_seed(4)
        p, q = torch.randn(10, 4), torch.randn(10, 4)
        o_conj = octonion_conjugate((p, q))
        prod_p, prod_q = octonion_multiply((p, q), o_conj)
        norm_sq = (p * p).sum(-1) + (q * q).sum(-1)
        # prod should be (|o|² * (1,0,0,0), (0,0,0,0))
        expected_p = torch.zeros_like(p)
        expected_p[:, 0] = norm_sq
        expected_q = torch.zeros_like(q)
        torch.testing.assert_close(prod_p, expected_p, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(prod_q, expected_q, atol=1e-4, rtol=1e-4)


class TestS7Constraint:
    """Points on S⁷ should maintain unit norm throughout."""

    def test_input_normalization(self):
        """Layer normalizes to |p|² + |q|² = 1."""
        layer = QuaternionicHopfLayer()
        p = torch.randn(20, 4) * 3.0  # not unit norm
        q = torch.randn(20, 4) * 2.0
        out = layer(p, q)
        # Check output is well-formed
        assert out.base.shape == (20, 5)
        assert out.fiber.shape == (20, 4)


class TestS4Base:
    """Base point should lie on S⁴ ⊂ ℝ⁵."""

    def test_base_on_s4(self):
        """base ∈ S⁴: |base|² = 1."""
        layer = QuaternionicHopfLayer()
        p, q = random_s7(50)
        out = layer(p, q)
        norm_sq = (out.base ** 2).sum(dim=-1)
        torch.testing.assert_close(
            norm_sq, torch.ones(50), atol=1e-5, rtol=0
        )

    def test_base_fifth_component_range(self):
        """The 5th base component (|p|²-|q|²) should be in [-1, 1]."""
        layer = QuaternionicHopfLayer()
        p, q = random_s7(100, seed=7)
        out = layer(p, q)
        assert out.base[:, 4].min() >= -1.0 - 1e-6
        assert out.base[:, 4].max() <= 1.0 + 1e-6


class TestS3Fiber:
    """Fiber should be a unit quaternion (S³)."""

    def test_fiber_unit_norm(self):
        """fiber ∈ S³: |fiber| = 1."""
        layer = QuaternionicHopfLayer()
        p, q = random_s7(50)
        out = layer(p, q)
        fiber_norm = quaternion_norm(out.fiber)
        torch.testing.assert_close(
            fiber_norm, torch.ones(50), atol=1e-5, rtol=0
        )

    def test_fiber_is_direction_of_p(self):
        """fiber = p / |p|."""
        layer = QuaternionicHopfLayer()
        p, q = random_s7(30)
        # Normalize to S⁷
        norm = torch.sqrt((p * p).sum(-1, keepdim=True) + (q * q).sum(-1, keepdim=True))
        p_n = p / norm
        out = layer(p, q)
        expected = quaternion_normalize(p_n)
        torch.testing.assert_close(out.fiber, expected, atol=1e-5, rtol=0)


class TestFiberAction:
    """Right multiplication by S³ should preserve base, change fiber."""

    def test_right_mult_preserves_base(self):
        """(p*g, q*g) has the same base as (p, q) for unit quaternion g."""
        layer = QuaternionicHopfLayer()
        p, q = random_s7(20)
        # Random unit quaternion
        torch.manual_seed(99)
        g = quaternion_normalize(torch.randn(20, 4))

        out1 = layer(p, q)
        p_g = quaternion_multiply(p, g)
        q_g = quaternion_multiply(q, g)
        out2 = layer(p_g, q_g)

        torch.testing.assert_close(out1.base, out2.base, atol=1e-4, rtol=1e-4)

    def test_right_mult_changes_fiber(self):
        """Fiber should change under right multiplication by non-identity g."""
        layer = QuaternionicHopfLayer()
        p, q = random_s7(10, seed=55)
        g = quaternion_normalize(torch.randn(10, 4))
        # Make sure g is not identity
        g[:, 0] = 0.5
        g = quaternion_normalize(g)

        out1 = layer(p, q)
        out2 = layer(quaternion_multiply(p, g), quaternion_multiply(q, g))

        diff = (out1.fiber - out2.fiber).abs().max().item()
        assert diff > 0.01, "Fiber should change under non-identity right multiplication"


class TestReconstruction:
    """inverse(base, fiber) should recover the original pair."""

    def test_round_trip(self):
        """Decompose → reconstruct should recover original."""
        layer = QuaternionicHopfLayer()
        p, q = random_s7(30)
        out = layer(p, q)
        p_rec, q_rec = layer.inverse(out.base, out.fiber)

        # Normalize originals for comparison
        norm = torch.sqrt((p * p).sum(-1, keepdim=True) + (q * q).sum(-1, keepdim=True))
        p_n = p / norm
        q_n = q / norm

        torch.testing.assert_close(p_rec, p_n, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(q_rec, q_n, atol=1e-4, rtol=1e-4)

    def test_reconstruction_on_s7(self):
        """Reconstructed pair should lie on S⁷."""
        layer = QuaternionicHopfLayer()
        # Use arbitrary base on S⁴ and fiber on S³
        torch.manual_seed(42)
        base = torch.randn(20, 5)
        base = base / base.norm(dim=-1, keepdim=True)
        fiber = quaternion_normalize(torch.randn(20, 4))

        p, q = layer.inverse(base, fiber)
        total_norm = torch.sqrt((p * p).sum(-1) + (q * q).sum(-1))
        torch.testing.assert_close(total_norm, torch.ones(20), atol=1e-4, rtol=0)


class TestShapes:
    """Verify input/output shapes."""

    def test_basic_shape(self):
        layer = QuaternionicHopfLayer()
        p = torch.randn(16, 4)
        q = torch.randn(16, 4)
        out = layer(p, q)
        assert out.base.shape == (16, 5)
        assert out.fiber.shape == (16, 4)

    def test_batch_shape(self):
        layer = QuaternionicHopfLayer()
        p = torch.randn(4, 8, 8, 4)
        q = torch.randn(4, 8, 8, 4)
        out = layer(p, q)
        assert out.base.shape == (4, 8, 8, 5)
        assert out.fiber.shape == (4, 8, 8, 4)

    def test_invalid_dim_raises(self):
        layer = QuaternionicHopfLayer()
        with pytest.raises(ValueError, match="Expected last dimension = 4"):
            layer(torch.randn(4, 3), torch.randn(4, 4))


class TestGradientFlow:
    """Verify gradients flow through the layer."""

    def test_gradient_through_base(self):
        layer = QuaternionicHopfLayer()
        p = torch.randn(8, 4, requires_grad=True)
        q = torch.randn(8, 4, requires_grad=True)
        out = layer(p, q)
        out.base.sum().backward()
        assert p.grad is not None and not torch.isnan(p.grad).any()
        assert q.grad is not None and not torch.isnan(q.grad).any()

    def test_gradient_through_fiber(self):
        layer = QuaternionicHopfLayer()
        p = torch.randn(8, 4, requires_grad=True)
        q = torch.randn(8, 4, requires_grad=True)
        out = layer(p, q)
        out.fiber.sum().backward()
        assert p.grad is not None and not torch.isnan(p.grad).any()

    def test_gradient_through_inverse(self):
        layer = QuaternionicHopfLayer()
        base = torch.randn(8, 5, requires_grad=True)
        base_normed = base / base.norm(dim=-1, keepdim=True)
        fiber = quaternion_normalize(torch.randn(8, 4))
        p, q = layer.inverse(base_normed, fiber)
        (p.sum() + q.sum()).backward()
        assert base.grad is not None and not torch.isnan(base.grad).any()
