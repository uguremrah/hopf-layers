"""Tests for quaternion algebra operations."""

import pytest
import torch

from hopf_layers.quaternion import (
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_norm,
    quaternion_normalize,
    quaternion_to_su2,
    su2_to_quaternion,
)


class TestNormalize:
    def test_unit_norm(self, random_quaternions):
        q = quaternion_normalize(random_quaternions)
        norms = quaternion_norm(q)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-6, rtol=0)

    def test_preserves_direction(self, random_quaternions):
        q = quaternion_normalize(random_quaternions)
        # Each normalised q should be parallel to original
        for i in range(len(random_quaternions)):
            ratio = random_quaternions[i] / q[i]
            torch.testing.assert_close(
                ratio, ratio[0].expand_as(ratio), atol=1e-5, rtol=0
            )

    def test_batch_shapes(self):
        for shape in [(4,), (3, 4), (2, 5, 4)]:
            q = torch.randn(shape)
            out = quaternion_normalize(q)
            assert out.shape == shape


class TestConjugateInverse:
    def test_conjugate_sign(self, unit_quaternions):
        q = unit_quaternions
        qc = quaternion_conjugate(q)
        assert torch.allclose(qc[..., 0], q[..., 0])  # real part unchanged
        assert torch.allclose(qc[..., 1:], -q[..., 1:])  # imaginary negated

    def test_unit_inverse_equals_conjugate(self, unit_quaternions):
        qc = quaternion_conjugate(unit_quaternions)
        qi = quaternion_inverse(unit_quaternions)
        torch.testing.assert_close(qi, qc, atol=1e-6, rtol=0)

    def test_q_times_qinv_is_identity(self, unit_quaternions):
        qi = quaternion_inverse(unit_quaternions)
        prod = quaternion_multiply(unit_quaternions, qi)
        identity = torch.zeros_like(prod)
        identity[..., 0] = 1.0
        torch.testing.assert_close(prod, identity, atol=1e-5, rtol=0)


class TestMultiply:
    def test_identity_element(self, unit_quaternions):
        identity = torch.zeros_like(unit_quaternions)
        identity[..., 0] = 1.0
        # q * 1 = q
        prod = quaternion_multiply(unit_quaternions, identity)
        torch.testing.assert_close(prod, unit_quaternions, atol=1e-6, rtol=0)

    def test_non_commutativity(self):
        p = torch.tensor([0.0, 1.0, 0.0, 0.0])  # i
        q = torch.tensor([0.0, 0.0, 1.0, 0.0])  # j
        pq = quaternion_multiply(p, q)  # i*j = k
        qp = quaternion_multiply(q, p)  # j*i = -k
        assert not torch.allclose(pq, qp)
        # i*j = k
        torch.testing.assert_close(pq, torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-7, rtol=0)
        # j*i = -k
        torch.testing.assert_close(qp, torch.tensor([0.0, 0.0, 0.0, -1.0]), atol=1e-7, rtol=0)

    def test_norm_multiplicativity(self, random_quaternions):
        p = random_quaternions[:4]
        q = random_quaternions[4:]
        pq = quaternion_multiply(p, q)
        # |p*q| = |p| * |q|
        torch.testing.assert_close(
            quaternion_norm(pq),
            quaternion_norm(p) * quaternion_norm(q),
            atol=1e-5,
            rtol=0,
        )


class TestSU2Conversion:
    def test_round_trip(self, unit_quaternions):
        U = quaternion_to_su2(unit_quaternions)
        q_recovered = su2_to_quaternion(U)
        torch.testing.assert_close(q_recovered, unit_quaternions, atol=1e-6, rtol=0)

    def test_unitarity(self, unit_quaternions):
        U = quaternion_to_su2(unit_quaternions)
        UUdag = torch.matmul(U, U.conj().transpose(-2, -1))
        eye = torch.eye(2, dtype=U.dtype).expand_as(UUdag)
        torch.testing.assert_close(UUdag, eye, atol=1e-6, rtol=0)

    def test_determinant_one(self, unit_quaternions):
        U = quaternion_to_su2(unit_quaternions)
        det = U[..., 0, 0] * U[..., 1, 1] - U[..., 0, 1] * U[..., 1, 0]
        torch.testing.assert_close(det.real, torch.ones(len(unit_quaternions)), atol=1e-6, rtol=0)
        torch.testing.assert_close(det.imag, torch.zeros(len(unit_quaternions)), atol=1e-6, rtol=0)


class TestGradientFlow:
    def test_all_ops_differentiable(self):
        q = torch.randn(4, 4, requires_grad=True)
        qn = quaternion_normalize(q)
        qc = quaternion_conjugate(qn)
        prod = quaternion_multiply(qn, qc)
        U = quaternion_to_su2(qn)
        loss = prod.sum() + U.abs().sum()
        loss.backward()
        assert q.grad is not None
        assert not torch.isnan(q.grad).any()
