"""Quaternion algebra operations for PyTorch with full gradient support.

Convention: q = (a0, a1, a2, a3) = a0 + a1*i + a2*j + a3*k

All functions operate on the last dimension of the input tensor,
supporting arbitrary batch dimensions via ``...`` indexing.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "quaternion_normalize",
    "quaternion_norm",
    "quaternion_conjugate",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_to_su2",
    "su2_to_quaternion",
]



def quaternion_normalize(q: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize quaternions to unit norm (project to S^3).

    Args:
        q: Quaternions of shape ``(..., 4)``.
        eps: Small constant added inside sqrt for numerical stability.

    Returns:
        Unit quaternions of shape ``(..., 4)`` satisfying ``|q| = 1``.
    """
    norm = torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True).clamp(min=eps))
    return q / norm


def quaternion_norm(q: Tensor) -> Tensor:
    """Compute quaternion norms.

    Args:
        q: Quaternions of shape ``(..., 4)``.

    Returns:
        Norms of shape ``(...)``.
    """
    return torch.sqrt(torch.sum(q * q, dim=-1))


def quaternion_conjugate(q: Tensor) -> Tensor:
    """Compute quaternion conjugate: ``q* = (a0, -a1, -a2, -a3)``.

    For unit quaternions, ``conjugate == inverse``.

    Args:
        q: Quaternions of shape ``(..., 4)``.

    Returns:
        Conjugated quaternions of shape ``(..., 4)``.
    """
    sign = torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)
    return q * sign


def quaternion_inverse(q: Tensor, eps: float = 1e-8) -> Tensor:
    """Compute quaternion inverse: ``q^{-1} = q* / |q|^2``.

    For unit quaternions this is equivalent to :func:`quaternion_conjugate`.

    Args:
        q: Quaternions of shape ``(..., 4)``.
        eps: Numerical stability constant.

    Returns:
        Inverted quaternions of shape ``(..., 4)``.
    """
    conj = quaternion_conjugate(q)
    norm_sq = torch.sum(q * q, dim=-1, keepdim=True) + eps
    return conj / norm_sq


def quaternion_multiply(p: Tensor, q: Tensor) -> Tensor:
    """Compute the Hamilton product of two quaternions.

    Implements ``(p0 + p1*i + p2*j + p3*k) * (q0 + q1*i + q2*j + q3*k)``
    using the rule ``i^2 = j^2 = k^2 = ijk = -1``.

    Args:
        p: First quaternion of shape ``(..., 4)``.
        q: Second quaternion of shape ``(..., 4)``.

    Returns:
        Hamilton product of shape ``(..., 4)``.
    """
    p0, p1, p2, p3 = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    r0 = p0 * q0 - p1 * q1 - p2 * q2 - p3 * q3
    r1 = p0 * q1 + p1 * q0 + p2 * q3 - p3 * q2
    r2 = p0 * q2 - p1 * q3 + p2 * q0 + p3 * q1
    r3 = p0 * q3 + p1 * q2 - p2 * q1 + p3 * q0

    return torch.stack([r0, r1, r2, r3], dim=-1)


def quaternion_to_su2(q: Tensor) -> Tensor:
    """Convert unit quaternion to SU(2) matrix.

    Maps ``q = a0 + a1*i + a2*j + a3*k`` to::

        U = [[a0 + i*a3,  a2 + i*a1],
             [-a2 + i*a1, a0 - i*a3]]

    Args:
        q: Unit quaternions of shape ``(..., 4)``.

    Returns:
        SU(2) matrices of shape ``(..., 2, 2)`` with complex dtype.
    """
    a0, a1, a2, a3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    U00 = torch.complex(a0, a3)
    U01 = torch.complex(a2, a1)
    U10 = torch.complex(-a2, a1)
    U11 = torch.complex(a0, -a3)

    row0 = torch.stack([U00, U01], dim=-1)
    row1 = torch.stack([U10, U11], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def su2_to_quaternion(U: Tensor) -> Tensor:
    """Extract quaternion from SU(2) matrix.

    Inverse of :func:`quaternion_to_su2`.

    Args:
        U: SU(2) matrices of shape ``(..., 2, 2)`` with complex dtype.

    Returns:
        Quaternions of shape ``(..., 4)``.
    """
    U00 = U[..., 0, 0]
    U01 = U[..., 0, 1]

    a0 = U00.real
    a3 = U00.imag
    a2 = U01.real
    a1 = U01.imag

    return torch.stack([a0, a1, a2, a3], dim=-1)
