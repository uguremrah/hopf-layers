"""Quaternionic Hopf fibration: S³ → S⁷ → S⁴.

The quaternionic Hopf map sends a unit vector in ℍ² (i.e. a point on S⁷,
represented as a pair of quaternions (p, q) with |p|² + |q|² = 1) to a
point on S⁴ via the projective construction.

**Cayley-Dickson representation**: An octonion is a pair of quaternions
(p, q), with multiplication defined by:
    (p, q)(r, s) = (pr - s*q, sp + qr*)
where * denotes quaternion conjugation.

**Quaternionic Hopf map** (S⁷ → S⁴):
    (p, q) ↦ (2p q*, |p|² - |q|²) ∈ ℝ⁴ × ℝ = ℝ⁵

The base point lies on S⁴ ⊂ ℝ⁵. The fiber over each base point is S³
(the group of unit quaternions acting by right multiplication).

**Fiber extraction**: Given (p, q) on S⁷, the S³ fiber element is
extracted as the unit quaternion g such that (p, q) = (p₀, q₀) · g
for some canonical representative (p₀, q₀) of the base point.
In practice, we use g = p / |p| when |p| > 0.

This is the second in the family:
    S⁰ → S¹ → S¹   (real, hopf_layers.real)
    S¹ → S³ → S²   (classical, hopf_layers.classical)
    S³ → S⁷ → S⁴   (quaternionic, this module)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn as nn

from hopf_layers.quaternion import (
    quaternion_conjugate,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_norm,
)
from hopf_layers.utils import EPS

__all__ = [
    "QuaternionicHopfLayer",
    "QuaternionicHopfOutput",
    "octonion_multiply",
    "octonion_conjugate",
    "octonion_norm",
]


# ---------------------------------------------------------------------------
# Cayley-Dickson octonion operations on quaternion pairs
# ---------------------------------------------------------------------------

def octonion_multiply(a: tuple[Tensor, Tensor], b: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Cayley-Dickson multiplication of two octonions.

    Each octonion is a pair ``(p, q)`` of quaternions, both of shape ``(..., 4)``.
    The product is:
        ``(p, q)(r, s) = (pr - s̄q, sp + qr̄)``
    where ``s̄`` and ``r̄`` denote quaternion conjugation.

    Note: Octonion multiplication is **not** associative.

    Args:
        a: Tuple ``(p, q)`` — first octonion.
        b: Tuple ``(r, s)`` — second octonion.

    Returns:
        Tuple ``(p', q')`` — product octonion.
    """
    p, q = a
    r, s = b
    # (pr - conj(s)*q, s*p + q*conj(r))
    p_out = quaternion_multiply(p, r) - quaternion_multiply(quaternion_conjugate(s), q)
    q_out = quaternion_multiply(s, p) + quaternion_multiply(q, quaternion_conjugate(r))
    return (p_out, q_out)


def octonion_conjugate(o: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """Conjugate an octonion: ``(p, q)* = (p*, -q)``.

    Args:
        o: Tuple ``(p, q)`` of shape ``(..., 4)`` each.

    Returns:
        Conjugated octonion ``(p*, -q)``.
    """
    p, q = o
    return (quaternion_conjugate(p), -q)


def octonion_norm(o: tuple[Tensor, Tensor]) -> Tensor:
    """Compute octonion norm: ``|o|² = |p|² + |q|²``.

    Args:
        o: Tuple ``(p, q)`` of shape ``(..., 4)`` each.

    Returns:
        Norms of shape ``(...)``.
    """
    p, q = o
    return torch.sqrt(
        (p * p).sum(dim=-1) + (q * q).sum(dim=-1)
    )


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

class QuaternionicHopfOutput:
    """Output of the quaternionic Hopf fibration decomposition.

    Attributes:
        base:  Base point on S⁴ ⊂ ℝ⁵,  shape ``(..., 5)``.
        fiber: Unit quaternion in S³,    shape ``(..., 4)``.
    """

    __slots__ = ("base", "fiber")

    def __init__(self, base: Tensor, fiber: Tensor):
        self.base = base
        self.fiber = fiber


# ---------------------------------------------------------------------------
# Main layer
# ---------------------------------------------------------------------------

class QuaternionicHopfLayer(nn.Module):
    """Quaternionic Hopf fibration layer: S⁷ → (S⁴, S³).

    Decomposes 8D unit vectors (points on S⁷, encoded as quaternion pairs
    ``(p, q)`` each of shape ``(..., 4)``) into:

    - **base**: point on S⁴ ⊂ ℝ⁵
    - **fiber**: unit quaternion in S³ ⊂ ℝ⁴

    The map: for ``(p, q)`` with ``|p|² + |q|² = 1``:

        base = (2·Re(pq*), 2·Im_i(pq*), 2·Im_j(pq*), 2·Im_k(pq*), |p|²-|q|²)
             = (2pq*, |p|²-|q|²) ∈ ℝ⁵

        fiber = p / |p|   (unit quaternion, the S³ fiber element)

    Reconstruction:
        Given (base, fiber), recover (p, q) by finding the canonical
        lift and applying the fiber rotation.

    Args:
        eps: Small constant for numerical stability.
    """

    def __init__(self, eps: float = EPS):
        super().__init__()
        self.eps = eps

    def forward(self, p: Tensor, q: Tensor) -> QuaternionicHopfOutput:
        """Decompose a quaternion pair (point on S⁷) via the quaternionic Hopf map.

        Args:
            p: First quaternion component, shape ``(..., 4)``.
            q: Second quaternion component, shape ``(..., 4)``.

        Returns:
            QuaternionicHopfOutput with base (S⁴) and fiber (S³).
        """
        if p.shape[-1] != 4 or q.shape[-1] != 4:
            raise ValueError(
                f"Expected last dimension = 4 for both inputs, "
                f"got p.shape={p.shape}, q.shape={q.shape}"
            )

        # Normalize to S⁷: |p|² + |q|² = 1
        norm_sq = (p * p).sum(dim=-1, keepdim=True) + (q * q).sum(dim=-1, keepdim=True)
        norm = torch.sqrt(norm_sq.clamp(min=self.eps))
        p = p / norm
        q = q / norm

        # --- Base: S⁴ projection ---
        # pq* is a quaternion: (a0, a1, a2, a3)
        pq_star = quaternion_multiply(p, quaternion_conjugate(q))

        # |p|² - |q|²
        p_norm_sq = (p * p).sum(dim=-1)
        q_norm_sq = (q * q).sum(dim=-1)
        diff = p_norm_sq - q_norm_sq  # scalar

        # base = (2*pq*[0], 2*pq*[1], 2*pq*[2], 2*pq*[3], |p|²-|q|²) ∈ ℝ⁵
        base = torch.cat([2.0 * pq_star, diff.unsqueeze(-1)], dim=-1)

        # --- Fiber: S³ element ---
        # g = p / |p| (the quaternionic phase)
        p_norm = torch.sqrt((p * p).sum(dim=-1, keepdim=True).clamp(min=self.eps))
        fiber = p / p_norm

        return QuaternionicHopfOutput(base=base, fiber=fiber)

    def inverse(self, base: Tensor, fiber: Tensor) -> tuple[Tensor, Tensor]:
        """Reconstruct quaternion pair (p, q) from base and fiber.

        Args:
            base: Point on S⁴, shape ``(..., 5)``.
            fiber: Unit quaternion (S³ fiber), shape ``(..., 4)``.

        Returns:
            Tuple ``(p, q)`` with ``|p|² + |q|² = 1``.
        """
        # Extract components from base
        # base = (2*pq*_0, 2*pq*_1, 2*pq*_2, 2*pq*_3, |p|²-|q|²)
        pq_star_2 = base[..., :4]  # 2*pq*
        diff = base[..., 4]        # |p|² - |q|²

        # From |p|² - |q|² = diff and |p|² + |q|² = 1:
        # |p|² = (1 + diff) / 2, |q|² = (1 - diff) / 2
        p_norm = torch.sqrt(((1 + diff) / 2).clamp(min=self.eps))
        q_norm = torch.sqrt(((1 - diff) / 2).clamp(min=self.eps))

        # p = |p| * fiber  (fiber is the unit quaternion direction of p)
        p = p_norm.unsqueeze(-1) * fiber

        # From pq* = pq_star_2/2, and p = |p|*g:
        # |p|*g*q* = pq_star_2/2
        # q* = g^(-1) * pq_star_2 / (2*|p|)
        # q = conj(g^(-1) * pq_star_2 / (2*|p|))
        g_inv = quaternion_conjugate(fiber)  # fiber is unit, so inverse = conjugate
        pq_star = pq_star_2 / 2.0
        # q* = g_inv * pq_star / |p|
        q_star = quaternion_multiply(g_inv, pq_star) / p_norm.unsqueeze(-1).clamp(min=self.eps)
        q = quaternion_conjugate(q_star)

        return (p, q)
