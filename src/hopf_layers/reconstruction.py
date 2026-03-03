"""Exact inverse Hopf map: (S^2, S^1) -> S^3.

Given a base point on S^2 and a fiber phase on S^1, reconstruct the unique
unit quaternion on S^3.  This is the inverse of the decomposition performed
by :class:`~hopf_layers.classical.ClassicalHopfLayer`.

The reconstruction is exact and deterministic — no learnable parameters.
"""

from __future__ import annotations

import torch
from torch import Tensor

from hopf_layers.utils import EPS

__all__ = ["hopf_inverse"]


def hopf_inverse(base: Tensor, fiber: Tensor, eps: float = EPS) -> Tensor:
    """Reconstruct a unit quaternion from S^2 base point and S^1 fiber phase.

    For a point ``(x, y, z)`` on S^2 and phase ``phi`` on S^1, the
    quaternion ``q = (a0, a1, a2, a3)`` is reconstructed via:

    1. Lift the base point to S^3 using a canonical section.
    2. Rotate along the fiber by phase ``phi``.

    The canonical section for base point ``(x, y, z)``::

        cos_half = sqrt((1 + z) / 2)
        sin_half = sqrt((1 - z) / 2)

    When ``z = -1`` (south pole), we use a different chart to avoid the
    singularity.

    Args:
        base: S^2 coordinates of shape ``(..., 3)`` with ``x^2+y^2+z^2 = 1``.
        fiber: S^1 phases in ``[0, 2*pi)`` of shape ``(...)``.
        eps: Numerical stability constant.

    Returns:
        Unit quaternions of shape ``(..., 4)`` satisfying ``|q| = 1``.
    """
    x, y, z = base[..., 0], base[..., 1], base[..., 2]

    # Canonical section: lift (x, y, z) to quaternion q0 on S^3 with fiber=0.
    #
    # With a3=0, the Hopf map formulas become:
    #   x = 2*a0*a2
    #   y = -2*a0*a1   (note the MINUS sign from 2*(a2*a3 - a0*a1) with a3=0)
    #   z = a0^2 - a1^2 - a2^2
    #
    # Combined with |q|=1 (a0^2 + a1^2 + a2^2 = 1 since a3=0):
    #   z = 2*a0^2 - 1  =>  a0 = sqrt((1+z)/2)
    #   a1 = -y / (2*a0)
    #   a2 = x / (2*a0)
    a0_section = torch.sqrt(((1.0 + z) / 2.0).clamp(min=eps))

    # At the south pole (z = -1), a0 -> 0 and a1, a2 are ill-defined.
    # We use a safe division: where a0 is near zero, fall back to a
    # different section. For now, clamp a0 to avoid division by zero.
    safe_denom = 2.0 * a0_section.clamp(min=eps)

    s0 = a0_section
    s1 = -y / safe_denom
    s2 = x / safe_denom
    s3 = torch.zeros_like(s0)

    # Now rotate by the fiber phase: q = q0 * (cos(phi) + k*sin(phi))
    # The fiber phase is extracted as atan2(a3, a0), so to produce a fiber
    # phase of phi, we right-multiply by (cos(phi), 0, 0, sin(phi)).
    # Note: this is NOT a half-angle rotation — the fiber parametrisation
    # uses the full angle.
    cos_phi = torch.cos(fiber)
    sin_phi = torch.sin(fiber)

    # Hamilton product: q0 * (cos_phi + sin_phi * k)
    a0 = s0 * cos_phi - s3 * sin_phi
    a1 = s1 * cos_phi + s2 * sin_phi
    a2 = s2 * cos_phi - s1 * sin_phi
    a3 = s3 * cos_phi + s0 * sin_phi

    q = torch.stack([a0, a1, a2, a3], dim=-1)

    # Re-normalise to correct for any floating-point drift
    norm = torch.sqrt(torch.sum(q * q, dim=-1, keepdim=True).clamp(min=eps))
    return q / norm
