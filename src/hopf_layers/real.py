"""Real Hopf fibration: S^0 -> S^1 -> S^1.

The real Hopf map decomposes a unit circle element (a point on S^1)
into a base point on S^1 (real projective line RP^1, which is also S^1)
and a fiber in S^0 = {+1, -1} (a discrete sign).

Concretely, for a unit complex number z = (x, y) with x^2 + y^2 = 1:
- Base:  angle  theta = 2 * atan2(y, x)   mod 2*pi
         (double covering: z and -z map to the same base point)
- Fiber: sign   s = sign(x)  (or +1 when x > 0, -1 when x < 0)

This is the simplest Hopf fibration and serves as a 1D analogue of the
classical S^3 -> S^2 map.  It can be used for:
- Decomposing 2D unit vector fields into angle + orientation
- 1D circle-valued data analysis
- Educational intro before the full quaternionic Hopf layer

The real Hopf fibration is the first in the family:
  S^0 -> S^1 -> S^1  (real, this module)
  S^1 -> S^3 -> S^2  (classical, hopf_layers.classical)
  S^3 -> S^7 -> S^4  (quaternionic, hopf_layers.quaternionic -- future)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn as nn

from hopf_layers.utils import EPS

__all__ = ["RealHopfLayer", "RealHopfOutput"]


class RealHopfOutput:
    """Output of the real Hopf fibration decomposition.

    Attributes:
        base:  Base angle on S^1,  shape ``(...)``,  range ``[0, 2*pi)``.
               The double-covering angle: z and -z give the same base.
        fiber: Discrete sign in {-1, +1},  shape ``(...)``.
               The S^0 fiber over each base point.
        input_angle: Original angle atan2(y, x), shape ``(...)``, range ``[0, 2*pi)``.
    """

    __slots__ = ("base", "fiber", "input_angle")

    def __init__(self, base: Tensor, fiber: Tensor, input_angle: Tensor):
        self.base = base
        self.fiber = fiber
        self.input_angle = input_angle


class RealHopfLayer(nn.Module):
    """Real Hopf fibration layer: S^1 -> (S^1, S^0).

    Decomposes 2D unit vectors (points on S^1) into:
    - **base**: angle on the quotient circle (double cover),  range [0, 2*pi)
    - **fiber**: discrete sign {+1, -1}

    The map: for input (x, y) with x^2 + y^2 = 1,
        alpha = atan2(y, x)      ∈ [0, 2π)   (original angle)
        base  = 2*alpha  mod 2π               (double-covering angle)
        fiber = sign(x)           ∈ {-1, +1}  (hemisphere)

    Reconstruction:
        alpha = base / 2                       if fiber = +1
        alpha = base / 2 + π                   if fiber = -1
        (x, y) = (cos(alpha), sin(alpha))

    Args:
        eps: Small constant for numerical stability.
    """

    def __init__(self, eps: float = EPS):
        super().__init__()
        self.eps = eps

    def forward(self, vectors: Tensor) -> RealHopfOutput:
        """Decompose 2D unit vectors via the real Hopf map.

        Args:
            vectors: Input of shape ``(..., 2)`` with unit norm along last dim.

        Returns:
            RealHopfOutput with base, fiber, and input_angle.
        """
        if vectors.shape[-1] != 2:
            raise ValueError(
                f"Expected last dimension = 2, got shape {vectors.shape}"
            )

        x = vectors[..., 0]
        y = vectors[..., 1]

        # Normalize for safety
        norm = torch.sqrt((x * x + y * y).clamp(min=self.eps))
        x = x / norm
        y = y / norm

        # Original angle in [0, 2*pi)
        alpha = torch.atan2(y, x)
        alpha = torch.remainder(alpha, 2 * math.pi)

        # Base: double-covering angle
        base = torch.remainder(2.0 * alpha, 2 * math.pi)

        # Fiber: sign of x (which hemisphere)
        # Use soft sign for gradient flow, but snap to {-1, +1} in forward
        fiber = torch.sign(x)
        # Handle exact zero: default to +1
        fiber = torch.where(fiber == 0, torch.ones_like(fiber), fiber)

        return RealHopfOutput(base=base, fiber=fiber, input_angle=alpha)

    def inverse(self, base: Tensor, fiber: Tensor) -> Tensor:
        """Reconstruct unit vectors from base angle and fiber sign.

        Args:
            base: Base angle in [0, 2*pi), shape ``(...)``.
            fiber: Sign in {-1, +1}, shape ``(...)``.

        Returns:
            Unit vectors of shape ``(..., 2)``.
        """
        # Two candidate angles from the double covering
        alpha1 = base / 2.0
        alpha2 = alpha1 + math.pi

        # Choose the candidate whose cos matches the fiber sign
        # fiber = sign(cos(alpha)), so pick alpha where fiber * cos(alpha) > 0
        use_alpha2 = (fiber * torch.cos(alpha1)) < 0
        alpha = torch.where(use_alpha2, alpha2, alpha1)

        x = torch.cos(alpha)
        y = torch.sin(alpha)
        return torch.stack([x, y], dim=-1)
