"""Classical Hopf fibration layer: S^1 -> S^3 -> S^2.

The Hopf fibration is a fiber bundle with total space S^3, base space S^2,
and fiber S^1.  Since SU(2) is diffeomorphic to S^3, this layer provides a
natural decomposition of quaternion-valued (gauge) fields into:

- **Base** (S^2): gauge-invariant directional content.
- **Fiber** (S^1): local gauge phase / rotation angle.
- **Transitions**: differentiable winding signals detecting topological defects.

The layer has **no learnable parameters** — it is a fixed geometric feature
extractor analogous to FFT or wavelet transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from hopf_layers.quaternion import quaternion_normalize
from hopf_layers.transitions import TransitionDetector
from hopf_layers.utils import TWO_PI, clipped_atan2

__all__ = ["ClassicalHopfLayer", "HopfOutput"]


@dataclass
class HopfOutput:
    """Structured output of :class:`ClassicalHopfLayer`.

    Attributes:
        base: S^2 coordinates ``(x, y, z)`` of shape ``(batch, 3, ...)``.
        fiber: S^1 phases in ``[0, 2*pi)`` of shape ``(batch, ...)``.
        transitions_x: Soft winding signal in the *x* direction.
        transitions_y: Soft winding signal in the *y* direction.
        quaternions: Normalised input quaternions (on S^3).
    """

    base: Tensor
    fiber: Tensor
    transitions_x: Tensor
    transitions_y: Tensor
    quaternions: Tensor


class ClassicalHopfLayer(nn.Module):
    """Differentiable Hopf fibration: S^3 -> S^2 with S^1 fiber extraction.

    This layer:

    1. Normalises quaternion inputs to S^3.
    2. Projects to S^2 base space via the Hopf map.
    3. Extracts the S^1 fiber phase via ``atan2`` with clipped gradients.
    4. Detects phase-winding transitions (soft, differentiable).

    Supports two input layouts:

    - **Site fields**: ``(batch, 4, Lx, Ly)`` — one quaternion per site.
    - **Link fields**: ``(batch, 4, 2, Lx, Ly)`` — one quaternion per link
      direction (2 directions in 2-D).

    Args:
        transition_temperature: Softness of winding detection.  For lattice
            spacing *a*, use ``a / 2``.  Default ``0.5``.
        atan2_max_grad: Maximum gradient magnitude for the clipped-atan2
            straight-through estimator.  Default ``100.0``.
        eps: Numerical stability constant for normalisation.  Default ``1e-8``.

    Example::

        layer = ClassicalHopfLayer()
        q = torch.randn(8, 4, 16, 16)
        out = layer(q)
        print(out.base.shape)   # (8, 3, 16, 16)
        print(out.fiber.shape)  # (8, 16, 16)
    """

    def __init__(
        self,
        transition_temperature: float = 0.5,
        atan2_max_grad: float = 100.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.transition_detector = TransitionDetector(temperature=transition_temperature)
        self.atan2_max_grad = atan2_max_grad
        self.eps = eps

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def hopf_map(self, q: Tensor) -> Tensor:
        """Compute the Hopf map S^3 -> S^2.

        For unit quaternion ``q = (a0, a1, a2, a3)``::

            x = 2 * (a1*a3 + a0*a2)
            y = 2 * (a2*a3 - a0*a1)
            z = a0^2 + a3^2 - a1^2 - a2^2

        Args:
            q: Unit quaternions of shape ``(..., 4)``.

        Returns:
            Points on S^2 of shape ``(..., 3)``.
        """
        a0, a1, a2, a3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

        x = 2.0 * (a1 * a3 + a0 * a2)
        y = 2.0 * (a2 * a3 - a0 * a1)
        z = a0 * a0 + a3 * a3 - a1 * a1 - a2 * a2

        return torch.stack([x, y, z], dim=-1)

    def extract_fiber(self, q: Tensor) -> Tensor:
        """Extract the S^1 fiber phase from a unit quaternion.

        ``phi = atan2(a3, a0)`` shifted to ``[0, 2*pi)``.

        Uses the clipped-gradient atan2 to avoid gradient spikes near the
        singularity at ``a0 = a3 = 0``.

        Args:
            q: Unit quaternions of shape ``(..., 4)``.

        Returns:
            Fiber phases in ``[0, 2*pi)`` of shape ``(...)``.
        """
        a0, a3 = q[..., 0], q[..., 3]
        phi = clipped_atan2(a3, a0, max_grad=self.atan2_max_grad)
        return torch.remainder(phi, TWO_PI)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, quaternions: Tensor) -> HopfOutput:
        """Apply the Hopf fibration decomposition.

        Args:
            quaternions: Input quaternion field.

                - ``(batch, 4, Lx, Ly)`` for site fields, or
                - ``(batch, 4, 2, Lx, Ly)`` for link fields.

        Returns:
            A :class:`HopfOutput` with *base*, *fiber*, *transitions_x*,
            *transitions_y*, and *quaternions*.
        """
        ndim = quaternions.ndim

        if ndim == 4:
            return self._forward_site(quaternions)
        elif ndim == 5:
            return self._forward_link(quaternions)
        else:
            raise ValueError(
                f"Expected 4-D (site) or 5-D (link) input, got {ndim}-D "
                f"with shape {quaternions.shape}"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _forward_site(self, quaternions: Tensor) -> HopfOutput:
        """Handle ``(batch, 4, Lx, Ly)`` site fields."""
        # (batch, 4, Lx, Ly) -> (batch, Lx, Ly, 4)
        q = quaternions.permute(0, 2, 3, 1)
        q = quaternion_normalize(q, eps=self.eps)

        base = self.hopf_map(q)              # (batch, Lx, Ly, 3)
        fiber = self.extract_fiber(q)         # (batch, Lx, Ly)
        tx, ty = self.transition_detector(fiber)

        # Permute base back to channel-first: (batch, 3, Lx, Ly)
        base = base.permute(0, 3, 1, 2)

        return HopfOutput(
            base=base,
            fiber=fiber,
            transitions_x=tx,
            transitions_y=ty,
            quaternions=q,
        )

    def _forward_link(self, quaternions: Tensor) -> HopfOutput:
        """Handle ``(batch, 4, 2, Lx, Ly)`` link fields."""
        # (batch, 4, 2, Lx, Ly) -> (batch, 2, Lx, Ly, 4)
        q = quaternions.permute(0, 2, 3, 4, 1)
        q = quaternion_normalize(q, eps=self.eps)

        base = self.hopf_map(q)       # (batch, 2, Lx, Ly, 3)
        fiber = self.extract_fiber(q)  # (batch, 2, Lx, Ly)

        # Detect transitions per link direction, then stack
        tx0, ty0 = self.transition_detector(fiber[:, 0])
        tx1, ty1 = self.transition_detector(fiber[:, 1])
        tx = torch.stack([tx0, tx1], dim=1)
        ty = torch.stack([ty0, ty1], dim=1)

        # Permute base: (batch, 3, 2, Lx, Ly)
        base = base.permute(0, 4, 1, 2, 3)

        return HopfOutput(
            base=base,
            fiber=fiber,
            transitions_x=tx,
            transitions_y=ty,
            quaternions=q,
        )

    def extra_repr(self) -> str:
        return (
            f"temperature={self.transition_detector.temperature}, "
            f"atan2_max_grad={self.atan2_max_grad}, "
            f"eps={self.eps}"
        )
