"""Differentiable phase-winding transition detection.

Detects topological winding events in fiber phase fields using a soft
``tanh``-based thresholding that preserves gradient flow.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from hopf_layers.utils import PI, TWO_PI

__all__ = ["TransitionDetector"]


class TransitionDetector(nn.Module):
    """Detect phase-winding transitions in a 2-D fiber-phase field.

    Given a scalar field of fiber phases on a lattice, this module computes
    soft transition signals in the *x* and *y* directions by:

    1. Computing nearest-neighbour phase differences.
    2. Unwrapping to ``[-pi, pi)`` via modular arithmetic.
    3. Extracting the *jump* (amount removed by unwrapping).
    4. Applying ``tanh(jump / temperature)`` for a differentiable ±1 signal.

    The temperature controls sharpness: lower values produce crisper signals
    but steeper gradients.

    Args:
        temperature: Softness of detection.  For lattice spacing *a*, a
            reasonable default is ``a / 2``.  Default ``0.5``.
    """

    def __init__(self, temperature: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, fiber: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute transition signals from a fiber-phase field.

        Args:
            fiber: Phase field of shape ``(batch, Lx, Ly)`` with values
                in ``[0, 2*pi)``.

        Returns:
            A tuple ``(transitions_x, transitions_y)`` each of shape
            ``(batch, Lx, Ly)`` with values in ``[-1, 1]``.
        """
        # Nearest-neighbour differences (periodic boundary via roll)
        delta_x = torch.roll(fiber, -1, dims=-2) - fiber
        delta_y = torch.roll(fiber, -1, dims=-1) - fiber

        # Unwrap to [-pi, pi)
        delta_x_unwrapped = torch.remainder(delta_x + PI, TWO_PI) - PI
        delta_y_unwrapped = torch.remainder(delta_y + PI, TWO_PI) - PI

        # Jump = how much the unwrapping removed
        jump_x = delta_x - delta_x_unwrapped
        jump_y = delta_y - delta_y_unwrapped

        # Soft sign
        transitions_x = torch.tanh(jump_x / self.temperature)
        transitions_y = torch.tanh(jump_y / self.temperature)

        return transitions_x, transitions_y

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"
