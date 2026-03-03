"""Shared utilities for hopf-layers.

Includes the clipped-gradient atan2 used for numerically stable fiber extraction.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = ["clipped_atan2", "EPS", "PI", "TWO_PI"]

EPS: float = 1e-8
PI: float = math.pi
TWO_PI: float = 2.0 * math.pi


class _ClippedAtan2Grad(torch.autograd.Function):
    """atan2 with gradient clipped to [-max_grad, max_grad].

    Forward pass: exact ``atan2(y, x)``.
    Backward pass: standard atan2 gradient with magnitude clamped to ``max_grad``
    near the singularity where ``x = y = 0``.  This acts as a straight-through
    estimator (STE) that stabilises training without altering the forward value.
    """

    @staticmethod
    def forward(ctx, y: Tensor, x: Tensor, max_grad: float) -> Tensor:  # noqa: N805
        ctx.save_for_backward(y, x)
        ctx.max_grad = max_grad
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # noqa: N805
        y, x = ctx.saved_tensors
        max_grad = ctx.max_grad

        denom = (x * x + y * y).clamp(min=EPS)
        # d(atan2)/dy =  x / (x^2 + y^2)
        # d(atan2)/dx = -y / (x^2 + y^2)
        grad_y = (x / denom).clamp(-max_grad, max_grad) * grad_output
        grad_x = (-y / denom).clamp(-max_grad, max_grad) * grad_output

        return grad_y, grad_x, None  # None for max_grad


def clipped_atan2(y: Tensor, x: Tensor, max_grad: float = 100.0) -> Tensor:
    """Compute ``atan2(y, x)`` with gradient magnitude clipped near the singularity.

    This is the default fiber-phase extractor for :class:`ClassicalHopfLayer`.
    The forward value is *exact*; only the backward pass is modified.

    Args:
        y: Numerator tensor.
        x: Denominator tensor.
        max_grad: Maximum absolute gradient value.  Gradients with magnitude
            exceeding this are clamped.  Default ``100.0`` is generous enough
            for typical lattice data.

    Returns:
        ``atan2(y, x)`` with shape matching the broadcast of *y* and *x*.
    """
    return _ClippedAtan2Grad.apply(y, x, max_grad)
