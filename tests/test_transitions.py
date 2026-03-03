"""Tests for the differentiable transition detector."""

import math

import torch

from hopf_layers.transitions import TransitionDetector


class TestTransitionDetector:
    def test_uniform_field_no_transitions(self):
        """Constant phase field should give zero transitions."""
        fiber = torch.ones(2, 8, 8) * 1.5  # constant phase
        detector = TransitionDetector(temperature=0.5)
        tx, ty = detector(fiber)
        assert tx.abs().max() < 1e-6
        assert ty.abs().max() < 1e-6

    def test_linear_ramp_no_transitions(self):
        """Slowly varying phase should give near-zero transitions."""
        x = torch.linspace(0, 1, 16).unsqueeze(0).unsqueeze(-1).expand(1, 16, 16)
        detector = TransitionDetector(temperature=0.5)
        tx, ty = detector(x)
        assert tx.abs().max() < 0.1

    def test_phase_jump_detected(self):
        """A 2*pi jump should produce strong transitions."""
        fiber = torch.zeros(1, 8, 8)
        # Insert a jump in the x-direction
        fiber[0, 4:, :] = 2 * math.pi - 0.01
        detector = TransitionDetector(temperature=0.5)
        tx, ty = detector(fiber)
        # Row 3->4 should have strong transition
        assert tx[0, 3, 0].abs() > 0.5

    def test_output_range(self):
        """Transitions should be in [-1, 1] due to tanh."""
        torch.manual_seed(0)
        fiber = torch.rand(4, 16, 16) * 2 * math.pi
        detector = TransitionDetector(temperature=0.5)
        tx, ty = detector(fiber)
        assert tx.min() >= -1.0
        assert tx.max() <= 1.0
        assert ty.min() >= -1.0
        assert ty.max() <= 1.0

    def test_temperature_effect(self):
        """Lower temperature should produce sharper (closer to ±1) signals."""
        fiber = torch.zeros(1, 8, 8)
        fiber[0, 4:, :] = 2 * math.pi - 0.01

        soft = TransitionDetector(temperature=2.0)
        sharp = TransitionDetector(temperature=0.1)

        tx_soft, _ = soft(fiber)
        tx_sharp, _ = sharp(fiber)

        # Sharp detector should have values closer to ±1
        assert tx_sharp[0, 3, 0].abs() > tx_soft[0, 3, 0].abs()

    def test_gradient_flow(self):
        # Use randn (not rand * 2pi) so the tensor itself is the leaf
        fiber = torch.randn(2, 8, 8, requires_grad=True)
        detector = TransitionDetector()
        tx, ty = detector(fiber)
        (tx.sum() + ty.sum()).backward()
        assert fiber.grad is not None
        assert not torch.isnan(fiber.grad).any()
