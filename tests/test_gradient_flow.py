"""End-to-end gradient flow tests through the full pipeline.

Verifies that gradients propagate correctly through:
ClassicalHopfLayer (forward) -> hopf_inverse (backward)
and through all output channels.
"""

import torch

from hopf_layers.classical import ClassicalHopfLayer
from hopf_layers.reconstruction import hopf_inverse


class TestEndToEndGradient:
    def test_full_pipeline_gradient(self):
        """Gradient flows from loss through all layer outputs to input."""
        q = torch.randn(4, 4, 8, 8, requires_grad=True)
        layer = ClassicalHopfLayer()
        out = layer(q)

        # Loss using all output channels
        loss = (
            out.base.mean()
            + out.fiber.mean()
            + out.transitions_x.mean()
            + out.transitions_y.mean()
        )
        loss.backward()

        assert q.grad is not None
        assert q.grad.shape == q.shape
        assert not torch.isnan(q.grad).any()
        assert not torch.isinf(q.grad).any()
        assert q.grad.abs().sum() > 0

    def test_decompose_reconstruct_gradient(self):
        """Gradient flows through decompose -> reconstruct pipeline."""
        q = torch.randn(4, 4, 4, 4, requires_grad=True)
        layer = ClassicalHopfLayer()
        out = layer(q)

        # Reconstruct
        base_last = out.base.permute(0, 2, 3, 1)
        q_rec = hopf_inverse(base_last, out.fiber)

        # Loss on reconstructed quaternions
        loss = q_rec.sum()
        loss.backward()

        assert q.grad is not None
        assert not torch.isnan(q.grad).any()

    def test_gradient_magnitude_reasonable(self):
        """Gradients should not explode for typical inputs."""
        torch.manual_seed(42)
        q = torch.randn(8, 4, 16, 16, requires_grad=True)
        layer = ClassicalHopfLayer()
        out = layer(q)

        loss = out.base.mean() + out.fiber.mean()
        loss.backward()

        grad_norm = q.grad.norm().item()
        # Gradient norm should be finite and reasonable
        assert grad_norm < 1000, f"Gradient norm {grad_norm} seems too large"
        assert grad_norm > 1e-10, f"Gradient norm {grad_norm} seems too small"

    def test_link_field_gradient(self):
        """Gradient flows for 5-D link field input."""
        q = torch.randn(2, 4, 2, 8, 8, requires_grad=True)
        layer = ClassicalHopfLayer()
        out = layer(q)
        loss = out.base.sum() + out.fiber.sum()
        loss.backward()

        assert q.grad is not None
        assert q.grad.shape == q.shape
        assert not torch.isnan(q.grad).any()
