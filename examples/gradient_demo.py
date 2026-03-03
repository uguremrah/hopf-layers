"""Gradient flow demonstration: HopfLayer components are differentiable."""

import torch
import torch.nn as nn
from hopf_layers import ClassicalHopfLayer, QuaternionicHopfLayer


def gradient_check_classical():
    """Show gradients flow through ClassicalHopfLayer."""
    print("Classical HopfLayer Gradient Check")
    print("=" * 50)

    layer = ClassicalHopfLayer()
    q = torch.randn(4, 4, 8, 8, requires_grad=True)
    out = layer(q)

    # Gradient through base
    loss_base = out.base.sum()
    loss_base.backward(retain_graph=True)
    print(f"d(base)/d(input):  grad exists={q.grad is not None}, "
          f"max|grad|={q.grad.abs().max():.4f}, "
          f"any NaN={torch.isnan(q.grad).any()}")

    q.grad.zero_()

    # Gradient through fiber
    loss_fiber = out.fiber.sum()
    loss_fiber.backward(retain_graph=True)
    print(f"d(fiber)/d(input): grad exists={q.grad is not None}, "
          f"max|grad|={q.grad.abs().max():.4f}, "
          f"any NaN={torch.isnan(q.grad).any()}")

    q.grad.zero_()

    # Gradient through transitions
    loss_trans = out.transitions_x.sum() + out.transitions_y.sum()
    loss_trans.backward()
    print(f"d(trans)/d(input): grad exists={q.grad is not None}, "
          f"max|grad|={q.grad.abs().max():.4f}, "
          f"any NaN={torch.isnan(q.grad).any()}")
    print()


def gradient_check_quaternionic():
    """Show gradients flow through QuaternionicHopfLayer."""
    print("Quaternionic HopfLayer Gradient Check")
    print("=" * 50)

    layer = QuaternionicHopfLayer()
    p = torch.randn(8, 4, requires_grad=True)
    q = torch.randn(8, 4, requires_grad=True)
    out = layer(p, q)

    loss = out.base.sum() + out.fiber.sum()
    loss.backward()
    print(f"d(output)/d(p): max|grad|={p.grad.abs().max():.4f}, "
          f"NaN={torch.isnan(p.grad).any()}")
    print(f"d(output)/d(q): max|grad|={q.grad.abs().max():.4f}, "
          f"NaN={torch.isnan(q.grad).any()}")
    print()


def end_to_end_training_demo():
    """Mini end-to-end training example with HopfLayer as feature extractor."""
    print("End-to-End Training Demo")
    print("=" * 50)

    class HopfClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.hopf = ClassicalHopfLayer()
            # Use S^2 base (3 channels) as input to a small conv net
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 2)

        def forward(self, q):
            out = self.hopf(q)
            x = out.base  # (B, 3, Lx, Ly)
            x = torch.relu(self.conv(x))
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = HopfClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # Synthetic data
    torch.manual_seed(42)
    X = torch.randn(32, 4, 8, 8)
    y = torch.randint(0, 2, (32,))

    print("Training for 5 epochs...")
    for epoch in range(5):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(1) == y).float().mean()
        print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.2%}")

    print("\nGradients flow correctly through HopfLayer during training!")


if __name__ == "__main__":
    gradient_check_classical()
    gradient_check_quaternionic()
    end_to_end_training_demo()
    print("\nAll gradient demos completed successfully!")
