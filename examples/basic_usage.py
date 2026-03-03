"""Basic usage of hopf-layers: all three Hopf fibrations."""

import torch
from hopf_layers import ClassicalHopfLayer, RealHopfLayer, QuaternionicHopfLayer
from hopf_layers.reconstruction import hopf_inverse


def demo_classical():
    """Classical Hopf fibration: S^1 -> S^3 -> S^2"""
    print("=" * 60)
    print("Classical Hopf Fibration: S^1 -> S^3 -> S^2")
    print("=" * 60)

    layer = ClassicalHopfLayer()

    # Site field: (batch, 4, Lx, Ly)
    q = torch.randn(4, 4, 8, 8)
    out = layer(q)

    print(f"Input:         {q.shape}")
    print(f"Base (S^2):    {out.base.shape}")
    print(f"Fiber (S^1):   {out.fiber.shape}")
    print(f"Transitions X: {out.transitions_x.shape}")
    print(f"Transitions Y: {out.transitions_y.shape}")

    # Verify S^2 constraint
    base_norm = (out.base ** 2).sum(dim=1)
    print(f"Base on S^2: max |norm-1| = {(base_norm - 1).abs().max():.2e}")

    # Reconstruction: base must be (..., 3), so permute from (B, 3, Lx, Ly)
    q_rec = hopf_inverse(
        out.base.permute(0, 2, 3, 1),  # to (B, Lx, Ly, 3)
        out.fiber,                       # (B, Lx, Ly)
    )
    print(f"Reconstructed: {q_rec.shape}")
    print()


def demo_real():
    """Real Hopf fibration: S^0 -> S^1 -> S^1"""
    print("=" * 60)
    print("Real Hopf Fibration: S^0 -> S^1 -> S^1")
    print("=" * 60)

    layer = RealHopfLayer()

    # Unit vectors on S^1
    z = torch.randn(16, 2)
    z = z / z.norm(dim=-1, keepdim=True)
    out = layer(z)

    print(f"Input:       {z.shape}")
    print(f"Base angle:  {out.base.shape} "
          f"(range [{out.base.min():.2f}, {out.base.max():.2f}])")
    print(f"Fiber sign:  {out.fiber.shape} "
          f"(values: {out.fiber.unique().tolist()})")

    # Reconstruction
    z_rec = layer.inverse(out.base, out.fiber)
    error = (z - z_rec).norm(dim=-1).max()
    print(f"Round-trip error: {error:.2e}")
    print()


def demo_quaternionic():
    """Quaternionic Hopf fibration: S^3 -> S^7 -> S^4"""
    print("=" * 60)
    print("Quaternionic Hopf Fibration: S^3 -> S^7 -> S^4")
    print("=" * 60)

    layer = QuaternionicHopfLayer()

    # Quaternion pair (= octonion)
    p = torch.randn(16, 4)
    q = torch.randn(16, 4)
    out = layer(p, q)

    print(f"Input p:      {p.shape}")
    print(f"Input q:      {q.shape}")
    print(f"Base (S^4):   {out.base.shape}")
    print(f"Fiber (S^3):  {out.fiber.shape}")

    # Verify S^4 constraint
    base_norm = (out.base ** 2).sum(dim=-1)
    print(f"Base on S^4: max |norm-1| = {(base_norm - 1).abs().max():.2e}")

    # Verify S^3 fiber
    fiber_norm = (out.fiber ** 2).sum(dim=-1)
    print(f"Fiber on S^3: max |norm-1| = {(fiber_norm - 1).abs().max():.2e}")

    # Round-trip
    p_rec, q_rec = layer.inverse(out.base, out.fiber)
    # Normalize originals for comparison (layer normalises internally)
    total_norm = torch.sqrt(
        (p ** 2).sum(-1, keepdim=True) + (q ** 2).sum(-1, keepdim=True)
    )
    p_n, q_n = p / total_norm, q / total_norm
    error = max(
        (p_rec - p_n).abs().max().item(),
        (q_rec - q_n).abs().max().item(),
    )
    print(f"Round-trip error: {error:.2e}")
    print()


if __name__ == "__main__":
    demo_classical()
    demo_real()
    demo_quaternionic()
    print("All demos completed successfully!")
