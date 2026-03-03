"""Lattice gauge theory example: decompose SU(2) gauge links via Hopf fibration.

Generates a 2D SU(2) pure gauge configuration using Metropolis-Hastings,
then decomposes it with ClassicalHopfLayer to extract base (S^2), fiber (S^1),
and transition signals.

The Wilson action for a single plaquette is:
    S_plaq = -beta * (1/2) * Re Tr(U_plaq)
where U_plaq = U_mu(x) U_nu(x+mu) U_mu(x+nu)^dag U_nu(x)^dag.

For SU(2), (1/2) Re Tr(U) = a0 when U = a0*I + i*a_k*sigma_k, so the
plaquette action simplifies to S_plaq = -beta * a0(U_plaq).
"""

import torch
import numpy as np
from hopf_layers import ClassicalHopfLayer
from hopf_layers.quaternion import quaternion_multiply, quaternion_conjugate


def random_su2_near_identity(shape, epsilon=0.3):
    """Generate random SU(2) elements near the identity.

    Args:
        shape: Shape prefix; output is (*shape, 4).
        epsilon: Spread of the random perturbation.

    Returns:
        Unit quaternions of shape (*shape, 4).
    """
    noise = torch.randn(*shape, 4) * epsilon
    noise[..., 0] += 1.0  # bias toward identity
    return noise / noise.norm(dim=-1, keepdim=True)


def plaquette_product(links, x, y, L):
    """Compute the plaquette quaternion at site (x, y).

    Plaquette = U_x(x,y) * U_y(x+1,y) * U_x(x,y+1)^dag * U_y(x,y)^dag

    Args:
        links: Gauge links of shape (L, L, 2, 4). Dim 2: 0=x-dir, 1=y-dir.
        x, y: Lattice coordinates.
        L: Lattice size (periodic boundary).

    Returns:
        Quaternion of shape (4,) representing the plaquette.
    """
    xp = (x + 1) % L
    yp = (y + 1) % L
    u1 = links[x, y, 0]       # U_x(x, y)
    u2 = links[xp, y, 1]      # U_y(x+1, y)
    u3 = quaternion_conjugate(links[x, yp, 0])   # U_x(x, y+1)^dag
    u4 = quaternion_conjugate(links[x, y, 1])     # U_y(x, y)^dag

    prod = quaternion_multiply(u1.unsqueeze(0), u2.unsqueeze(0)).squeeze(0)
    prod = quaternion_multiply(prod.unsqueeze(0), u3.unsqueeze(0)).squeeze(0)
    prod = quaternion_multiply(prod.unsqueeze(0), u4.unsqueeze(0)).squeeze(0)
    return prod


def compute_staple(links, x, y, mu, L):
    """Compute the staple sum for link U_mu at site (x, y).

    The staple is the part of the plaquette product that does NOT include
    the link being updated. For 2D there is one plaquette per link direction.

    For direction mu=0 (x-link) at site (x,y):
        staple = U_1(x+1,y) * U_0(x,y+1)^dag * U_1(x,y)^dag

    For direction mu=1 (y-link) at site (x,y):
        staple = U_0(x,y+1) * U_1(x+1,y)^dag * U_0(x,y)^dag

    Returns:
        Quaternion of shape (4,).
    """
    xp = (x + 1) % L
    yp = (y + 1) % L
    nu = 1 - mu  # the other direction

    if mu == 0:
        # Forward staple for x-link
        s1 = links[xp, y, 1]                           # U_y(x+1, y)
        s2 = quaternion_conjugate(links[x, yp, 0])     # U_x(x, y+1)^dag
        s3 = quaternion_conjugate(links[x, y, 1])       # U_y(x, y)^dag
    else:
        # Forward staple for y-link
        s1 = links[x, yp, 0]                           # U_x(x, y+1)
        s2 = quaternion_conjugate(links[xp, y, 1])     # U_y(x+1, y)^dag
        s3 = quaternion_conjugate(links[x, y, 0])       # U_x(x, y)^dag

    prod = quaternion_multiply(s1.unsqueeze(0), s2.unsqueeze(0)).squeeze(0)
    prod = quaternion_multiply(prod.unsqueeze(0), s3.unsqueeze(0)).squeeze(0)
    return prod


def metropolis_sweep(links, beta, L):
    """Perform one Metropolis sweep over all links.

    For each link, proposes a new value by left-multiplying with a random
    SU(2) element near identity. Accepts/rejects based on the change in
    Wilson action: delta_S = -beta * (a0_new - a0_old) where a0 is the
    scalar part of (link * staple).
    """
    accepted = 0
    total = 0

    for x in range(L):
        for y in range(L):
            for mu in range(2):
                old_link = links[x, y, mu].clone()
                staple = compute_staple(links, x, y, mu, L)

                # Old action contribution: a0 of (old_link * staple)
                old_plaq = quaternion_multiply(
                    old_link.unsqueeze(0), staple.unsqueeze(0)
                ).squeeze(0)
                old_a0 = old_plaq[0]

                # Propose new link
                noise = random_su2_near_identity((), epsilon=0.3)
                new_link = quaternion_multiply(
                    noise.unsqueeze(0), old_link.unsqueeze(0)
                ).squeeze(0)
                new_link = new_link / new_link.norm()

                # New action contribution
                new_plaq = quaternion_multiply(
                    new_link.unsqueeze(0), staple.unsqueeze(0)
                ).squeeze(0)
                new_a0 = new_plaq[0]

                # delta_S = -beta * (new_a0 - old_a0)
                delta_S = -beta * (new_a0 - old_a0)

                if delta_S < 0 or torch.rand(1).item() < np.exp(
                    -delta_S.item()
                ):
                    links[x, y, mu] = new_link
                    accepted += 1
                total += 1

    return links, accepted / total


def analyze_gauge_config(links, layer, L):
    """Decompose gauge config via Hopf fibration and print statistics.

    Args:
        links: Gauge links of shape (L, L, 2, 4).
        layer: ClassicalHopfLayer instance.
        L: Lattice size.
    """
    # Reshape to ClassicalHopfLayer's expected site-field format:
    # (batch=2, 4, L, L) -- treat two directions as batch dim
    q = links.permute(2, 3, 0, 1)  # (2, 4, L, L)
    out = layer(q)

    base = out.base       # (2, 3, L, L)
    fiber = out.fiber      # (2, L, L)
    tx = out.transitions_x  # (2, L, L)
    ty = out.transitions_y  # (2, L, L)

    base_norm_sq = (base ** 2).sum(dim=1).mean()
    print(f"  Base S^2 norm (should be 1): {base_norm_sq:.6f}")
    print(f"  Mean fiber phase: {fiber.mean():.4f} rad")
    print(f"  Transition signals |tx|: mean={tx.abs().mean():.4f}")
    print(f"  Transition signals |ty|: mean={ty.abs().mean():.4f}")

    n_trans = ((tx.abs() > 0.5).sum() + (ty.abs() > 0.5).sum()).item()
    print(f"  Significant transitions (|t|>0.5): {n_trans}")

    # Average plaquette (order parameter)
    plaq_sum = 0.0
    for x in range(L):
        for y in range(L):
            plaq = plaquette_product(links, x, y, L)
            plaq_sum += plaq[0].item()
    avg_plaq = plaq_sum / (L * L)
    print(f"  Average plaquette a0: {avg_plaq:.4f} (1=ordered, 0=disordered)")


def main():
    print("Lattice Gauge Theory + Hopf Fibration Demo")
    print("=" * 60)

    L = 6
    n_therm = 100
    layer = ClassicalHopfLayer()

    for beta in [0.5, 4.0]:
        print(f"\nbeta = {beta} (coupling strength)")
        print("-" * 40)

        # Hot start: random SU(2) links (fully disordered)
        links = torch.randn(L, L, 2, 4)
        links = links / links.norm(dim=-1, keepdim=True)

        print(f"  Running {n_therm} Metropolis sweeps (hot start)...")
        for sweep in range(n_therm):
            links, acc_rate = metropolis_sweep(links, beta, L)
        print(f"  Final acceptance rate: {acc_rate:.2%}")

        print("  Hopf decomposition:")
        analyze_gauge_config(links, layer, L)

    print("\n" + "=" * 60)
    print("Key insight: higher beta -> smoother config -> fewer transitions")


if __name__ == "__main__":
    main()
