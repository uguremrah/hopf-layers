"""Topological charge computation for 2D SU(2) gauge configurations.

The "topological charge" in 2D is related to the total vortex winding:
    Q = (1/2pi) sum_P theta_P
where theta_P = arccos(a0_P) is the angle of the plaquette quaternion,
and a0_P is the scalar component of the plaquette product.

For SU(2), the plaquette P = U_x(x,y) U_y(x+1,y) U_x^dag(x,y+1) U_y^dag(x,y)
is itself an SU(2) element with Tr(P) = 2*a0_P.
"""

from __future__ import annotations

import numpy as np


def compute_plaquette_phases(links: np.ndarray) -> np.ndarray:
    """Compute plaquette phases for all sites.

    Args:
        links: (4, 2, Lx, Ly) gauge link field in quaternion representation.

    Returns:
        phases: (Lx, Ly) array of plaquette phases = arccos(a0_P).

    The plaquette P = U_x(x,y) U_y(x+1,y) U_x^dag(x,y+1) U_y^dag(x,y) has
    trace Tr(P) = 2*a0_P. The "phase" angle is alpha = arccos(a0_P) in [0, pi].
    """
    from mc_generation.su2_metropolis import quat_multiply, quat_conjugate

    Lx, Ly = links.shape[2], links.shape[3]
    phases = np.zeros((Lx, Ly))

    for x in range(Lx):
        for y in range(Ly):
            # P = U_x(x,y) * U_y(x+1,y) * U_x^dag(x,y+1) * U_y^dag(x,y)
            ux = links[:, 0, x, y]                    # U in x-direction at (x,y)
            uy_xp = links[:, 1, (x + 1) % Lx, y]     # U in y-direction at (x+1,y)
            ux_yp = links[:, 0, x, (y + 1) % Ly]      # U in x-direction at (x,y+1)
            uy = links[:, 1, x, y]                     # U in y-direction at (x,y)

            p = quat_multiply(ux, uy_xp)
            p = quat_multiply(p, quat_conjugate(ux_yp))
            p = quat_multiply(p, quat_conjugate(uy))

            # Phase: arccos(a0) gives angle in [0, pi]
            phases[x, y] = np.arccos(np.clip(p[0], -1, 1))

    return phases


def compute_topological_charge(links: np.ndarray) -> float:
    """Compute total topological charge Q from gauge configuration.

    For 2D SU(2), the topological charge is related to the total
    plaquette winding: Q = (1/2pi) sum_P theta_P where theta_P = arccos(a0_P).

    This gives a continuous measure that correlates with vortex content.

    Args:
        links: (4, 2, Lx, Ly) gauge link field.

    Returns:
        Q: float, topological charge (continuous).
    """
    phases = compute_plaquette_phases(links)
    # Normalize by 2*pi to get charge
    Q = phases.sum() / (2 * np.pi)
    return float(Q)


def compute_charge_batch(configs_list: list[np.ndarray]) -> np.ndarray:
    """Compute topological charge for a batch of configurations.

    Args:
        configs_list: List of (4, 2, Lx, Ly) arrays.

    Returns:
        charges: (N,) array of topological charges.
    """
    return np.array([compute_topological_charge(c) for c in configs_list])
