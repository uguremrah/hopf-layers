"""Exact analytical results for 2D SU(2) lattice gauge theory.

In 2D, pure gauge theory is exactly solvable because each plaquette
variable is independent (after gauge fixing).  The single-plaquette
partition function for SU(2) with Wilson action
    S = beta * sum(1 - (1/2) Re Tr U_P)
is:
    Z(beta) = (2/pi) int_0^pi sin^2(alpha) exp(beta cos(alpha)) dalpha
            = 2 I_1(beta) / beta

where I_n is the modified Bessel function of the first kind.

The exact average plaquette is:
    <(1/2) Re Tr U_P> = d/d(beta) ln Z(beta)
                      = I_0(beta)/I_1(beta) - 2/beta

NOTE: The formula I_1(beta)/I_0(beta) sometimes seen in the literature
is for U(1) gauge theory, NOT SU(2).
"""

from __future__ import annotations

import numpy as np
from scipy.special import iv as bessel_iv


def su2_plaquette_exact(beta: float) -> float:
    """Exact expectation value of (1/2) Re Tr U_P in 2D SU(2).

    Uses the Migdal-Rusakov exact solution:
        <(1/2) Tr U_P> = I_0(beta)/I_1(beta) - 2/beta

    Args:
        beta: Inverse coupling constant (must be > 0).

    Returns:
        The exact average plaquette value.

    Examples:
        >>> su2_plaquette_exact(2.0)  # doctest: +ELLIPSIS
        0.433...
        >>> su2_plaquette_exact(4.0)  # doctest: +ELLIPSIS
        0.710...
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    i0 = bessel_iv(0, beta)
    i1 = bessel_iv(1, beta)
    return float(i0 / i1 - 2.0 / beta)


def su2_plaquette_exact_array(betas: np.ndarray) -> np.ndarray:
    """Vectorized version for an array of beta values."""
    betas = np.asarray(betas, dtype=np.float64)
    i0 = bessel_iv(0, betas)
    i1 = bessel_iv(1, betas)
    return i0 / i1 - 2.0 / betas


def su2_internal_energy(beta: float) -> float:
    """Exact internal energy density e = 1 - <(1/2) Tr U_P>."""
    return 1.0 - su2_plaquette_exact(beta)


def su2_specific_heat(beta: float, dbeta: float = 1e-5) -> float:
    """Specific heat C = -beta^2 d<e>/d(beta) via numerical derivative."""
    e_plus = su2_internal_energy(beta + dbeta)
    e_minus = su2_internal_energy(beta - dbeta)
    de_dbeta = (e_plus - e_minus) / (2 * dbeta)
    return -beta**2 * de_dbeta
