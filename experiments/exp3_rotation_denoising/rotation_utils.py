"""Rotation denoising utilities.

Generates synthetic smooth quaternion rotation fields on 2D grids,
adds noise, and provides geodesic distance metrics.
"""

import numpy as np
import torch
from torch import Tensor


def random_smooth_rotation_field(Lx: int, Ly: int, n_modes: int = 3, seed=None):
    """Generate a smooth quaternion rotation field on an Lx x Ly grid.

    Creates a smooth field by superimposing low-frequency Fourier modes
    in quaternion space, then normalizing to unit quaternions.

    Args:
        Lx, Ly: Grid dimensions.
        n_modes: Number of Fourier modes per direction (controls smoothness).
        seed: Optional random seed.

    Returns:
        field: (4, 2, Lx, Ly) quaternion link field.
               The "2 directions" store the same rotation to be compatible
               with the HopfLayer link field format.
    """
    rng = np.random.default_rng(seed)

    # Generate smooth quaternion field using Fourier modes
    q = np.zeros((4, Lx, Ly))
    for comp in range(4):
        for kx in range(1, n_modes + 1):
            for ky in range(1, n_modes + 1):
                amp = rng.normal() / (kx * ky)
                phase_x = rng.uniform(0, 2 * np.pi)
                phase_y = rng.uniform(0, 2 * np.pi)
                xs = np.arange(Lx) * 2 * np.pi / Lx
                ys = np.arange(Ly) * 2 * np.pi / Ly
                X, Y = np.meshgrid(xs, ys, indexing="ij")
                q[comp] += amp * np.sin(kx * X + phase_x) * np.sin(ky * Y + phase_y)

    # Add identity component to a0 to avoid zero quaternions
    q[0] += 1.0

    # Normalize to unit quaternions
    norm = np.sqrt((q**2).sum(axis=0, keepdims=True)).clip(min=1e-8)
    q = q / norm

    # Pack into link field format: (4, 2, Lx, Ly) -- duplicate for 2 directions
    field = np.stack([q, q], axis=1)
    return field.astype(np.float32)


def add_quaternion_noise(field, sigma, seed=None):
    """Add Gaussian noise to quaternion field and renormalize.

    Args:
        field: (4, 2, Lx, Ly) clean quaternion link field.
        sigma: Noise standard deviation.
        seed: Optional random seed.

    Returns:
        noisy: (4, 2, Lx, Ly) noisy quaternion link field (renormalized).
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, size=field.shape).astype(np.float32)
    noisy = field + noise
    # Renormalize along quaternion axis (axis 0)
    norm = np.sqrt((noisy**2).sum(axis=0, keepdims=True)).clip(min=1e-8)
    noisy = noisy / norm
    return noisy


def quaternion_to_rotation_matrix(q):
    """Convert unit quaternion to 3x3 rotation matrix.

    Args:
        q: (..., 4) quaternion tensor (a0, a1, a2, a3).

    Returns:
        R: (..., 3, 3) rotation matrix.
    """
    a0, a1, a2, a3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.stack(
        [
            1 - 2 * (a2**2 + a3**2),
            2 * (a1 * a2 - a0 * a3),
            2 * (a1 * a3 + a0 * a2),
            2 * (a1 * a2 + a0 * a3),
            1 - 2 * (a1**2 + a3**2),
            2 * (a2 * a3 - a0 * a1),
            2 * (a1 * a3 - a0 * a2),
            2 * (a2 * a3 + a0 * a1),
            1 - 2 * (a1**2 + a2**2),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)

    return R


def geodesic_distance(R1, R2):
    """Compute geodesic distance on SO(3) between rotation matrices.

    d(R1, R2) = arccos((tr(R1^T R2) - 1) / 2)

    Args:
        R1, R2: (..., 3, 3) rotation matrices.

    Returns:
        d: (...) geodesic distances in radians.
    """
    R_diff = torch.matmul(R1.transpose(-2, -1), R2)
    trace = R_diff.diagonal(dim1=-2, dim2=-1).sum(-1)
    # Clamp for numerical stability
    cos_angle = ((trace - 1) / 2).clamp(-1, 1)
    return torch.acos(cos_angle)


def generate_denoising_dataset(n_samples, Lx, Ly, sigma, n_modes=3, seed_base=42):
    """Generate a denoising dataset: (noisy_field, clean_rotation_matrices).

    Args:
        n_samples: Number of samples.
        Lx, Ly: Grid size.
        sigma: Noise level.
        n_modes: Smoothness of clean field.
        seed_base: Base random seed.

    Returns:
        noisy_fields: (N, 4, 2, Lx, Ly) noisy quaternion link fields.
        clean_rotmats: (N, Lx, Ly, 9) flattened clean rotation matrices
                       (target for regression, one per site).
    """
    noisy_fields = []
    clean_rotmats = []

    for i in range(n_samples):
        seed = seed_base + i
        clean = random_smooth_rotation_field(Lx, Ly, n_modes=n_modes, seed=seed)
        noisy = add_quaternion_noise(clean, sigma, seed=seed + 10000)

        # Clean rotation matrix target -- use direction 0 quaternions
        clean_q = torch.from_numpy(clean[:, 0]).permute(1, 2, 0)  # (Lx, Ly, 4)
        clean_R = quaternion_to_rotation_matrix(clean_q)  # (Lx, Ly, 3, 3)
        clean_R_flat = clean_R.reshape(Lx, Ly, 9)  # (Lx, Ly, 9)

        noisy_fields.append(noisy)
        clean_rotmats.append(clean_R_flat.numpy())

    return np.stack(noisy_fields), np.stack(clean_rotmats)
