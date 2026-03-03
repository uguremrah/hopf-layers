"""Device selection utilities for hopf-layers."""

from __future__ import annotations

import torch


def get_device(override: str | None = None) -> torch.device:
    """Return the best available device.

    Args:
        override: Force a specific device string (e.g. ``"cpu"``, ``"cuda:0"``).
            If ``None`` or ``"auto"``, selects CUDA when available, else CPU.

    Returns:
        A :class:`torch.device` instance.
    """
    if override is not None and override != "auto":
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
