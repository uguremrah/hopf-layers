"""Shared fixtures for hopf-layers tests."""

import pytest
import torch


_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")


@pytest.fixture(params=_DEVICES)
def device(request):
    """Parametrized device fixture — yields cpu and cuda (when available)."""
    return torch.device(request.param)


@pytest.fixture
def random_quaternions():
    """Batch of random (unnormalised) quaternions."""
    torch.manual_seed(42)
    return torch.randn(8, 4)


@pytest.fixture
def unit_quaternions(random_quaternions):
    """Batch of unit quaternions on S^3."""
    norm = torch.sqrt(torch.sum(random_quaternions ** 2, dim=-1, keepdim=True))
    return random_quaternions / norm


@pytest.fixture
def site_field():
    """Random quaternion site field: (batch, 4, Lx, Ly)."""
    torch.manual_seed(42)
    return torch.randn(4, 4, 8, 8)


@pytest.fixture
def link_field():
    """Random quaternion link field: (batch, 4, 2, Lx, Ly)."""
    torch.manual_seed(42)
    return torch.randn(4, 4, 2, 8, 8)
