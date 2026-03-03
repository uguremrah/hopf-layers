"""GPU compatibility tests for hopf-layers.

All tests are skipped when CUDA is not available.
When CUDA is present, verifies CPU/CUDA parity for all three Hopf layers,
gradient flow on GPU, and the get_device() utility.
"""

import pytest
import torch

from hopf_layers import (
    ClassicalHopfLayer,
    QuaternionicHopfLayer,
    RealHopfLayer,
    get_device,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ── get_device utility ──────────────────────────────────────────────


class TestGetDevice:
    def test_cpu_override(self):
        assert get_device("cpu") == torch.device("cpu")

    def test_auto_string(self):
        d = get_device("auto")
        expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert d == expected

    def test_none_default(self):
        d = get_device()
        expected = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert d == expected

    @requires_cuda
    def test_cuda_override(self):
        assert get_device("cuda") == torch.device("cuda")

    @requires_cuda
    def test_cuda_index_override(self):
        assert get_device("cuda:0") == torch.device("cuda:0")


# ── Classical HopfLayer CPU/CUDA parity ─────────────────────────────


@requires_cuda
class TestClassicalGPU:
    def test_forward_parity(self):
        torch.manual_seed(42)
        x = torch.randn(4, 4, 8, 8)
        layer = ClassicalHopfLayer()

        out_cpu = layer(x)
        out_gpu = layer.cuda()(x.cuda())

        assert out_gpu.base.device.type == "cuda"
        torch.testing.assert_close(out_cpu.base, out_gpu.base.cpu(), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_cpu.fiber, out_gpu.fiber.cpu(), atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self):
        x = torch.randn(2, 4, 8, 8, device="cuda", requires_grad=True)
        layer = ClassicalHopfLayer().cuda()
        out = layer(x)
        loss = out.base.sum() + out.fiber.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.device.type == "cuda"
        assert x.grad.abs().sum() > 0

    def test_output_device_matches_input(self):
        x = torch.randn(2, 4, 8, 8, device="cuda")
        layer = ClassicalHopfLayer().cuda()
        out = layer(x)
        assert out.base.device.type == "cuda"
        assert out.fiber.device.type == "cuda"
        assert out.transitions_x.device.type == "cuda"
        assert out.transitions_y.device.type == "cuda"


# ── Real HopfLayer CPU/CUDA parity ──────────────────────────────────


@requires_cuda
class TestRealGPU:
    def test_forward_parity(self):
        torch.manual_seed(42)
        x = torch.randn(4, 2, 16)
        layer = RealHopfLayer()

        out_cpu = layer(x)
        out_gpu = layer.cuda()(x.cuda())

        assert out_gpu.base.device.type == "cuda"
        torch.testing.assert_close(out_cpu.base, out_gpu.base.cpu(), atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self):
        x = torch.randn(4, 2, 16, device="cuda", requires_grad=True)
        layer = RealHopfLayer().cuda()
        out = layer(x)
        loss = out.base.sum() + out.fiber.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# ── Quaternionic HopfLayer CPU/CUDA parity ───────────────────────────


@requires_cuda
class TestQuaternionicGPU:
    def test_forward_parity(self):
        torch.manual_seed(42)
        x = torch.randn(4, 8, 8, 8)
        layer = QuaternionicHopfLayer()

        out_cpu = layer(x)
        out_gpu = layer.cuda()(x.cuda())

        assert out_gpu.base.device.type == "cuda"
        torch.testing.assert_close(out_cpu.base, out_gpu.base.cpu(), atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self):
        x = torch.randn(2, 8, 8, 8, device="cuda", requires_grad=True)
        layer = QuaternionicHopfLayer().cuda()
        out = layer(x)
        loss = out.base.sum() + out.fiber.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
