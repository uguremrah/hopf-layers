# hopf-layers

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-96%20passing-brightgreen.svg)](https://github.com/uguremrah/hopf-layers/actions)

Differentiable fiber bundle decompositions for geometric deep learning.

**hopf-layers** implements all three Hopf fibrations as PyTorch `nn.Module` layers
with full gradient flow, enabling topologically-aware feature extraction with
zero learnable parameters.

---

## Key Features

- **All three Hopf fibrations** -- real (S^0 -> S^1 -> S^1), classical (S^1 -> S^3 -> S^2), and quaternionic (S^3 -> S^7 -> S^4)
- **Zero learnable parameters** -- pure geometric feature extraction, no training required
- **Full gradient flow** -- seamlessly integrates into end-to-end training pipelines
- **Differentiable topological transition detection** -- soft tanh thresholding for phase winding
- **Exact inverse maps** -- (S^2, S^1) -> S^3 reconstruction via `hopf_inverse`
- **Complete algebraic support** -- quaternion and octonion (Cayley-Dickson) arithmetic

---

## Installation

```bash
pip install hopf-layers
```

From source:

```bash
git clone https://github.com/uguremrah/hopf-layers
cd hopf-layers && pip install -e ".[all]"
```

**Requirements:** Python >= 3.10, PyTorch >= 2.0, NumPy >= 1.24

---

## Quick Start

All three Hopf fibrations in action:

```python
import torch
from hopf_layers import ClassicalHopfLayer, RealHopfLayer, QuaternionicHopfLayer

# --- Classical Hopf fibration: S^1 -> S^3 -> S^2 ---
layer = ClassicalHopfLayer()
q = torch.randn(8, 4, 16, 16)  # batch of quaternion fields
out = layer(q)
out.base          # (8, 3, 16, 16) -- S^2 coordinates
out.fiber         # (8, 16, 16)    -- S^1 phases
out.transitions_x # (8, 16, 16)    -- x-direction winding signals

# --- Real Hopf fibration: S^0 -> S^1 -> S^1 ---
real_layer = RealHopfLayer()
z = torch.randn(32, 2)
z = z / z.norm(dim=-1, keepdim=True)
out = real_layer(z)
out.base   # (32,) -- double-covering angle [0, 2pi)
out.fiber  # (32,) -- sign fiber in {-1, +1}

# --- Quaternionic Hopf fibration: S^3 -> S^7 -> S^4 ---
quat_layer = QuaternionicHopfLayer()
p, q = torch.randn(16, 4), torch.randn(16, 4)
out = quat_layer(p, q)
out.base   # (16, 5) -- point on S^4
out.fiber  # (16, 4) -- unit quaternion on S^3
```

---

## Architecture

Each `HopfLayer` follows a four-stage decomposition pipeline:

```
Input tensor
    |
    v
[1. Normalize] -- project onto the unit sphere (S^1, S^3, or S^7)
    |
    v
[2. Hopf map]  -- compute base-space coordinates (S^1, S^2, or S^4)
    |
    v
[3. Fiber extraction] -- recover the fiber phase (S^0, S^1, or S^3)
    |
    v
[4. Transition detection] -- detect topological winding (classical layer)
    |
    v
NamedTuple output: (base, fiber, transitions_x, transitions_y)
```

All operations are differentiable. Gradients flow through normalization, the Hopf
map, fiber extraction, and transition detection without interruption.

---

## API Reference

### Layers

| Class | Fibration | Input Shape | Output Fields |
|-------|-----------|-------------|---------------|
| `RealHopfLayer` | S^0 -> S^1 -> S^1 | `(B, 2)` | `base`, `fiber` |
| `ClassicalHopfLayer` | S^1 -> S^3 -> S^2 | `(B, 4, Lx, Ly)` | `base`, `fiber`, `transitions_x`, `transitions_y` |
| `QuaternionicHopfLayer` | S^3 -> S^7 -> S^4 | `p: (B, 4), q: (B, 4)` | `base`, `fiber` |

### Modules

| Class / Function | Description |
|-----------------|-------------|
| `TransitionDetector` | Differentiable phase-winding detection with soft tanh thresholding |
| `hopf_inverse` | Exact inverse Hopf map: (S^2, S^1) -> S^3 reconstruction |

### Algebra Utilities

| Module | Functions |
|--------|-----------|
| Quaternion | `multiply`, `conjugate`, `normalize`, `inverse`, `to_su2`, `from_su2` |
| Octonion (Cayley-Dickson) | `multiply`, `conjugate`, `norm` |

---

## Mathematical Background

### Classical Hopf Fibration (S^1 -> S^3 -> S^2)

A unit quaternion q = (a_0, a_1, a_2, a_3) on S^3 maps to S^2 via:

```
n_1 = 2(a_1 a_3 + a_0 a_2)
n_2 = 2(a_2 a_3 - a_0 a_1)
n_3 = a_0^2 + a_3^2 - a_1^2 - a_2^2
```

The fiber phase is extracted as:

```
phi = atan2(a_3, a_0)
```

### Transition Detection

Phase differences between neighboring lattice sites are unwrapped and passed
through a differentiable soft-threshold:

```
T(x) = tanh(alpha * (|Delta phi| - threshold))
```

This produces continuous transition signals in [-1, 1] that indicate topological
winding, with `alpha` controlling the sharpness of the detection.

### Real Hopf Fibration (S^0 -> S^1 -> S^1)

A unit vector z = (x, y) on S^1 maps to a double-covering angle theta = atan2(2xy, x^2 - y^2),
with fiber given by the sign of x (or y when x is near zero).

### Quaternionic Hopf Fibration (S^3 -> S^7 -> S^4)

A pair of quaternions (p, q) representing a point on S^7 maps to S^4 via:

```
base = (2 * Re(p * conj(q)),  2 * Im(p * conj(q)),  |p|^2 - |q|^2)
```

The fiber is recovered as a unit quaternion on S^3.

---

## Experiments

The package includes three experiments on SU(2) lattice gauge theory configurations,
demonstrating the utility of Hopf fibration features for physics tasks.

| Experiment | Task | Architecture | Key Result |
|-----------|------|-------------|------------|
| Phase Classification | SU(2) lattice gauge phase detection | CNN + HopfLayer | **100% accuracy** |
| Topological Charge | Continuous charge regression | CNN + HopfLayer | **R^2 = 0.932** |
| Rotation Denoising | Per-site rotation matrix recovery | Structured output | **Geodesic = 1.43** |

### Running Experiments

```bash
# Generate Monte Carlo lattice configurations
python experiments/mc_generation/generate_su2_configs.py

# Run an experiment
python experiments/exp1_phase_classification/train.py --config configs/full_hopf.yaml
```

See `experiments/README.md` for detailed instructions and configuration options.

---

## Development

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Run the test suite (96 tests)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=hopf_layers --cov-report=term-missing
```

### Project Structure

```
hopf-layers/
  src/hopf_layers/       # Core library
  tests/                 # Test suite (96 tests)
  experiments/           # Reproducible experiments
  examples/              # Usage examples
  notebooks/             # Analysis and visualization
  paper/                 # Manuscript and figures
```

---

## Citation

```bibtex
@software{surat2026hopflayers,
  author = {Surat, Ugur Emrah},
  title = {hopf-layers: Differentiable Fiber Bundle Decompositions
           for Geometric Deep Learning},
  year = {2026},
  url = {https://github.com/uguremrah/hopf-layers}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
