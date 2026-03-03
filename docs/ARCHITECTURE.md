# hopf-layers Architecture

## 1. Overview

`hopf-layers` implements all three Hopf fibrations as differentiable PyTorch modules:

| Fibration | Fiber | Total Space | Base Space | Module |
|-----------|-------|-------------|------------|--------|
| Real | S^0 = {-1, +1} | S^1 | S^1 | `RealHopfLayer` |
| Classical (complex) | S^1 | S^3 | S^2 | `ClassicalHopfLayer` |
| Quaternionic | S^3 | S^7 | S^4 | `QuaternionicHopfLayer` |

Each layer decomposes structured input (points on the total space sphere) into **base space coordinates**, **fiber phases**, and (for the classical case) **transition/winding signals** -- all with full gradient flow through every operation.

**Key properties:**

- **Zero learnable parameters** -- the fibration layers are pure geometric feature extractors, analogous to FFT or wavelet transforms. All "learning" happens in downstream networks.
- **Differentiable transition detection** -- the `TransitionDetector` identifies topological defects (phase windings) using a soft `tanh`-based threshold that preserves gradient flow, replacing the non-differentiable hard `sign()` function.
- **Numerically stable** -- custom `clipped_atan2` autograd function, epsilon-clamped normalizations, and careful handling of measure-zero singularities ensure NaN-free forward and backward passes.

## 2. Package Source Structure

```
src/hopf_layers/
    __init__.py          -- Public API exports
    classical.py         -- ClassicalHopfLayer (S^1 -> S^3 -> S^2)
    real.py              -- RealHopfLayer (S^0 -> S^1 -> S^1)
    quaternionic.py      -- QuaternionicHopfLayer (S^3 -> S^7 -> S^4)
    quaternion.py        -- Quaternion algebra utilities
    transitions.py       -- TransitionDetector (phase-winding detection)
    reconstruction.py    -- hopf_inverse (exact inverse map)
    utils.py             -- clipped_atan2, constants (EPS, PI, TWO_PI)
```

## 3. Module Dependency Graph

```
utils.py
    Exports: EPS, PI, TWO_PI, clipped_atan2
    Dependencies: torch, math
    |
    +---> quaternion.py
    |        Exports: quaternion_multiply, quaternion_conjugate,
    |                 quaternion_normalize, quaternion_inverse,
    |                 quaternion_norm, quaternion_to_su2, su2_to_quaternion
    |        Dependencies: torch
    |        |
    |        +---> classical.py
    |        |        Uses: quaternion_normalize (from quaternion.py)
    |        |              clipped_atan2, TWO_PI (from utils.py)
    |        |              TransitionDetector (from transitions.py)
    |        |
    |        +---> quaternionic.py
    |                 Uses: quaternion_multiply, quaternion_conjugate,
    |                       quaternion_normalize, quaternion_norm (from quaternion.py)
    |                       EPS (from utils.py)
    |
    +---> transitions.py
    |        Uses: PI, TWO_PI (from utils.py)
    |
    +---> reconstruction.py
    |        Uses: EPS (from utils.py)
    |
    +---> real.py
             Uses: EPS (from utils.py)
```

Simplified view:

```
utils.py (constants, clipped_atan2)
    ^
    |
quaternion.py (multiply, conjugate, normalize, inverse, to/from SU(2))
    ^              ^
    |              |
classical.py    quaternionic.py
    ^
    |
transitions.py (TransitionDetector)
    ^
    |
reconstruction.py (hopf_inverse)
```

All arrows point upward toward dependencies. `real.py` depends only on `utils.py`.

## 4. Data Flow: ClassicalHopfLayer

The `ClassicalHopfLayer` is the primary module. It accepts quaternion fields in channel-first format and produces a structured `HopfOutput` containing base coordinates, fiber phases, and transition signals.

### Full Pipeline

```
Input: (B, 4, [2,] Lx, Ly) quaternion field
  |
  +-- Permute to (..., 4) last-dim layout
  |     Site: (B, 4, Lx, Ly) -> (B, Lx, Ly, 4)
  |     Link: (B, 4, 2, Lx, Ly) -> (B, 2, Lx, Ly, 4)
  |
  +-- Normalize to S^3: q = q / |q|
  |     norm = sqrt(clamp(sum(q*q, dim=-1), min=eps))
  |     q = q / norm
  |
  +-- Hopf Map: q -> (x, y, z) in S^2
  |     For unit quaternion q = (a0, a1, a2, a3):
  |       x = 2(a1*a3 + a0*a2)
  |       y = 2(a2*a3 - a0*a1)
  |       z = a0^2 + a3^2 - a1^2 - a2^2
  |     Output satisfies x^2 + y^2 + z^2 = 1
  |
  +-- Fiber Extraction: phi = atan2(a3, a0) mod 2*pi
  |     Uses clipped-gradient atan2 for stability
  |     phi in [0, 2*pi)
  |
  +-- Transition Detection (per spatial direction):
  |     1. delta = roll(phi, -1) - phi        (nearest-neighbor differences)
  |     2. unwrap = mod(delta + pi, 2*pi) - pi  (project to [-pi, pi))
  |     3. jump = delta - unwrap              (winding component)
  |     4. signal = tanh(jump / temperature)  (soft thresholding)
  |     Applied independently in x and y directions
  |
  +-- Permute base back to channel-first
  |     (B, Lx, Ly, 3) -> (B, 3, Lx, Ly)
  |
  +-- Output: HopfOutput(base, fiber, transitions_x, transitions_y, quaternions)
```

### Why This Works

The Hopf map `S^3 -> S^2` identifies all quaternions in the same S^1 fiber: for any unit quaternion `g = cos(t) + k*sin(t)`, we have `hopf(q) = hopf(q*g)`. The fiber phase `phi = atan2(a3, a0)` parameterizes where within this S^1 orbit a given quaternion sits. The transition detector then finds sites where this fiber phase winds by +/-2*pi, corresponding to topological defects (vortices/anti-vortices) in the gauge field.

## 5. Data Flow: RealHopfLayer

The real Hopf fibration is the simplest of the family: a double covering of S^1 by itself.

```
Input: (..., 2) unit vectors on S^1
  |
  +-- Normalize to unit circle
  |     norm = sqrt(clamp(x^2 + y^2, min=eps))
  |     (x, y) = (x/norm, y/norm)
  |
  +-- alpha = atan2(y, x) mod 2*pi    (input angle)
  |
  +-- base = 2*alpha mod 2*pi         (double covering)
  |     z and -z map to the same base point
  |
  +-- fiber = sign(x) in {-1, +1}     (discrete S^0 fiber)
  |     Encodes which hemisphere the point lies in
  |     Zero defaults to +1
  |
  +-- Output: RealHopfOutput(base, fiber, input_angle)
```

### Inverse Reconstruction

```
alpha = base / 2             if fiber = +1
alpha = base / 2 + pi        if fiber = -1
(x, y) = (cos(alpha), sin(alpha))
```

Implementation note: the inverse selects between the two candidate angles `alpha1 = base/2` and `alpha2 = base/2 + pi` by checking which one's cosine matches the fiber sign.

## 6. Data Flow: QuaternionicHopfLayer

The quaternionic Hopf fibration uses the Cayley-Dickson construction to represent octonions as pairs of quaternions.

```
Input: p in R^4, q in R^4 (quaternion pair = octonion via Cayley-Dickson)
  |
  +-- Normalize to S^7: |p|^2 + |q|^2 = 1
  |     norm = sqrt(clamp(sum(p*p) + sum(q*q), min=eps))
  |     p, q = p/norm, q/norm
  |
  +-- Base (S^4 projection):
  |     pq* = quaternion_multiply(p, conjugate(q))    (quaternion product)
  |     diff = |p|^2 - |q|^2                          (scalar)
  |     base = (2*pq*[0], 2*pq*[1], 2*pq*[2], 2*pq*[3], diff) in R^5
  |     Satisfies |base| = 1 (on S^4)
  |
  +-- Fiber (S^3 element):
  |     fiber = p / |p|      (unit quaternion)
  |
  +-- Output: QuaternionicHopfOutput(base, fiber)
```

### Inverse Reconstruction

Given `(base, fiber)`:

1. Extract `|p|^2 = (1 + diff)/2` and `|q|^2 = (1 - diff)/2` from the 5th base component.
2. Recover `p = |p| * fiber`.
3. Recover `q` from `pq* = base[:4]/2` by solving `q* = fiber^{-1} * (pq*) / |p|`.

### Cayley-Dickson Multiplication

The octonion product is defined for pairs of quaternions:

```
(p, q) * (r, s) = (pr - conj(s)*q,  s*p + q*conj(r))
```

where `conj()` denotes quaternion conjugation. Octonion multiplication is **non-associative** -- this is verified in tests.

## 7. Gradient Flow Design

The entire package is designed for end-to-end differentiable use inside PyTorch training loops. Several design decisions ensure stable gradient flow:

### clipped_atan2: Straight-Through Estimator

```python
class _ClippedAtan2Grad(torch.autograd.Function):
    # Forward:  exact atan2(y, x)
    # Backward: d(atan2)/dy =  x / (x^2 + y^2)  -- clamped to [-max_grad, max_grad]
    #           d(atan2)/dx = -y / (x^2 + y^2)  -- clamped to [-max_grad, max_grad]
    #           Denominator clamped: max(x^2 + y^2, EPS)
```

- **Forward** is mathematically exact -- no approximation.
- **Backward** clips gradient magnitude to `max_grad` (default: 100.0) near the singularity where `x = y = 0`.
- This is a **straight-through estimator (STE)**: the forward value is exact, but the backward pass is modified for stability.
- The denominator `x^2 + y^2` is clamped to `min=EPS=1e-8` to prevent division by zero.

### Soft Transition Detection

Instead of the non-differentiable `sign(jump)`:

```python
signal = tanh(jump / temperature)    # temperature controls sharpness
```

- `temperature = 0.5` (default): moderate smoothing, suitable for lattice spacing `a = 1`.
- As `temperature -> 0`: approaches hard `sign()`, sharper detection but steeper gradients.
- As `temperature -> inf`: all signals shrink toward 0, no detection.
- The `tanh` function is smooth everywhere, so gradients flow freely.

### Normalization Guards

All normalization operations protect against zero-norm inputs:

```python
norm = sqrt(clamp(sum(q * q, dim=-1), min=eps))    # eps = 1e-8
q = q / norm
```

This pattern appears in:
- `quaternion_normalize()` -- projecting to S^3
- `ClassicalHopfLayer` input normalization
- `QuaternionicHopfLayer` S^7 normalization
- `hopf_inverse` re-normalization
- `quaternion_inverse()` -- denominator guard

### No Discontinuities

All maps are smooth functions on their domains. The only mathematical singularities are:
- `atan2(0, 0)` -- handled by clipped gradient
- South pole of S^2 in `hopf_inverse` (`z = -1`, where the canonical section degenerates) -- handled by clamping `a0` to `min=eps`
- Zero-norm quaternions -- handled by eps-clamped normalization

These singularities have measure zero and are handled by epsilon clamping without altering the function values at non-singular points.

## 8. Input Format Conventions

| Layer | Input Shape | Description | Notes |
|-------|------------|-------------|-------|
| `ClassicalHopfLayer` | `(B, 4, Lx, Ly)` | Site field | One quaternion per lattice site |
| `ClassicalHopfLayer` | `(B, 4, 2, Lx, Ly)` | Link field | One quaternion per link direction (2 dirs in 2D) |
| `RealHopfLayer` | `(..., 2)` | Unit vectors | Arbitrary batch dims, last dim must be 2 |
| `QuaternionicHopfLayer` | `p: (..., 4)`, `q: (..., 4)` | Quaternion pair | Two quaternions forming an octonion |

The `ClassicalHopfLayer` automatically dispatches based on input dimensionality:
- 4D tensor -> site field path (`_forward_site`)
- 5D tensor -> link field path (`_forward_link`)
- Other -> `ValueError`

Input normalization is always applied: raw inputs need not lie exactly on the sphere.

## 9. Output Format Conventions

### HopfOutput (ClassicalHopfLayer)

| Field | Shape (site) | Shape (link) | Range | Description |
|-------|-------------|-------------|-------|-------------|
| `base` | `(B, 3, Lx, Ly)` | `(B, 3, 2, Lx, Ly)` | x^2+y^2+z^2 = 1 | S^2 coordinates (channel-first) |
| `fiber` | `(B, Lx, Ly)` | `(B, 2, Lx, Ly)` | [0, 2*pi) | S^1 fiber phase |
| `transitions_x` | `(B, Lx, Ly)` | `(B, 2, Lx, Ly)` | [-1, 1] | Soft winding in x-direction |
| `transitions_y` | `(B, Lx, Ly)` | `(B, 2, Lx, Ly)` | [-1, 1] | Soft winding in y-direction |
| `quaternions` | `(B, Lx, Ly, 4)` | `(B, 2, Lx, Ly, 4)` | |q| = 1 | Normalized input quaternions |

### RealHopfOutput (RealHopfLayer)

| Field | Shape | Range | Description |
|-------|-------|-------|-------------|
| `base` | `(...)` | [0, 2*pi) | Double-covering angle on S^1 |
| `fiber` | `(...)` | {-1, +1} | Discrete S^0 fiber (hemisphere sign) |
| `input_angle` | `(...)` | [0, 2*pi) | Original angle atan2(y, x) |

### QuaternionicHopfOutput (QuaternionicHopfLayer)

| Field | Shape | Range | Description |
|-------|-------|-------|-------------|
| `base` | `(..., 5)` | |base| = 1 | Point on S^4 in R^5 |
| `fiber` | `(..., 4)` | |fiber| = 1 | Unit quaternion on S^3 |

## 10. Quaternion Algebra Module

Convention: `q = (a0, a1, a2, a3) = a0 + a1*i + a2*j + a3*k`

All functions operate on the **last dimension** of the input tensor, supporting arbitrary batch dimensions via `...` indexing.

| Function | Signature | Description |
|----------|-----------|-------------|
| `quaternion_multiply(p, q)` | `(..., 4), (..., 4) -> (..., 4)` | Hamilton product: `(p0+p1i+p2j+p3k)(q0+q1i+q2j+q3k)` using `i^2=j^2=k^2=ijk=-1` |
| `quaternion_conjugate(q)` | `(..., 4) -> (..., 4)` | `q* = (a0, -a1, -a2, -a3)`. For unit quaternions, `conjugate == inverse`. |
| `quaternion_normalize(q, eps)` | `(..., 4) -> (..., 4)` | `q / |q|`, projecting to S^3. Denominator clamped to `min=eps`. |
| `quaternion_inverse(q, eps)` | `(..., 4) -> (..., 4)` | `q^{-1} = q* / |q|^2`. For unit quaternions, equivalent to conjugate. |
| `quaternion_norm(q)` | `(..., 4) -> (...)` | `|q| = sqrt(a0^2 + a1^2 + a2^2 + a3^2)` |
| `quaternion_to_su2(q)` | `(..., 4) -> (..., 2, 2) complex` | Maps unit quaternion to SU(2) matrix: `[[a0+ia3, a2+ia1], [-a2+ia1, a0-ia3]]` |
| `su2_to_quaternion(U)` | `(..., 2, 2) complex -> (..., 4)` | Inverse of `quaternion_to_su2`. Extracts `(Re(U00), Im(U01), Re(U01), Im(U00))`. |

Implementation detail: conjugation uses a cached `[1, -1, -1, -1]` sign tensor, lazily initialized per device and dtype.

## 11. Octonion Algebra (Cayley-Dickson)

Octonions are represented as tuples of two quaternion tensors: `(p, q)` where `p, q` each have shape `(..., 4)`.

| Function | Signature | Mathematical Definition |
|----------|-----------|------------------------|
| `octonion_multiply(a, b)` | `((T,T), (T,T)) -> (T,T)` | `(p,q)(r,s) = (pr - conj(s)*q, s*p + q*conj(r))` |
| `octonion_conjugate(o)` | `(T,T) -> (T,T)` | `(p, q)* = (conj(p), -q)` |
| `octonion_norm(o)` | `(T,T) -> T` | `|o| = sqrt(|p|^2 + |q|^2)` |

where `T = Tensor` with shape `(..., 4)`, and `conj()` denotes quaternion conjugation.

**Important**: Octonion multiplication is **non-associative**. In general:

```
(a * b) * c  !=  a * (b * c)
```

This is a fundamental mathematical property (the octonions form a non-associative normed division algebra) and is verified in the test suite.

The Cayley-Dickson construction builds each algebra from the previous:

```
Reals  --Cayley-Dickson-->  Complex numbers  (lose ordering)
Complex  --Cayley-Dickson-->  Quaternions    (lose commutativity)
Quaternions  --Cayley-Dickson-->  Octonions  (lose associativity)
```

## 12. Testing Architecture

```
tests/
    conftest.py                 -- Shared fixtures
    test_classical_hopf.py      -- 12 tests: S^2 constraint, fiber, transitions, forward paths
    test_real_hopf.py           -- 14 tests: double covering, reconstruction, edge cases
    test_quaternionic_hopf.py   -- 20 tests: octonion algebra, S^4/S^3, reconstruction
    test_quaternion.py          -- 13 tests: Hamilton product, conjugate, normalize, SU(2)
    test_reconstruction.py      --  4 tests: hopf_inverse round-trip
    test_transitions.py         --  6 tests: winding detection, temperature sensitivity
    test_gradient_flow.py       --  4 tests: end-to-end gradient propagation
    test_mc.py                  -- 16 tests: Monte Carlo SU(2) generation
    test_mc_higgs.py            -- 10 tests: Monte Carlo SU(2)+Higgs generation
                                   ---
                            Total: 99 tests
```

### Test Categories

| Category | What is Verified | Example |
|----------|-----------------|---------|
| **Sphere constraints** | Output lies on correct sphere (S^2, S^4, etc.) | `|base|^2 == 1` within tolerance |
| **Fiber extraction** | Correct fiber phase / sign | `phi = atan2(a3, a0) mod 2*pi` |
| **Fiber action invariance** | Base is invariant under fiber rotation | `hopf(q) == hopf(q * g)` for fiber `g` |
| **Reconstruction round-trip** | `inverse(forward(x)) == x` | `hopf_inverse(base, fiber) == q` |
| **Gradient flow** | Gradients propagate through all operations | `q.grad is not None` after `loss.backward()` |
| **Edge cases** | Singularities, zero inputs, boundary values | South pole, zero quaternion, exact fiber = 0 |
| **Double covering** | Antipodal points map to same base | `real_hopf(v) == real_hopf(-v)` (base only) |
| **Non-associativity** | Octonion product is not associative | `(a*b)*c != a*(b*c)` for generic octonions |
| **MC correctness** | Metropolis-generated configs satisfy physics | Plaquette values, acceptance rates |

## 13. Experiment Infrastructure

The experiment framework provides a standardized ablation study pipeline comparing raw quaternion inputs against progressively richer Hopf-decomposed features.

```
experiments/
    shared/
        ablations.py   -- AblationMode enum, HopfFeatureExtractor, channel counts
        models.py      -- CNNBackbone, ClassificationHead, RegressionHead, ExperimentModel
        data.py        -- GaugeDataset, HiggsDataset, PrecomputedFeatureDataset
        training.py    -- train_classification, train_regression, TrainConfig, TrainResult
    mc_generation/
        su2_metropolis.py   -- Pure SU(2) Metropolis Monte Carlo
        su2_higgs.py        -- SU(2)+Higgs Monte Carlo
        analytical.py       -- Analytical observables
        generate_configs.py -- Config generation scripts
    exp1_phase_classification/
        run_exp1.py    -- Phase classification experiment
    exp2_topological_charge/
        run_exp2.py    -- Topological charge regression
        charge_utils.py
    exp3_rotation_denoising/
        run_exp3.py    -- Rotation denoising experiment
        rotation_utils.py
```

### Ablation Modes

The four ablation modes isolate the contribution of each Hopf decomposition component:

| Mode | Channels | Input to CNN | What It Tests |
|------|----------|-------------|---------------|
| `RAW` | 8 | `(B, 8, Lx, Ly)` -- raw quaternion link field | Baseline: no geometric decomposition |
| `BASE_ONLY` | 6 | `(B, 6, Lx, Ly)` -- S^2 base coords x 2 dirs | Value of gauge-invariant projection |
| `BASE_FIBER` | 8 | `(B, 8, Lx, Ly)` -- base + fiber phase | Value of fiber phase information |
| `FULL_HOPF` | 12 | `(B, 12, Lx, Ly)` -- base + fiber + transitions | Full topological feature set |

Channel counts assume 2 link directions (2D lattice). The `HopfFeatureExtractor` class handles the conversion from `(B, 4, 2, Lx, Ly)` gauge links to the appropriate `(B, C, Lx, Ly)` feature tensor.

### Model Architecture

All experiments use the same CNN backbone for fair comparison:

```
Input (B, C, Lx, Ly)
  |
  +-- ConvBlock(C -> 32):    Conv2d + BatchNorm2d + ReLU
  +-- ConvBlock(32 -> 64):   Conv2d + BatchNorm2d + ReLU
  +-- ConvBlock(64 -> 128):  Conv2d + BatchNorm2d + ReLU
  +-- AdaptiveAvgPool2d(1):  Global average pooling -> (B, 128)
  |
  +-- Task Head:
       Classification: Linear(128, 64) + ReLU + Dropout(0.2) + Linear(64, num_classes)
       Regression:     Linear(128, 64) + ReLU + Dropout(0.2) + Linear(64, 1)
```

### Training Pipeline

- Optimizer: AdamW with weight decay 1e-4
- Classification loss: CrossEntropyLoss; metric: accuracy
- Regression loss: MSELoss; metric: R^2
- Early stopping with configurable patience (default: 10 epochs)
- Train/val split (default: 80/20) with fixed seed for reproducibility

## 14. Design Principles

1. **No learnable parameters in fibration layers.** The Hopf map is a fixed mathematical construction. Placing learnable parameters inside the map would destroy its topological guarantees (e.g., the S^2 constraint on the base). All learning happens in downstream networks.

2. **Mathematical correctness verified by property-based tests.** Tests verify algebraic identities (`|hopf(q)| = 1`, `hopf(q*g) = hopf(q)`, `inverse(forward(q)) = q`) rather than just checking specific numerical values. This provides stronger guarantees.

3. **All operations numerically stable.** Every division, square root, and inverse trigonometric function is guarded by epsilon clamping. Gradient magnitudes are bounded by the clipped-atan2 STE. No operation can produce NaN or Inf for finite inputs.

4. **Consistent API across all three fibrations.** Each layer:
   - Is an `nn.Module` subclass
   - Takes normalized (or normalizable) sphere inputs
   - Returns a structured output with `base` and `fiber` fields
   - Has an `inverse()` method for reconstruction

5. **Channel-first convention.** The `ClassicalHopfLayer` follows PyTorch's CNN convention: spatial tensors use `(B, C, H, W)` layout. Internally the layer permutes to `(..., 4)` last-dim for the algebra, then permutes back before returning.

6. **Separation of concerns.** The Hopf map, fiber extraction, and transition detection are separate methods/modules that can be used independently. The `TransitionDetector` can be applied to any 2D scalar phase field, not just Hopf fibers.
