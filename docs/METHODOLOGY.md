# Methodology

## 1. Introduction

The `hopf-layers` package implements a novel approach to geometric deep learning: using the mathematical structure of Hopf fibrations to decompose structured signals into geometrically meaningful components. Unlike learned feature extractors, this decomposition is exact, parameter-free, and preserves the topological structure of the input data.

The key idea is to treat quaternion-valued fields --- such as those arising from SU(2) lattice gauge theory --- not as opaque 4-vectors, but as points on S^3, which carry a natural fiber bundle structure via the Hopf fibration. The package decomposes each quaternion into a gauge-invariant base point on S^2, a gauge phase on S^1, and differentiable winding signals that detect topological defects. Every operation is implemented as a differentiable PyTorch module with full gradient flow, enabling end-to-end training of neural networks that exploit this geometric prior.

The package implements all three Hopf fibrations over the real, complex, and quaternionic division algebras:

| Fibration | Fiber | Total Space | Base Space | Implementation |
|-----------|-------|-------------|------------|----------------|
| Real | S^0 = {+1, -1} | S^1 | S^1 | `RealHopfLayer` |
| Classical | S^1 (circle) | S^3 (SU(2)) | S^2 (sphere) | `ClassicalHopfLayer` |
| Quaternionic | S^3 (SU(2)) | S^7 | S^4 | `QuaternionicHopfLayer` |

## 2. Mathematical Foundations

### 2.1 Fiber Bundles

A fiber bundle `(E, B, pi, F)` consists of:
- **Total space** `E`: the space of all data
- **Base space** `B`: the "essential" or gauge-invariant part
- **Projection** `pi: E -> B`: the map that forgets the fiber
- **Fiber** `F`: the "redundant" or gauge-dependent part

The defining property is local triviality: for each point `b` in `B`, there is a neighborhood `U` such that `pi^{-1}(U)` is homeomorphic to `U x F`. The Hopf fibrations are the canonical non-trivial examples, where the total space is a sphere that is globally twisted relative to the product `B x F`.

### 2.2 The Three Hopf Fibrations

#### Real Hopf Fibration: S^0 -> S^1 -> S^1

The simplest Hopf fibration decomposes unit 2-vectors (points on S^1) into an angle on the quotient circle and a discrete sign.

- **Total space**: S^1 (unit circle in R^2)
- **Base space**: S^1 (via double covering: z and -z map to the same base point)
- **Fiber**: S^0 = {+1, -1} (hemisphere sign)
- **Map**: For a unit vector `(x, y)` with angle `alpha = atan2(y, x)`:
  ```
  base  = 2*alpha  mod 2*pi    (double covering angle)
  fiber = sign(x)               (hemisphere: +1 or -1)
  ```
- **Section (inverse)**: Given base angle `beta` and fiber sign `s`:
  ```
  alpha = beta/2           if s = +1
  alpha = beta/2 + pi      if s = -1
  (x, y) = (cos(alpha), sin(alpha))
  ```
- **Key property**: Antipodal points z and -z on S^1 project to the same base point but have opposite fiber signs. As the input traverses S^1 once, the base traverses S^1 twice (double cover).

This layer is implemented in `hopf_layers.real.RealHopfLayer`.

#### Classical Hopf Fibration: S^1 -> S^3 -> S^2

The central construction of the package. It decomposes unit quaternions (points on S^3, equivalently elements of SU(2)) into a direction on the 2-sphere and a U(1) phase.

- **Total space**: S^3, diffeomorphic to SU(2) (group of unit quaternions)
- **Base space**: S^2 (unit sphere in R^3)
- **Fiber**: S^1 (circle, equivalently U(1) phase)
- **Hopf map**: For a unit quaternion `q = (a0, a1, a2, a3)`:
  ```
  x = 2*(a1*a3 + a0*a2)
  y = 2*(a2*a3 - a0*a1)
  z = a0^2 + a3^2 - a1^2 - a2^2
  ```
  This is equivalent to the map `q -> q i q*` under the standard embedding of R^3 into the imaginary quaternions.
- **Fiber phase**: `phi = atan2(a3, a0)`, shifted to `[0, 2*pi)`
- **Key property**: Right multiplication by `e^{i*phi} = (cos(phi), 0, 0, sin(phi))` changes the fiber phase while preserving the base point. This is exactly the U(1) gauge transformation in the physics context.

**Canonical section (inverse)**:

Given a base point `(x, y, z)` on S^2, the canonical lift to S^3 with fiber phase 0 is:
```
a0 = sqrt((1 + z) / 2)
a1 = -y / (2*a0)
a2 =  x / (2*a0)
a3 = 0
```
To apply a fiber phase `phi`, right-multiply by `(cos(phi), 0, 0, sin(phi))` via the Hamilton product. The full reconstruction is exact and deterministic (implemented in `hopf_layers.reconstruction.hopf_inverse`).

This layer is implemented in `hopf_layers.classical.ClassicalHopfLayer`.

#### Quaternionic Hopf Fibration: S^3 -> S^7 -> S^4

The highest-dimensional Hopf fibration, operating on pairs of quaternions via the Cayley-Dickson construction.

- **Total space**: S^7 (unit vector in H^2, i.e. a pair of quaternions `(p, q)` with `|p|^2 + |q|^2 = 1`)
- **Base space**: S^4 (quaternionic projective line HP^1, embedded in R^5)
- **Fiber**: S^3 (unit quaternions, equivalently SU(2))
- **Quaternionic Hopf map**: For `(p, q)` on S^7:
  ```
  base = (2*p*q*, |p|^2 - |q|^2)  in  R^4 x R = R^5
  ```
  where `p*q*` denotes the quaternion product of `p` with the conjugate of `q`, yielding four real components, and `|p|^2 - |q|^2` is a scalar. The resulting 5-vector lies on S^4.
- **Fiber extraction**: `g = p / |p|` (the unit quaternion direction of `p`)
- **Key property**: Right multiplication `(p, q) -> (p*g, q*g)` by a unit quaternion `g` preserves the base point but rotates the fiber. This is the SU(2) gauge transformation.

**Cayley-Dickson multiplication**:

An octonion is represented as a pair of quaternions `(p, q)`. The product is:
```
(p, q) * (r, s) = (p*r - conj(s)*q,  s*p + q*conj(r))
```
This multiplication is **not associative** --- a property explicitly verified in the test suite. The octonions are, however, a normed division algebra: `|a*b| = |a|*|b|`.

This layer is implemented in `hopf_layers.quaternionic.QuaternionicHopfLayer`.

### 2.3 Connection to Gauge Theory

SU(2) lattice gauge theory discretizes gauge fields onto the links of a lattice, with each link variable being a unit quaternion (an element of SU(2)). The Hopf fibration provides a physically motivated decomposition:

- **Base (S^2)**: Gauge-invariant directional content. Under a gauge transformation `q -> g*q` (left multiplication), the base point rotates by the corresponding SO(3) rotation, but the decomposition structure is preserved.
- **Fiber (S^1)**: Local gauge phase. Under right multiplication by a U(1) element, the base is invariant and only the fiber changes. This is the residual gauge freedom after fixing the direction.
- **Transitions**: Phase-winding signals that detect topological defects. When the fiber phase jumps by approximately 2*pi between neighboring lattice sites, this signals a vortex or monopole-like object.

The transition detector computes finite differences of the fiber phase field, unwraps them to `[-pi, pi)`, extracts the jump (the amount removed by unwrapping), and applies `tanh(jump / T)` to produce a differentiable signal in `[-1, 1]`.

## 3. Differentiability Design

A key contribution of `hopf-layers` is making these mathematically exact decompositions fully differentiable for use in gradient-based learning. Three specific challenges arise and are addressed with targeted solutions.

### 3.1 The Gradient Challenge

#### Challenge 1: atan2 Singularity

The fiber phase is extracted via `phi = atan2(a3, a0)`. The standard gradients of atan2 are:
```
d(atan2)/dy =  x / (x^2 + y^2)
d(atan2)/dx = -y / (x^2 + y^2)
```
At `(x, y) = (0, 0)`, these gradients diverge. In the quaternion context, this corresponds to `a0 = a3 = 0`, which occurs when the quaternion lies in the `(a1, a2)` plane.

**Solution**: Clipped-gradient atan2 via a custom `torch.autograd.Function`.

- **Forward pass**: Exact `atan2(y, x)` --- no approximation.
- **Backward pass**: Standard atan2 gradient, but with magnitude clamped to `[-max_grad, max_grad]`. The denominator `x^2 + y^2` is clamped to a minimum of `eps = 1e-8` before computing the gradient.
- **Effect**: This is a straight-through estimator (STE). The forward value is mathematically correct; only the backward pass is regularized. The default `max_grad = 100.0` is generous enough for typical lattice data while preventing gradient explosions near the singularity.

Implementation: `hopf_layers.utils._ClippedAtan2Grad` and `hopf_layers.utils.clipped_atan2`.

#### Challenge 2: Phase Wrapping Discontinuity

The fiber phase `phi` is defined in `[0, 2*pi)`. When phi crosses from just below `2*pi` to just above `0`, the numerical value jumps discontinuously, even though the underlying geometric quantity (a point on S^1) varies smoothly. Computing finite differences on the raw phase yields spurious large gradients at wrapping boundaries.

**Solution**: Soft transition detection via `tanh`.

1. Compute nearest-neighbor phase differences: `delta = phi[i+1] - phi[i]`
2. Unwrap to `[-pi, pi)`: `delta_unwrapped = (delta + pi) mod 2*pi - pi`
3. Extract the jump: `jump = delta - delta_unwrapped` (nonzero only when wrapping occurred)
4. Apply soft thresholding: `transition = tanh(jump / T)`

The temperature `T` controls sharpness:
- `T -> 0`: Hard sign function, crisp `{-1, 0, +1}` output
- `T = 0.5` (default): Smooth differentiable signal
- `T = 2.0`: Very soft, gradual transitions

All operations in this pipeline are differentiable (modular arithmetic via `torch.remainder`, `tanh`), so gradients flow back through the transition signals to the input quaternions.

Implementation: `hopf_layers.transitions.TransitionDetector`.

#### Challenge 3: Normalization Near Zero

Several operations require computing `sqrt(x)` where `x` can approach zero (e.g., normalizing a quaternion that is near-zero, or computing `sqrt((1+z)/2)` when the base point is near the south pole `z = -1`). The gradient of `sqrt(x)` is `1/(2*sqrt(x))`, which diverges as `x -> 0`.

**Solution**: Clamp the argument before taking the square root:
```
sqrt(clamp(x, min=eps))
```
with `eps = 1e-8` throughout. This caps the maximum gradient at `1/(2*sqrt(eps)) ~ 5000`, which is large but finite. Combined with the atan2 gradient clipping, this ensures no NaN or Inf gradients anywhere in the pipeline.

### 3.2 Gradient Flow Verification

All layers are verified through automated gradient tests (in `tests/test_gradient_flow.py`, `tests/test_classical_hopf.py`, `tests/test_quaternionic_hopf.py`, and `tests/test_real_hopf.py`). The verification protocol is:

1. **Per-output gradient flow**: For each output component (base, fiber, transitions_x, transitions_y), verify that `output.sum().backward()` produces non-zero, finite gradients on the input.
2. **Round-trip gradient flow**: Verify that `inverse(decompose(x)).sum().backward()` produces finite gradients on `x`.
3. **Gradient magnitude bounds**: Verify that gradient norms are bounded (< 1000) for typical random inputs, confirming that the STE clipping is effective.
4. **Singularity robustness**: Explicitly test with inputs near the atan2 singularity (`a0 ~ 1e-10, a3 ~ 1e-10`) and verify that gradients remain finite and bounded.

No NaN or Inf gradients are produced in any test configuration across all three Hopf layers.

## 4. Experimental Methodology

### 4.1 Ablation Framework

To rigorously isolate the contribution of each component of the Hopf decomposition, experiments use a systematic 4-way ablation design:

| Mode | Input Channels | Components | Purpose |
|------|----------------|------------|---------|
| RAW | 8 | Raw quaternion link variables (4 components x 2 directions) | Baseline: no geometric decomposition |
| BASE_ONLY | 6 | S^2 coordinates only (3 components x 2 directions) | What does the gauge-invariant base capture? |
| BASE_FIBER | 8 | S^2 + S^1 phase (3 + 1 = 4 components x 2 directions) | What does the fiber phase add? |
| FULL_HOPF | 12 | S^2 + S^1 + transitions (3 + 1 + 2 = 6 components x 2 directions) | What do winding transitions add? |

All experiments use an identical CNN architecture (3-layer convolutional network with AdaptiveAvgPool2d) varying only the number of input channels. This ensures that any performance difference is attributable to the geometric decomposition, not to differences in model capacity or architecture.

### 4.2 Experiment 1: SU(2) Phase Classification

- **Task**: Binary classification of SU(2) lattice gauge configurations by thermodynamic phase (confined vs. Higgs).
- **Data**: SU(2) + adjoint Higgs model simulated at different hopping parameter values (kappa) on an 8x8 lattice. Configurations below the phase transition are labeled "confined"; configurations above are labeled "Higgs".
- **Metric**: Test accuracy.
- **Expected outcome**: The Hopf decomposition should capture phase-distinguishing features in the base (gauge-invariant direction), with fiber and transitions providing additional robustness.
- **Result**: Full HopfLayer achieves 100% test accuracy, Base-only achieves 97.2%, Raw achieves 100%.
- **Insight**: The base space captures most of the phase information. The full decomposition matches raw performance, confirming that the geometric decomposition does not lose information.

### 4.3 Experiment 2: Topological Charge Detection

- **Task**: Regression of continuous topological charge `Q = (1/(2*pi)) * sum(arccos(a0_P))` from gauge configurations, where `a0_P` is the scalar part of the plaquette product.
- **Data**: 2D SU(2) pure gauge configurations at inverse coupling `beta in {1, 2, 4, 6}` on an 8x8 lattice.
- **Metric**: R^2 score (coefficient of determination).
- **Expected outcome**: Topological charge is inherently related to quaternion products around plaquettes. The raw quaternion representation contains this information directly. The Hopf decomposition may lose some information in the projection but should recover it through the transition signals.
- **Result**: Raw R^2 = 0.965, Full HopfLayer R^2 = 0.932, Base-only R^2 = 0.923.
- **Insight**: Raw quaternions contain topological charge information most directly (Q is computed from quaternion products around plaquettes, and the raw representation preserves this structure). However, Full HopfLayer > Base-only, confirming that the transition signals add measurable information about topological features.

### 4.4 Experiment 3: Rotation Denoising

- **Task**: Recover clean SO(3) rotation matrices from noisy quaternion fields.
- **Data**: Smooth random rotation fields on an 8x8 lattice, with additive Gaussian noise applied to the quaternion components.
- **Metric**: Geodesic distance on SO(3): `d(R1, R2) = arccos((tr(R1^T * R2) - 1) / 2)`.
- **Expected outcome**: The Hopf decomposition separates the rotation (base) from the phase (fiber), potentially enabling better denoising by operating on the geometrically meaningful components separately.
- **Result**: Raw geodesic distance = 1.14, Full HopfLayer = 1.43, Noisy baseline = 0.92.
- **Insight**: At small scale (100 training samples, 8x8 lattice), no model outperforms the noisy baseline. This validates the experimental pipeline (no information leakage, correct metric computation) but indicates that larger-scale experiments are needed to observe a benefit from the geometric decomposition.

### 4.5 Mutual Information Analysis

To understand what information each Hopf component captures, histogram-based mutual information (MI) estimation is performed:

1. Generate controlled SU(2) configurations at varying noise levels.
2. Compute the Hopf decomposition: base (S^2), fiber (S^1), transitions.
3. Estimate MI(component, target) for multiple target variables: noise level, plaquette energy, topological signal.
4. Use uniform-width histogram binning for the MI estimator.

This analysis quantifies the information-theoretic content of each decomposition component and identifies which physical observables are best captured by base, fiber, or transitions.

**Limitation**: Histogram-based MI estimation has known systematic bias (it tends to overestimate MI for small sample sizes and underestimate for continuous variables with complex distributions). Kernel density estimation or k-nearest-neighbor MI estimators would provide more accurate estimates.

### 4.6 Computational Cost Analysis

Performance benchmarks characterize the overhead of the Hopf decomposition:

- **Forward/backward timing**: Wall-clock time for each of the three layers (real, classical, quaternionic).
- **Spatial scaling**: Timing as a function of lattice size `L` from 4 to 64.
- **Batch scaling**: Timing as a function of batch size `B` from 1 to 64.
- **Memory overhead**: Peak memory delta measured via `tracemalloc`.
- **Environment**: CPU-only benchmarks for reproducibility across hardware. GPU timing would differ due to parallelism and memory bandwidth characteristics.

## 5. Validation Methodology

### 5.1 Property-Based Testing

Each layer is validated against the mathematical properties that define the Hopf fibration. Rather than checking outputs against hardcoded expected values, the test suite verifies invariants that must hold for any valid input.

#### Sphere Constraints

Every output must lie on the correct sphere:
- **Classical**: Base on S^2 (`|base|^2 = 1` with tolerance 1e-6), verified for both float32 and float64 inputs. A dedicated precision test with exact unit quaternions in float64 achieves error below 1e-12.
- **Quaternionic**: Base on S^4 (`|base|^2 = 1`), fiber on S^3 (`|fiber| = 1`).
- **Real**: Base angle in `[0, 2*pi)`, fiber in `{-1, +1}` (exactly).

#### Fiber Action Invariance

The defining property of a fiber bundle: the fiber group acts on the total space while preserving the base.
- **Classical**: Right multiplication by U(1) element `(cos(phi), 0, 0, sin(phi))` changes the fiber phase but preserves the base point on S^2. Left multiplication by SU(2) rotates the base but preserves the decomposition structure. The identity element preserves everything (verified explicitly).
- **Quaternionic**: Right multiplication `(p, q) -> (p*g, q*g)` by a unit quaternion `g` preserves the base on S^4 (to tolerance 1e-4) but changes the fiber (verified with non-identity `g`).
- **Real**: Antipodal points `z` and `-z` map to the same base but opposite fibers.

#### Reconstruction (Round-Trip)

The decomposition must be invertible:
- **Classical**: `decompose(x) -> (base, fiber) -> hopf_inverse(base, fiber) -> decompose` recovers the same base and fiber (tolerance 1e-3). The reconstructed quaternion has unit norm (tolerance 1e-5). Known values are tested: identity quaternion `(1, 0, 0, 0)` maps to north pole `(0, 0, 1)` with fiber phase 0.
- **Quaternionic**: `layer(p, q) -> (base, fiber) -> layer.inverse(base, fiber)` recovers the normalized `(p, q)` to tolerance 1e-4. Reconstructed pairs lie on S^7 (`|p|^2 + |q|^2 = 1`).
- **Real**: `layer(z) -> (base, fiber) -> layer.inverse(base, fiber)` recovers the original unit vector to tolerance 1e-5.

#### Algebraic Properties

The underlying algebras satisfy specific identities that are verified independently:
- **Quaternion norm multiplicativity**: `|p*q| = |p|*|q|` for all quaternions `p, q`.
- **Quaternion non-commutativity**: `i*j = k` but `j*i = -k` (verified with explicit basis elements).
- **Quaternion inverse**: `q * q^{-1} = 1` (identity quaternion) for all unit quaternions.
- **SU(2) unitarity**: `U * U^dagger = I` and `det(U) = 1` for all matrices produced by `quaternion_to_su2`.
- **SU(2) round-trip**: `su2_to_quaternion(quaternion_to_su2(q)) = q` for all unit quaternions.
- **Octonion norm multiplicativity**: `|a*b| = |a|*|b|` (normed division algebra property).
- **Octonion non-associativity**: `(a*b)*c != a*(b*c)` in general (verified explicitly with random elements; difference exceeds 0.01).
- **Octonion conjugate involution**: `(o*)* = o` for all octonions.
- **Octonion conjugate-norm**: `o * o* = |o|^2 * (1, 0, 0, 0, 0, 0, 0, 0)` (product with conjugate gives a real scalar times identity).

### 5.2 Topological Transition Validation

The `TransitionDetector` is validated against known configurations:

- **Uniform field**: Constant phase field produces zero transitions (verified below 1e-6).
- **Linear ramp**: Slowly varying phase produces near-zero transitions (below 0.1).
- **Phase jump**: A sharp 2*pi discontinuity produces strong transitions (above 0.5).
- **Output range**: All transitions lie in `[-1, 1]` due to the `tanh` activation (verified for random fields).
- **Temperature effect**: Lower temperature produces sharper signals; explicitly verified that `T = 0.1` gives stronger transitions than `T = 2.0` on the same discontinuity.
- **Vortex configuration**: A vortex field `phi(x, y) = atan2(y - cy, x - cx)` centered on the lattice produces detectable transitions in both x and y directions (above 0.1).

### 5.3 Automated Validation Gates

Validation notebooks contain numbered assertion gates with tight numerical tolerances (typically 1e-5) that account for floating-point arithmetic but leave no room for qualitative errors. Gate counts by layer:

| Layer | Number of Validation Gates |
|-------|---------------------------|
| Quaternionic Hopf | 12 |
| Classical Hopf | 7 |
| Real Hopf | 5 |
| Experiments (each) | 3 (convergence, metric threshold, correlation) |

## 6. Reproducibility

All experiments and benchmarks are designed for exact reproducibility:

- **Random seeds**: All random operations use explicit seeds via `torch.manual_seed`. Test fixtures use seed 42 by default; experiments run with seeds {42, 123} and report mean +/- standard deviation.
- **Deterministic data splits**: Training/validation/test splits use `torch.utils.data.random_split` with a seeded `torch.Generator` (`Generator().manual_seed(seed)`), ensuring identical splits across runs.
- **Deterministic operations**: No non-deterministic CUDA operations are used (all benchmarks are CPU-only).
- **ArXiv-compatible output**: PDF figures use fonttype 42 (TrueType) for embedding compatibility. Tables use LaTeX booktabs formatting. Raw data is saved as CSV/JSON for independent analysis.

## 7. Limitations and Future Work

### Current Limitations

1. **Small-scale experiments**: All experiments use 8x8 lattices with 100--300 samples. Results at this scale establish pipeline correctness and provide directional insights, but quantitative conclusions about the benefit of Hopf decomposition require scaling to larger lattices (32x32 or 64x64) and larger datasets (1000+ samples).

2. **CPU-only benchmarks**: All timing measurements are CPU-only for reproducibility. GPU performance characteristics would differ significantly due to parallelism, memory bandwidth, and kernel launch overhead. GPU benchmarks are planned but not yet implemented.

3. **MI estimation bias**: Histogram-based mutual information estimation has known systematic bias. For continuous variables, kernel density estimation or k-nearest-neighbor MI estimators (e.g., Kraskov-Stogbauer-Grassberger) would provide more accurate and less biased estimates.

4. **No quaternionic transition detection**: The `QuaternionicHopfLayer` does not yet implement transition detection analogous to the `TransitionDetector` used in the classical layer. The S^3 fiber of the quaternionic fibration is 3-dimensional, making winding detection more complex (requiring detection of elements of pi_3(S^3) = Z rather than pi_1(S^1) = Z).

5. **No learnable variants**: All layers are purely geometric with no learnable parameters. Future work could introduce learnable temperature in the transition detector, learnable sections (replacing the canonical section with a parametrized family), or learnable fiber weighting.

### Planned Extensions

- **GPU benchmarks and CUDA kernels**: Fused forward/backward kernels for the classical Hopf layer to reduce memory overhead from intermediate tensors.
- **Higher-dimensional lattices**: Extension to 3D and 4D lattice gauge theory, where the link structure has 3 or 4 directions per site.
- **Quaternionic transition detection**: Winding number computation for S^3-valued fiber fields using the quaternionic generalization of phase unwrapping.
- **Learnable Hopf layers**: Parametrized sections, learnable temperature schedules, and attention-weighted component aggregation.
- **Integration with equivariant networks**: Combining Hopf decomposition with SE(3)-equivariant architectures for applications in molecular dynamics and protein structure prediction.
