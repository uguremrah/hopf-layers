"""hopf-layers: Differentiable fiber bundle decompositions for geometric deep learning.

Provides neural network layers implementing the Hopf fibrations as differentiable
PyTorch modules. Each layer decomposes structured input (points on a sphere) into
base space coordinates, fiber phases, and transition/winding signals — all with
full gradient flow.

Implemented fibrations:
    - Classical (complex): S¹ → S³ → S² (quaternion → base + fiber)
    - Real: S⁰ → S¹ → S¹ (circle → semicircle + sign)
    - Quaternionic: S³ → S⁷ → S⁴ (octonion pair → base + fiber)

Example::

    import torch
    from hopf_layers import ClassicalHopfLayer

    layer = ClassicalHopfLayer()
    q = torch.randn(8, 4, 16, 16)  # batch of quaternion fields
    output = layer(q)

    output.base          # (8, 3, 16, 16) — S² coordinates
    output.fiber         # (8, 16, 16)    — S¹ phases
    output.transitions_x # (8, 16, 16)    — x-direction winding
    output.transitions_y # (8, 16, 16)    — y-direction winding
"""

__version__ = "0.1.0"

from hopf_layers.classical import ClassicalHopfLayer, HopfOutput
from hopf_layers.device import get_device
from hopf_layers.quaternion import (
    quaternion_conjugate,
    quaternion_inverse,
    quaternion_multiply,
    quaternion_normalize,
    quaternion_to_su2,
    su2_to_quaternion,
)
from hopf_layers.quaternionic import (
    QuaternionicHopfLayer,
    QuaternionicHopfOutput,
    octonion_conjugate,
    octonion_multiply,
    octonion_norm,
)
from hopf_layers.real import RealHopfLayer, RealHopfOutput
from hopf_layers.reconstruction import hopf_inverse
from hopf_layers.transitions import TransitionDetector

__all__ = [
    # Device utility
    "get_device",
    # Core layers
    "ClassicalHopfLayer",
    "HopfOutput",
    "QuaternionicHopfLayer",
    "QuaternionicHopfOutput",
    "RealHopfLayer",
    "RealHopfOutput",
    "TransitionDetector",
    # Reconstruction
    "hopf_inverse",
    # Quaternion algebra
    "quaternion_normalize",
    "quaternion_conjugate",
    "quaternion_inverse",
    "quaternion_multiply",
    "quaternion_to_su2",
    "su2_to_quaternion",
    # Octonion algebra
    "octonion_multiply",
    "octonion_conjugate",
    "octonion_norm",
]
