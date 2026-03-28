"""AnyDexRetarget - Hand Pose Retargeting Module.

Provides hand pose retargeting from MediaPipe format to dexterous robot hand joint angles.

Main classes:
- Retargeter: High-level unified interface (recommended)
- BaseOptimizer: Low-level optimizer access

Example:
    from anydexretarget import Retargeter

    retargeter = Retargeter.from_yaml("config/mediapipe/mediapipe_shadow_hand.yaml", hand_side="right")
    qpos = retargeter.retarget(raw_keypoints)  # (21, 3) -> (22,)
"""

from .retarget import Retargeter
from .optimizer import BaseOptimizer, LPFilter
from .mediapipe import apply_mediapipe_transformations

__all__ = [
    "Retargeter",
    "BaseOptimizer",
    "LPFilter",
    "apply_mediapipe_transformations",
]
