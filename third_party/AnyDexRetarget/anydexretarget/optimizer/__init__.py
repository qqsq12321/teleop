"""Optimizers for hand retargeting.

AdaptiveOptimizerAnalytical - Recommended optimizer using Huber loss + analytical gradients + NLopt SLSQP.
Uses adaptive blending between TipDirVec and FullHandVec based on pinch distance.

All parameters are read from YAML configuration files.
"""

from .base_optimizer import BaseOptimizer
from .utils import (
    LPFilter,
    TimingStats,
    M_TO_CM,
    CM_TO_M,
)
from .analytical_optimizer import AdaptiveOptimizerAnalytical


__all__ = [
    "BaseOptimizer",
    "AdaptiveOptimizerAnalytical",
    "LPFilter",
    "TimingStats",
    "M_TO_CM",
    "CM_TO_M",
]
