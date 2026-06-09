"""Shared Autonomy 辅助对准模块（视觉检测、视觉伺服、自动下降等）。"""

from xrobotoolkit_teleop.shared_autonomy.config import TargetDetectionConfig
from xrobotoolkit_teleop.shared_autonomy.target_detection import (
    TargetDetectionResult,
    TargetSmoother,
    detect_and_draw,
    detect_target,
    draw_target_overlay,
)

__all__ = [
    "TargetDetectionConfig",
    "TargetDetectionResult",
    "TargetSmoother",
    "detect_target",
    "draw_target_overlay",
    "detect_and_draw",
]
