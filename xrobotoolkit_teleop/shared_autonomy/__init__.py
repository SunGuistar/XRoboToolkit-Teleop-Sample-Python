"""Shared Autonomy 辅助对准模块（视觉检测、视觉伺服、自动下降等）。"""

from xrobotoolkit_teleop.shared_autonomy.alignment import (
    AlignmentError,
    compute_alignment_error,
    draw_alignment_overlay,
)
from xrobotoolkit_teleop.shared_autonomy.config import TargetDetectionConfig
from xrobotoolkit_teleop.shared_autonomy.gripper_color_config import GripperColorConfig
from xrobotoolkit_teleop.shared_autonomy.gripper_detection import (
    GripperDetectionResult,
    GripperSmoother,
    detect_gripper,
    draw_gripper_overlay,
)
from xrobotoolkit_teleop.shared_autonomy.target_detection import (
    TargetDetectionResult,
    TargetSmoother,
    detect_and_draw,
    detect_target,
    draw_target_overlay,
)

__all__ = [
    "TargetDetectionConfig",
    "GripperColorConfig",
    "TargetDetectionResult",
    "GripperDetectionResult",
    "AlignmentError",
    "TargetSmoother",
    "GripperSmoother",
    "detect_target",
    "detect_gripper",
    "compute_alignment_error",
    "draw_target_overlay",
    "draw_gripper_overlay",
    "draw_alignment_overlay",
    "detect_and_draw",
]
