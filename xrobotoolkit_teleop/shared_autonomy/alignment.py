"""目标与夹爪像素对准误差。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from xrobotoolkit_teleop.shared_autonomy.gripper_detection import GripperDetectionResult
from xrobotoolkit_teleop.shared_autonomy.target_detection import TargetDetectionResult


@dataclass
class AlignmentError:
    """像素对准误差：du = u_target - u_gripper, dv = v_target - v_gripper。"""

    valid: bool
    du: float = 0.0
    dv: float = 0.0
    u_target: int | None = None
    v_target: int | None = None
    u_gripper: int | None = None
    v_gripper: int | None = None


def compute_alignment_error(
    target: TargetDetectionResult,
    gripper: GripperDetectionResult,
) -> AlignmentError:
    if (
        not target.found
        or not gripper.found
        or target.center_u is None
        or target.center_v is None
        or gripper.gripper_u is None
        or gripper.gripper_v is None
    ):
        return AlignmentError(valid=False)

    du = float(target.center_u - gripper.gripper_u)
    dv = float(target.center_v - gripper.gripper_v)
    return AlignmentError(
        valid=True,
        du=du,
        dv=dv,
        u_target=target.center_u,
        v_target=target.center_v,
        u_gripper=gripper.gripper_u,
        v_gripper=gripper.gripper_v,
    )


def draw_alignment_overlay(
    frame_bgr: np.ndarray,
    alignment: AlignmentError,
    *,
    line_color_bgr: tuple[int, int, int] = (0, 255, 255),
    text_color_bgr: tuple[int, int, int] = (0, 255, 255),
) -> np.ndarray:
    """绘制对准连线与 du/dv 文字（青色）。"""
    out = frame_bgr
    if not alignment.valid:
        cv2.putText(
            out,
            "alignment: N/A",
            (12, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color_bgr,
            2,
            cv2.LINE_AA,
        )
        return out

    p0 = (alignment.u_gripper, alignment.v_gripper)
    p1 = (alignment.u_target, alignment.v_target)
    cv2.arrowedLine(out, p0, p1, line_color_bgr, 2, tipLength=0.15, line_type=cv2.LINE_AA)
    label = f"du={alignment.du:+.0f}  dv={alignment.dv:+.0f} px"
    cv2.putText(
        out,
        label,
        (12, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color_bgr,
        2,
        cv2.LINE_AA,
    )
    return out
