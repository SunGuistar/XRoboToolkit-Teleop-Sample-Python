#!/usr/bin/env python3
"""Shared Autonomy 视觉联调：目标(蓝) + 夹爪(绿) + 像素误差 du/dv。

  python scripts/hardware/debug_alignment_vision.py --serial 148522072134
  python scripts/hardware/debug_alignment_vision.py --serial 148522072134 --show-gripper-mask

  # 夹爪仍不在浅黄框内：按画面微调 ROI（u 左右, v 上下，均为 0~1）
  python scripts/hardware/debug_alignment_vision.py \\
    --serial 148522072134 --grip-roi-x-min 0.20 --grip-roi-x-max 0.95 \\
    --grip-roi-y-min 0.0 --grip-roi-y-max 0.75

窗口：q 退出，s 打印对准误差
"""

from __future__ import annotations

import sys
from typing import Literal

import cv2
import numpy as np
import pyrealsense2 as rs
import tyro

from xrobotoolkit_teleop.shared_autonomy.alignment import compute_alignment_error, draw_alignment_overlay
from xrobotoolkit_teleop.shared_autonomy.config import COLOR_HSV_PRESETS, TargetDetectionConfig
from xrobotoolkit_teleop.shared_autonomy.gripper_color_config import GripperColorConfig
from xrobotoolkit_teleop.shared_autonomy.gripper_detection import (
    GripperSmoother,
    detect_gripper,
    draw_gripper_overlay,
)
from xrobotoolkit_teleop.shared_autonomy.target_detection import TargetSmoother, detect_target, draw_target_overlay

DEFAULT_AUX_SERIAL = "148522072134"


def _list_devices() -> list[tuple[str, str]]:
    devices = []
    for dev in rs.context().query_devices():
        devices.append((dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number)))
    return devices


def main(
    serial: str = DEFAULT_AUX_SERIAL,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    show_mask: bool = False,
    show_gripper_mask: bool = False,
    color_preset: Literal["target", "silver", "white", "black"] = "target",
    hsv_lower: tuple[int, int, int] | None = None,
    hsv_upper: tuple[int, int, int] | None = None,
    roi_y_min: float = 0.46,
    roi_y_max: float = 0.88,
    roi_x_min: float = 0.22,
    roi_x_max: float = 0.58,
    grip_roi_y_min: float = 0.02,
    grip_roi_y_max: float = 0.82,
    grip_roi_x_min: float = 0.28,
    grip_roi_x_max: float = 0.98,
    grip_hsv_lower: tuple[int, int, int] = (0, 0, 0),
    grip_hsv_upper: tuple[int, int, int] = (180, 255, 70),
    grip_min_area: float = 1200.0,
    grip_offset_u: float = 0.0,
    grip_offset_v: float = 0.0,
    enable_target_smooth: bool = True,
    enable_gripper_smooth: bool = True,
) -> None:
    devices = _list_devices()
    if not devices:
        print("未找到 RealSense 相机。", file=sys.stderr)
        raise SystemExit(1)
    if serial not in {s for _, s in devices}:
        print(f"serial {serial} 不在已连接设备列表中。", file=sys.stderr)
        raise SystemExit(1)

    preset_lower, preset_upper = COLOR_HSV_PRESETS[color_preset]
    target_config = TargetDetectionConfig(
        hsv_lower=preset_lower if hsv_lower is None else hsv_lower,
        hsv_upper=preset_upper if hsv_upper is None else hsv_upper,
        roi_y_min_ratio=roi_y_min,
        roi_y_max_ratio=roi_y_max,
        roi_x_min_ratio=roi_x_min,
        roi_x_max_ratio=roi_x_max,
        enable_temporal_smooth=enable_target_smooth,
    )
    gripper_config = GripperColorConfig(
        hsv_lower=grip_hsv_lower,
        hsv_upper=grip_hsv_upper,
        roi_y_min_ratio=grip_roi_y_min,
        roi_y_max_ratio=grip_roi_y_max,
        roi_x_min_ratio=grip_roi_x_min,
        roi_x_max_ratio=grip_roi_x_max,
        min_contour_area=grip_min_area,
        gripper_offset_u=grip_offset_u,
        gripper_offset_v=grip_offset_v,
        enable_temporal_smooth=enable_gripper_smooth,
    )

    target_smoother = TargetSmoother(target_config)
    gripper_smoother = GripperSmoother(gripper_config)

    print(f"辅助相机 {serial}  目标={color_preset}")
    print(f"夹爪 HSV upper={grip_hsv_upper}  GRIPPER ROI x=[{grip_roi_x_min},{grip_roi_x_max}] y=[{grip_roi_y_min},{grip_roi_y_max}]")
    print("黄框=目标ROI  浅黄=夹爪ROI  蓝=目标  绿=夹爪  青=du/dv")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    pipeline.start(rs_config)
    window = f"alignment_vision [{serial}]"

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = np.asanyarray(frames.get_color_frame().get_data())

            target_raw = detect_target(color, target_config, return_mask=show_mask)
            target = target_smoother.smooth(target_raw)
            grip_raw = detect_gripper(color, gripper_config, return_mask=show_gripper_mask)
            grip = gripper_smoother.smooth(grip_raw)
            alignment = compute_alignment_error(target, grip)

            overlay = draw_target_overlay(color, target, target_config)
            overlay = draw_gripper_overlay(overlay, grip, gripper_config)
            overlay = draw_alignment_overlay(overlay, alignment)

            panels = [overlay]
            if show_mask and target_raw.mask is not None:
                panels.append(cv2.cvtColor(target_raw.mask, cv2.COLOR_GRAY2BGR))
            if show_gripper_mask and grip_raw.mask is not None:
                panels.append(cv2.cvtColor(grip_raw.mask, cv2.COLOR_GRAY2BGR))
            display = np.hstack(panels) if len(panels) > 1 else overlay

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if alignment.valid:
                    print(
                        f"[对准] target=({alignment.u_target},{alignment.v_target})  "
                        f"gripper=({alignment.u_gripper},{alignment.v_gripper})  "
                        f"du={alignment.du:+.1f}  dv={alignment.dv:+.1f}"
                    )
                else:
                    print(f"[对准] 无效  target={target.found}  gripper={grip.found}")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main)
