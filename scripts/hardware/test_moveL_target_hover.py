#!/usr/bin/env python3
"""检测 + 深度 + 外参 -> 基座目标点 -> MoveL hover（默认 dry-run）。

  # 先标定
  python scripts/hardware/calibrate_fixed_camera.py --serial 148522072134

  # 只打印位姿，不动臂
  python scripts/hardware/test_moveL_target_hover.py --serial 148522072134 --dry-run

  # 真机 MoveL（确认工作空间后）
  python scripts/hardware/test_moveL_target_hover.py --serial 148522072134 --execute

  q 退出   s 单次采样   m 执行一次 MoveL（需 --execute）
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import pyrealsense2 as rs
import tyro

from xrobotoolkit_teleop.hardware.interface.universal_robots import (
    DEFAULT_MOVEL_ACCELERATION,
    DEFAULT_MOVEL_SPEED,
    DEFAULT_ROBOT_IP,
    URController,
)
from xrobotoolkit_teleop.shared_autonomy.alignment import AlignmentVisionRenderer
from xrobotoolkit_teleop.shared_autonomy.camera_extrinsic import CameraExtrinsic, extrinsic_path_for_serial
from xrobotoolkit_teleop.shared_autonomy.config import COLOR_HSV_PRESETS, TargetDetectionConfig
from xrobotoolkit_teleop.shared_autonomy.depth_sample import sample_depth_at_pixel
from xrobotoolkit_teleop.shared_autonomy.gripper_color_config import GripperColorConfig
from xrobotoolkit_teleop.shared_autonomy.hover_pose import (
    build_hover_tcp_pose,
    target_base_xyz_from_samples,
)

DEFAULT_AUX_SERIAL = "148522072134"


def _collect_samples(
    pipeline: rs.pipeline,
    align: rs.align,
    renderer: AlignmentVisionRenderer,
    *,
    frames: int,
    patch_radius: int,
) -> list:
    buf = []
    for _ in range(frames):
        rs_frames = pipeline.wait_for_frames()
        aligned = align.process(rs_frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue
        color = np.asanyarray(color_frame.get_data())
        frame = renderer.process(color)
        if not frame.target.found or frame.target.center_u is None or frame.target.center_v is None:
            continue
        buf.append(
            sample_depth_at_pixel(
                depth_frame,
                color_frame,
                frame.target.center_u,
                frame.target.center_v,
                radius_px=patch_radius,
            )
        )
    return buf


def _print_plan(
    *,
    target_base: np.ndarray,
    hover_pose: np.ndarray,
    current_tcp: np.ndarray,
    approach_height_m: float,
    dry_run: bool,
) -> None:
    horiz_mm = np.linalg.norm(target_base[:2] - current_tcp[:2]) * 1000.0
    print("\n--- MoveL hover 计划 ---")
    print(f"  目标基座 xyz (m): {target_base}")
    print(f"  当前 TCP xyz (m): {current_tcp[:3]}")
    print(f"  水平偏差:         {horiz_mm:.1f} mm")
    print(f"  approach_height:  {approach_height_m * 1000:.0f} mm")
    print(f"  hover TCP (m):    {hover_pose}")
    print(f"  模式:             {'DRY-RUN' if dry_run else 'EXECUTE'}")


def _safety_check(
    target_base: np.ndarray,
    hover_pose: np.ndarray,
    current_tcp: np.ndarray,
    *,
    max_horiz_m: float,
    min_z_m: float,
    max_z_m: float,
) -> None:
    horiz = float(np.linalg.norm(target_base[:2] - current_tcp[:2]))
    if horiz > max_horiz_m:
        raise RuntimeError(f"水平位移 {horiz * 1000:.0f} mm 超过限制 {max_horiz_m * 1000:.0f} mm")
    if hover_pose[2] < min_z_m or hover_pose[2] > max_z_m:
        raise RuntimeError(f"hover Z={hover_pose[2]:.3f} m 超出 [{min_z_m}, {max_z_m}]")


def main(
    serial: str = DEFAULT_AUX_SERIAL,
    robot_ip: str = DEFAULT_ROBOT_IP,
    extrinsic_path: Path | None = None,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    patch_radius: int = 15,
    sample_frames: int = 15,
    approach_height_m: float = 0.10,
    move_speed: float = DEFAULT_MOVEL_SPEED,
    move_acceleration: float = DEFAULT_MOVEL_ACCELERATION,
    max_horiz_m: float = 0.35,
    min_z_m: float = 0.05,
    max_z_m: float = 0.80,
    color_preset: Literal["target", "silver", "white", "black"] = "target",
    dry_run: bool = True,
    execute: bool = False,
) -> None:
    if execute:
        dry_run = False

    ext_path = extrinsic_path or extrinsic_path_for_serial(serial)
    if not ext_path.is_file():
        print(f"外参文件不存在: {ext_path}", file=sys.stderr)
        print("请先运行: python scripts/hardware/calibrate_fixed_camera.py", file=sys.stderr)
        raise SystemExit(1)
    extrinsic = CameraExtrinsic.load(ext_path)

    devices = [
        (dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
        for dev in rs.context().query_devices()
    ]
    if serial not in {s for _, s in devices}:
        print(f"serial {serial} 不在已连接设备列表中。", file=sys.stderr)
        raise SystemExit(1)

    preset_lower, preset_upper = COLOR_HSV_PRESETS[color_preset]
    renderer = AlignmentVisionRenderer(
        TargetDetectionConfig(hsv_lower=preset_lower, hsv_upper=preset_upper),
        GripperColorConfig(),
    )

    print(f"正在启动相机 {serial} ...")
    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    pipeline.start(rs_config)
    align = rs.align(rs.stream.color)
    print("相机已就绪。")

    print(f"正在连接机械臂 RTDE {robot_ip}（不可达时会阻塞数十秒）...")
    robot = URController(robot_ip, np.zeros(6), use_gripper=False)
    print("机械臂已连接。")
    window = f"moveL_hover [{serial}]"
    print(f"外参: {ext_path}")
    print(f"dry_run={dry_run}  approach={approach_height_m * 1000:.0f}mm  q=退出 s=采样 m=MoveL")

    try:
        while True:
            rs_frames = pipeline.wait_for_frames()
            aligned = align.process(rs_frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data())
            frame = renderer.process(color)
            display = frame.overlay
            if frame.target.found and frame.target.center_u is not None:
                cv2.circle(display, (frame.target.center_u, frame.target.center_v), 6, (255, 0, 255), 2)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key not in (ord("s"), ord("m")):
                continue

            samples = _collect_samples(
                pipeline, align, renderer, frames=sample_frames, patch_radius=patch_radius
            )
            target_base = target_base_xyz_from_samples(samples, extrinsic)
            if target_base is None:
                print("目标/深度无效，跳过。")
                continue

            current_tcp = robot.get_current_tcp_pose()
            hover_pose = build_hover_tcp_pose(target_base, current_tcp, approach_height_m)
            _print_plan(
                target_base=target_base,
                hover_pose=hover_pose,
                current_tcp=current_tcp,
                approach_height_m=approach_height_m,
                dry_run=dry_run,
            )

            if key == ord("m") and not dry_run:
                _safety_check(
                    target_base,
                    hover_pose,
                    current_tcp,
                    max_horiz_m=max_horiz_m,
                    min_z_m=min_z_m,
                    max_z_m=max_z_m,
                )
                print("执行 moveL...")
                ok = robot.move_l(hover_pose, velocity=move_speed, acceleration=move_acceleration)
                if not ok:
                    print("moveL 被拒绝（检查 Remote 模式/保护停）", file=sys.stderr)
                else:
                    time.sleep(0.5)
                    after = robot.get_current_tcp_pose()
                    err_mm = np.linalg.norm(after[:3] - hover_pose[:3]) * 1000.0
                    print(f"到位 TCP xyz: {after[:3]}  位置误差: {err_mm:.1f} mm")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        robot.close()


if __name__ == "__main__":
    tyro.cli(main)
