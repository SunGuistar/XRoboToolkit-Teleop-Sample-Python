#!/usr/bin/env python3
"""固定辅机单点外参标定（相机 3D -> 基座 3D）。

流程：
  1. 用示教器把 TCP 移到目标中心正上方（或轻触目标中心）。
  2. 终端按 Enter，脚本采集多帧目标深度并保存外参 JSON。

  python scripts/hardware/calibrate_fixed_camera.py --serial 148522072134

标定假设：当前 TCP 位置即为目标点在基座系下的 3D 坐标（R 默认单位阵，可用 --rpy-deg 微调）。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pyrealsense2 as rs
import tyro

from xrobotoolkit_teleop.hardware.interface.universal_robots import DEFAULT_ROBOT_IP, URController
from xrobotoolkit_teleop.shared_autonomy.alignment import AlignmentVisionRenderer
from xrobotoolkit_teleop.shared_autonomy.camera_extrinsic import CameraExtrinsic, extrinsic_path_for_serial
from xrobotoolkit_teleop.shared_autonomy.config import COLOR_HSV_PRESETS, TargetDetectionConfig
from xrobotoolkit_teleop.shared_autonomy.depth_sample import sample_depth_at_pixel
from xrobotoolkit_teleop.shared_autonomy.gripper_color_config import GripperColorConfig
from xrobotoolkit_teleop.shared_autonomy.hover_pose import median_cam_xyz

DEFAULT_AUX_SERIAL = "148522072134"


def _collect_target_cam_samples(
    pipeline: rs.pipeline,
    align: rs.align,
    renderer: AlignmentVisionRenderer,
    *,
    frames: int,
    patch_radius: int,
) -> list:
    samples = []
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
        samples.append(
            sample_depth_at_pixel(
                depth_frame,
                color_frame,
                frame.target.center_u,
                frame.target.center_v,
                radius_px=patch_radius,
            )
        )
    return samples


def main(
    serial: str = DEFAULT_AUX_SERIAL,
    robot_ip: str = DEFAULT_ROBOT_IP,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    patch_radius: int = 15,
    sample_frames: int = 30,
    color_preset: Literal["target", "silver", "white", "black"] = "target",
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    output: Path | None = None,
) -> None:
    devices = [
        (dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
        for dev in rs.context().query_devices()
    ]
    if not devices:
        print("未找到 RealSense 相机。", file=sys.stderr)
        raise SystemExit(1)
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

    print(f"正在连接机械臂 RTDE {robot_ip}（不可达时会阻塞数十秒，可 Ctrl+C 中断）...")
    robot = URController(robot_ip, np.zeros(6), use_gripper=False)
    print("机械臂已连接。")
    try:
        print("=== 固定相机单点外参标定 ===")
        print("1) 用示教器把 TCP 对准目标中心（正上方或轻触盒面）")
        print("2) 确认辅机画面里蓝框稳定后，回到此终端按 Enter")
        input("就绪后按 Enter 采集... ")

        tcp_pose = robot.get_current_tcp_pose()
        tcp_xyz = tcp_pose[:3]
        print(f"TCP xyz (m): {tcp_xyz}")

        print(f"采集 {sample_frames} 帧目标深度...")
        samples = _collect_target_cam_samples(
            pipeline, align, renderer, frames=sample_frames, patch_radius=patch_radius
        )
        p_cam = median_cam_xyz(samples)
        valid_n = sum(1 for s in samples if s.valid)
        if p_cam is None:
            print(f"有效深度帧 {valid_n}/{len(samples)}，标定失败。请检查目标检测与光照。", file=sys.stderr)
            raise SystemExit(1)

        print(f"目标 cam_xyz (m): {p_cam}  (valid {valid_n}/{len(samples)} frames)")
        extrinsic = CameraExtrinsic.from_one_point(
            camera_serial=serial,
            tcp_xyz_m=tcp_xyz,
            target_cam_xyz_m=p_cam,
            rpy_deg=rpy_deg,
            note="one-point calibration: TCP at target center",
        )
        out_path = extrinsic.save(output)
        p_base = extrinsic.camera_to_base(p_cam)
        err_mm = np.linalg.norm(p_base - tcp_xyz) * 1000.0
        print(f"\n已保存: {out_path}")
        print(f"  translation_m: {extrinsic.translation_m}")
        print(f"  rpy_deg:       {extrinsic.rpy_deg}")
        print(f"  回代误差 |P_base - TCP|: {err_mm:.1f} mm")
        if err_mm > 30.0:
            print("  => 误差偏大，可重标定或微调 --rpy-deg")
        else:
            print("  => 可继续 test_moveL_target_hover.py --dry-run 验证")
    finally:
        pipeline.stop()
        robot.close()


if __name__ == "__main__":
    tyro.cli(main)
