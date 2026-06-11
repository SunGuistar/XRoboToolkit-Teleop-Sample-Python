#!/usr/bin/env python3
"""离线验证 D435 深度：在目标蓝框中心及邻域采样深度（不动机械臂）。

  python scripts/hardware/debug_target_depth.py --serial 148522072134

  # 更大采样邻域、打印统计
  python scripts/hardware/debug_target_depth.py --serial 148522072134 --patch-radius 15

窗口：
  q  退出
  s  打印当前帧深度统计
  r  连续记录 30 帧稳定性报告（看中位数/标准差是否稳）

显示：左=检测叠加  右=对齐后的深度伪彩 + 采样圆
"""

from __future__ import annotations

import sys
import time
from typing import Literal

import cv2
import numpy as np
import pyrealsense2 as rs
import tyro

from xrobotoolkit_teleop.shared_autonomy.alignment import AlignmentVisionRenderer
from xrobotoolkit_teleop.shared_autonomy.config import COLOR_HSV_PRESETS, TargetDetectionConfig
from xrobotoolkit_teleop.shared_autonomy.depth_sample import DepthPatchSample, sample_depth_at_pixel
from xrobotoolkit_teleop.shared_autonomy.gripper_color_config import GripperColorConfig

DEFAULT_AUX_SERIAL = "148522072134"


def _list_devices() -> list[tuple[str, str]]:
    return [
        (dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
        for dev in rs.context().query_devices()
    ]


def _draw_depth_panel(
    depth_frame: rs.depth_frame,
    sample: DepthPatchSample | None,
    *,
    alpha: float = 0.03,
) -> np.ndarray:
    depth_u16 = np.asanyarray(depth_frame.get_data())
    vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_u16, alpha=alpha), cv2.COLORMAP_JET)
    if sample is not None:
        cv2.circle(vis, (sample.center_u, sample.center_v), sample.radius_px, (255, 255, 255), 2)
        cv2.drawMarker(vis, (sample.center_u, sample.center_v), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
        if sample.valid:
            label = f"med {sample.median_m * 1000:.0f}mm  std {sample.std_m * 1000:.1f}mm"
            color = (255, 255, 255)
        else:
            label = f"INVALID valid={sample.valid_ratio * 100:.0f}%"
            color = (0, 0, 255)
        cv2.putText(vis, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return vis


def _draw_depth_on_overlay(overlay: np.ndarray, sample: DepthPatchSample | None) -> np.ndarray:
    out = overlay.copy()
    if sample is None:
        return out
    cv2.circle(out, (sample.center_u, sample.center_v), sample.radius_px, (255, 0, 255), 2)
    if sample.valid:
        text = (
            f"Z_med={sample.median_m * 1000:.0f}mm  "
            f"camZ={sample.camera_z * 1000:.0f}mm  valid={sample.valid_ratio * 100:.0f}%"
        )
        color = (255, 0, 255)
    else:
        text = f"depth INVALID  valid={sample.valid_ratio * 100:.0f}%"
        color = (0, 0, 255)
    cv2.putText(out, text, (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return out


def _stability_report(samples: list[DepthPatchSample]) -> None:
    valid = [s for s in samples if s.valid]
    print(f"\n=== 稳定性记录 n={len(samples)}  valid={len(valid)} ===")
    if not valid:
        print("无有效深度帧，请调整目标检测或光照/距离。")
        return
    meds = np.array([s.median_m for s in valid]) * 1000.0
    cam_z = np.array([s.camera_z for s in valid]) * 1000.0
    stds = np.array([s.std_m for s in valid]) * 1000.0
    print(f"  median depth (patch): {meds.mean():.1f} +/- {meds.std():.1f} mm  (frame jitter)")
    print(f"  camera Z:             {cam_z.mean():.1f} +/- {cam_z.std():.1f} mm")
    print(f"  patch std (in-frame): {stds.mean():.1f} mm avg")
    if meds.std() < 5.0:
        print("  => 帧间稳定，深度可用于 MoveL 粗定位")
    elif meds.std() < 15.0:
        print("  => 中等抖动，建议多帧中值或更大 patch-radius")
    else:
        print("  => 不稳定，需检查反光/遮挡/检测中心是否落在盒面")


def main(
    serial: str = DEFAULT_AUX_SERIAL,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    patch_radius: int = 10,
    record_frames: int = 30,
    color_preset: Literal["target", "silver", "white", "black"] = "target",
    depth_alpha: float = 0.03,
) -> None:
    devices = _list_devices()
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

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    profile = pipeline.start(rs_config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)
    print(f"相机 {serial}  color={width}x{height}@{fps}  depth_scale={depth_scale:.6f} m/unit")
    print(f"目标 preset={color_preset}  patch_radius={patch_radius}px")
    print("q=退出  s=打印深度  r=记录30帧稳定性")

    window = f"target_depth [{serial}]"
    last_sample: DepthPatchSample | None = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            frame = renderer.process(color)

            sample: DepthPatchSample | None = None
            if frame.target.found and frame.target.center_u is not None and frame.target.center_v is not None:
                sample = sample_depth_at_pixel(
                    depth_frame,
                    color_frame,
                    frame.target.center_u,
                    frame.target.center_v,
                    radius_px=patch_radius,
                )
                last_sample = sample

            overlay = _draw_depth_on_overlay(frame.overlay, sample)
            depth_panel = _draw_depth_panel(depth_frame, sample, alpha=depth_alpha)
            display = np.hstack((overlay, depth_panel))
            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if last_sample is None:
                    print("[深度] 目标未检出")
                else:
                    print(f"[深度] {last_sample.format_log()}")
            if key == ord("r"):
                print(f"记录 {record_frames} 帧...")
                buf: list[DepthPatchSample] = []
                t0 = time.monotonic()
                for _ in range(record_frames):
                    frames = pipeline.wait_for_frames()
                    aligned = align.process(frames)
                    cf = aligned.get_color_frame()
                    df = aligned.get_depth_frame()
                    if not cf or not df:
                        continue
                    c = np.asanyarray(cf.get_data())
                    fr = renderer.process(c)
                    if fr.target.found and fr.target.center_u is not None and fr.target.center_v is not None:
                        buf.append(
                            sample_depth_at_pixel(
                                df, cf, fr.target.center_u, fr.target.center_v, radius_px=patch_radius
                            )
                        )
                _stability_report(buf)
                print(f"耗时 {time.monotonic() - t0:.1f}s\n")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main)
