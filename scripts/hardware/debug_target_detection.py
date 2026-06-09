#!/usr/bin/env python3
"""调试桌面目标检测：在辅助 D435 画面上显示蓝色框与目标中心。

用于 Shared Autonomy 第一步——调 HSV、面积、形态学参数，确认能稳定框出盒子。

典型用法（白盒+银色图案+白墙，默认 silver + ROI）：
  python scripts/hardware/debug_target_detection.py --serial 148522072134 --show-mask

  # 误检右侧机械臂：缩小 ROI 右边界
  python scripts/hardware/debug_target_detection.py \\
    --serial 148522072134 --show-mask --roi-x-max 0.72

  # 银色图案偏亮/偏暗：微调 V 范围
  python scripts/hardware/debug_target_detection.py \\
    --serial 148522072134 --show-mask --hsv-lower 0 0 45 --hsv-upper 180 50 165

窗口操作：
  q — 退出
  s — 打印当前帧检测结果到终端

Requires: pyrealsense2, opencv-python, numpy, tyro
"""

from __future__ import annotations

import sys
from typing import Literal

import cv2
import numpy as np
import pyrealsense2 as rs
import tyro

from xrobotoolkit_teleop.shared_autonomy.config import COLOR_HSV_PRESETS, TargetDetectionConfig
from xrobotoolkit_teleop.shared_autonomy.target_detection import TargetSmoother, detect_and_draw

# 与 teleop_single_ur5e_vr_stream.py 默认辅助相机一致
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
    color_preset: Literal["silver", "white", "black"] = "silver",
    hsv_lower: tuple[int, int, int] | None = None,
    hsv_upper: tuple[int, int, int] | None = None,
    roi_y_min: float = 0.46,
    roi_y_max: float = 0.88,
    roi_x_min: float = 0.22,
    roi_x_max: float = 0.58,
    min_contour_area: float = 800.0,
    max_contour_area: float = 45_000.0,
    max_contour_roi_ratio: float = 0.28,
    morph_kernel_size: int = 5,
    morph_open_iterations: int = 1,
    morph_close_iterations: int = 2,
    enable_smooth: bool = True,
    center_ema_alpha: float = 0.35,
    bbox_ema_alpha: float = 0.30,
    max_hold_frames: int = 6,
) -> None:
    devices = _list_devices()
    if not devices:
        print("未找到 RealSense 相机。", file=sys.stderr)
        raise SystemExit(1)

    for name, sn in devices:
        print(f"  {name}  (serial: {sn})")

    if serial not in {s for _, s in devices}:
        print(f"serial {serial} 不在已连接设备列表中。", file=sys.stderr)
        raise SystemExit(1)

    preset_lower, preset_upper = COLOR_HSV_PRESETS[color_preset]
    use_lower = preset_lower if hsv_lower is None else hsv_lower
    use_upper = preset_upper if hsv_upper is None else hsv_upper

    config = TargetDetectionConfig(
        hsv_lower=use_lower,
        hsv_upper=use_upper,
        roi_y_min_ratio=roi_y_min,
        roi_y_max_ratio=roi_y_max,
        roi_x_min_ratio=roi_x_min,
        roi_x_max_ratio=roi_x_max,
        min_contour_area=min_contour_area,
        max_contour_area=max_contour_area,
        max_contour_roi_area_ratio=max_contour_roi_ratio,
        morph_kernel_size=morph_kernel_size,
        morph_open_iterations=morph_open_iterations,
        morph_close_iterations=morph_close_iterations,
        enable_temporal_smooth=enable_smooth,
        center_ema_alpha=center_ema_alpha,
        bbox_ema_alpha=bbox_ema_alpha,
        max_hold_frames=max_hold_frames,
    )
    smoother = TargetSmoother(config)

    print(f"颜色预设: {color_preset}  HSV lower={use_lower}  upper={use_upper}")
    if enable_smooth:
        print(f"时间平滑: center_ema={center_ema_alpha}  bbox_ema={bbox_ema_alpha}  hold={max_hold_frames} frames")
    print(f"ROI: y=[{roi_y_min}, {roi_y_max}]  x=[{roi_x_min}, {roi_x_max}]  (黄框=搜索区)")
    if color_preset == "white":
        print("提示: 白墙/白桌场景易误检，建议 --color-preset silver")

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    print(f"启动辅助相机 {serial} ({width}x{height} @ {fps}fps)")
    print("按 q 退出，按 s 打印当前检测结果")
    pipeline.start(rs_config)
    window = f"target_detection [{serial}]"

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color = np.asanyarray(frames.get_color_frame().get_data())
            overlay, result = detect_and_draw(color, config, return_mask=show_mask, smoother=smoother)

            if show_mask and result.mask is not None:
                mask_bgr = cv2.cvtColor(result.mask, cv2.COLOR_GRAY2BGR)
                display = np.hstack((overlay, mask_bgr))
            else:
                display = overlay

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if result.found:
                    print(
                        f"[检测] 中心=({result.center_u}, {result.center_v})  "
                        f"bbox={result.bbox}  area={result.contour_area:.0f}"
                    )
                else:
                    print(
                        f"[检测] 未找到目标  mask={result.debug_mask_ratio * 100:.1f}%  "
                        f"contours={result.debug_contour_count}  "
                        f"roi_largest={result.debug_largest_area_in_roi:.0f}  "
                        f"cands={result.debug_candidates_in_roi}  "
                        f"面积∈[{min_contour_area}, {max_contour_area}]"
                    )
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    tyro.cli(main)
