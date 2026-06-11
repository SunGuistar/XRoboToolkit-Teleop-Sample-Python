"""D435 深度采样：在目标像素邻域统计深度并反投影到相机坐标系。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyrealsense2 as rs


@dataclass
class DepthPatchSample:
    valid: bool
    center_u: int
    center_v: int
    radius_px: int
    depth_m: float = 0.0
    median_m: float = 0.0
    mean_m: float = 0.0
    std_m: float = 0.0
    min_m: float = 0.0
    max_m: float = 0.0
    valid_ratio: float = 0.0
    valid_count: int = 0
    total_count: int = 0
    camera_x: float = 0.0
    camera_y: float = 0.0
    camera_z: float = 0.0

    def format_log(self) -> str:
        if not self.valid:
            return (
                f"depth INVALID @ ({self.center_u},{self.center_v})  "
                f"valid_ratio={self.valid_ratio * 100:.1f}%"
            )
        return (
            f"depth @ ({self.center_u},{self.center_v})  "
            f"median={self.median_m * 1000:.1f}mm  mean={self.mean_m * 1000:.1f}mm  "
            f"std={self.std_m * 1000:.1f}mm  range=[{self.min_m * 1000:.0f},{self.max_m * 1000:.0f}]mm  "
            f"valid={self.valid_ratio * 100:.0f}%  "
            f"cam_xyz=({self.camera_x * 1000:.0f},{self.camera_y * 1000:.0f},{self.camera_z * 1000:.0f})mm"
        )


def sample_depth_at_pixel(
    depth_frame: rs.depth_frame,
    color_frame: rs.video_frame,
    center_u: int,
    center_v: int,
    *,
    radius_px: int = 10,
    use_median_for_point: bool = True,
) -> DepthPatchSample:
    """在对齐到 color 的深度图上，以 (u,v) 为中心取邻域统计。"""
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_scale = float(depth_frame.get_units())
    h, w = depth_image.shape[:2]
    cu = int(np.clip(center_u, 0, w - 1))
    cv = int(np.clip(center_v, 0, h - 1))
    r = max(1, radius_px)
    x0, x1 = max(0, cu - r), min(w, cu + r + 1)
    y0, y1 = max(0, cv - r), min(h, cv + r + 1)
    patch = depth_image[y0:y1, x0:x1]
    total = int(patch.size)
    valid_mask = patch > 0
    valid_count = int(np.count_nonzero(valid_mask))
    ratio = float(valid_count) / float(total) if total > 0 else 0.0

    base = DepthPatchSample(
        valid=False,
        center_u=cu,
        center_v=cv,
        radius_px=r,
        valid_ratio=ratio,
        valid_count=valid_count,
        total_count=total,
    )
    if valid_count == 0:
        return base

    depths_m = patch[valid_mask].astype(np.float64) * depth_scale
    center_raw = float(depth_image[cv, cu])
    center_m = center_raw * depth_scale if center_raw > 0 else float(np.median(depths_m))
    rep_m = float(np.median(depths_m)) if use_median_for_point else center_m

    color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    cam_x, cam_y, cam_z = rs.rs2_deproject_pixel_to_point(color_intrinsics, [float(cu), float(cv)], rep_m)

    return DepthPatchSample(
        valid=True,
        center_u=cu,
        center_v=cv,
        radius_px=r,
        depth_m=center_m,
        median_m=float(np.median(depths_m)),
        mean_m=float(np.mean(depths_m)),
        std_m=float(np.std(depths_m)),
        min_m=float(np.min(depths_m)),
        max_m=float(np.max(depths_m)),
        valid_ratio=ratio,
        valid_count=valid_count,
        total_count=total,
        camera_x=float(cam_x),
        camera_y=float(cam_y),
        camera_z=float(cam_z),
    )
