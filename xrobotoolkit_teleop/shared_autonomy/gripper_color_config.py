"""夹爪颜色/轮廓检测参数。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GripperColorConfig:
    """通过深色夹爪（如黑色 DH 夹爪）轮廓估计夹爪中心。

    在画面上方/中央 ROI 内找最大深色块（对准盒子时夹爪通常在中上，不在画面最右侧）。
    默认取外接矩形底边中点为夹爪中心。
    """

    # --- HSV：深色/黑色夹爪 ---
    hsv_lower: tuple[int, int, int] = (0, 0, 0)
    hsv_upper: tuple[int, int, int] = (180, 255, 70)

    # --- 夹爪搜索 ROI（浅黄框；与下方目标黄框错开，覆盖中上区域）---
    roi_y_min_ratio: float = 0.02
    roi_y_max_ratio: float = 0.82
    roi_x_min_ratio: float = 0.28
    roi_x_max_ratio: float = 0.98

    prefer_center_u_ratio: float = 0.55
    prefer_center_v_ratio: float = 0.32
    center_distance_penalty: float = 1.5

    min_contour_area: float = 1200.0
    max_contour_area: float = 90_000.0
    max_contour_roi_area_ratio: float = 0.55

    # True：外接矩形底边中点；False：矩形中心
    use_bbox_bottom_center: bool = True
    gripper_offset_u: float = 0.0
    gripper_offset_v: float = 0.0

    morph_kernel_size: int = 5
    morph_open_iterations: int = 1
    morph_close_iterations: int = 2

    enable_temporal_smooth: bool = True
    center_ema_alpha: float = 0.40
    max_hold_frames: int = 6

    draw_roi_guide: bool = True
    roi_guide_color_bgr: tuple[int, int, int] = (255, 255, 0)  # 青黄，与目标黄框区分

    box_color_bgr: tuple[int, int, int] = (0, 255, 0)
    gripper_center_color_bgr: tuple[int, int, int] = (0, 255, 0)
    box_thickness: int = 2
    gripper_center_radius: int = 7
    text_color_bgr: tuple[int, int, int] = (0, 255, 0)
    text_scale: float = 0.55
    text_thickness: int = 2
    lost_text: str = "GRIPPER NOT FOUND"
