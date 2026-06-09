"""Shared Autonomy 可调参数配置。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ColorPreset = Literal["silver", "white", "black"]

# 颜色预设：白盒贴在白墙/白桌上时，优先用 silver（盒面中间银色图案），避免误检背景。
COLOR_HSV_PRESETS: dict[ColorPreset, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    # 银色图案：中等亮度灰，V 上限要低，否则白墙/浅桌（V 很高）会占满 mask。
    "silver": ((0, 0, 65), (180, 40, 138)),
    # 白色：低饱和度 + 高亮度。易与墙面/桌面混淆，仅作备选。
    "white": ((0, 0, 120), (180, 80, 255)),
    # 黑色笔迹：低亮度。笔划往往碎成多块，仅作备选。
    "black": ((0, 0, 0), (180, 255, 50)),
}


@dataclass
class TargetDetectionConfig:
    """桌面目标（盒子）检测参数。

    第一版使用 HSV 颜色阈值 + ROI + 轮廓评分，不使用深度学习。
    白盒+银色图案+白墙背景：默认 color_preset=silver，并限制 ROI 排除墙面与机械臂。
    实验前请用 debug_target_detection.py --show-mask 微调 HSV、ROI 与面积。
    """

    # --- HSV 颜色阈值（OpenCV: H∈[0,179], S∈[0,255], V∈[0,255]）---
    hsv_lower: tuple[int, int, int] = COLOR_HSV_PRESETS["silver"][0]
    hsv_upper: tuple[int, int, int] = COLOR_HSV_PRESETS["silver"][1]

    # --- 搜索 ROI（归一化 0~1）---
    # 黄框仅表示「只在框内找目标」，不是检测到的蓝框。默认框住桌面中央盒子区域。
    roi_y_min_ratio: float = 0.46
    roi_y_max_ratio: float = 0.88
    roi_x_min_ratio: float = 0.22
    roi_x_max_ratio: float = 0.58

    # --- 轮廓优选（偏小 + 靠近偏好中心；避免选满 ROI 的大块墙面）---
    prefer_center_u_ratio: float = 0.40
    prefer_center_v_ratio: float = 0.72
    center_distance_penalty: float = 3.0
    area_penalty: float = 0.004

    # --- 轮廓面积过滤（单位：像素²）---
    min_contour_area: float = 800.0
    max_contour_area: float = 45_000.0
    # 轮廓面积不得超过 ROI 面积的该比例（防止蓝框与黄框等大）
    max_contour_roi_area_ratio: float = 0.28

    # --- 时间平滑（减轻蓝框逐帧抖动 / found-lost 闪烁）---
    enable_temporal_smooth: bool = True
    # EMA 新观测权重：越小越稳、延迟越大。中心比框尺寸更敏感，可分别调。
    center_ema_alpha: float = 0.35
    bbox_ema_alpha: float = 0.30
    # 短暂丢检时仍显示上一帧平滑结果，避免框闪没
    max_hold_frames: int = 6

    # --- 形态学处理（去噪、填洞）---
    morph_kernel_size: int = 5
    morph_open_iterations: int = 1
    morph_close_iterations: int = 2

    # 调试画面是否绘制 ROI 黄框
    draw_roi_guide: bool = True

    # --- 叠加显示（BGR 颜色；蓝色 = (255, 0, 0)）---
    box_color_bgr: tuple[int, int, int] = (255, 0, 0)
    center_color_bgr: tuple[int, int, int] = (255, 0, 0)
    box_thickness: int = 2
    center_radius: int = 6
    text_color_bgr: tuple[int, int, int] = (255, 0, 0)
    text_scale: float = 0.6
    text_thickness: int = 2

    # 未检测到目标时的提示（须 ASCII：OpenCV 默认字体无法显示中文，会变成蓝色问号）
    lost_text: str = "TARGET NOT FOUND"

    @classmethod
    def from_color_preset(
        cls,
        preset: ColorPreset = "silver",
        **kwargs,
    ) -> "TargetDetectionConfig":
        """按颜色预设生成配置，kwargs 可覆盖任意字段。"""
        lower, upper = COLOR_HSV_PRESETS[preset]
        return cls(hsv_lower=lower, hsv_upper=upper, **kwargs)
