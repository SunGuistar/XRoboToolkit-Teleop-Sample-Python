"""桌面目标检测与画面叠加（第一版：颜色阈值 + ROI + 轮廓评分）。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from xrobotoolkit_teleop.shared_autonomy.config import TargetDetectionConfig


@dataclass
class TargetDetectionResult:
    """单次检测输出。"""

    found: bool
    center_u: int | None = None
    center_v: int | None = None
    bbox: tuple[int, int, int, int] | None = None  # x, y, w, h
    contour_area: float = 0.0
    mask: np.ndarray | None = None
    debug_mask_ratio: float = 0.0
    debug_contour_count: int = 0
    debug_largest_area: float = 0.0
    debug_largest_area_in_roi: float = 0.0
    debug_candidates_in_roi: int = 0


def _roi_pixel_bounds(frame_w: int, frame_h: int, config: TargetDetectionConfig) -> tuple[int, int, int, int]:
    """返回 ROI 像素范围 (x0, y0, x1, y1)，x1/y1 为开区间上界。"""
    x0 = int(config.roi_x_min_ratio * frame_w)
    x1 = int(config.roi_x_max_ratio * frame_w)
    y0 = int(config.roi_y_min_ratio * frame_h)
    y1 = int(config.roi_y_max_ratio * frame_h)
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def _apply_roi_to_mask(mask: np.ndarray, config: TargetDetectionConfig) -> np.ndarray:
    """把 ROI 外像素清零，避免墙面/机械臂与桌面连成超大轮廓。"""
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = _roi_pixel_bounds(w, h, config)
    cropped = np.zeros_like(mask)
    cropped[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return cropped


def _build_binary_mask(frame_bgr: np.ndarray, config: TargetDetectionConfig) -> np.ndarray:
    """BGR → HSV → 阈值 → 形态学 → ROI 裁剪，得到用于找轮廓的 mask。"""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(config.hsv_lower, dtype=np.uint8)
    upper = np.array(config.hsv_upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    k = max(1, config.morph_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    if config.morph_open_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iterations)
    if config.morph_close_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.morph_close_iterations)
    return _apply_roi_to_mask(mask, config)


def _contour_centroid(cnt: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(cnt)
    if moments["m00"] > 1e-6:
        return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]
    x, y, w, h = cv2.boundingRect(cnt)
    return x + w / 2.0, y + h / 2.0


def _centroid_in_roi(cu: float, cv: float, frame_w: int, frame_h: int, config: TargetDetectionConfig) -> bool:
    return (
        config.roi_x_min_ratio * frame_w <= cu <= config.roi_x_max_ratio * frame_w
        and config.roi_y_min_ratio * frame_h <= cv <= config.roi_y_max_ratio * frame_h
    )


def _effective_max_contour_area(frame_w: int, frame_h: int, config: TargetDetectionConfig) -> float:
    """综合绝对上限与 ROI 占比上限，防止检出整块搜索区。"""
    x0, y0, x1, y1 = _roi_pixel_bounds(frame_w, frame_h, config)
    roi_area = float((x1 - x0) * (y1 - y0))
    ratio_cap = config.max_contour_roi_area_ratio * roi_area
    return min(config.max_contour_area, ratio_cap)


def _contour_is_valid_size(area: float, frame_w: int, frame_h: int, config: TargetDetectionConfig) -> bool:
    return config.min_contour_area <= area <= _effective_max_contour_area(frame_w, frame_h, config)


def _select_best_contour(
    contours: list[np.ndarray],
    frame_w: int,
    frame_h: int,
    config: TargetDetectionConfig,
) -> tuple[np.ndarray | None, float]:
    """在 ROI 内选轮廓：偏小且靠近偏好中心（score = -area_penalty*area - dist_penalty*dist）。"""
    pref_u = config.prefer_center_u_ratio * frame_w
    pref_v = config.prefer_center_v_ratio * frame_h

    best: np.ndarray | None = None
    best_area = 0.0
    best_score = -math.inf

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not _contour_is_valid_size(area, frame_w, frame_h, config):
            continue

        cu, cv = _contour_centroid(cnt)
        if not _centroid_in_roi(cu, cv, frame_w, frame_h, config):
            continue

        dist = math.hypot(cu - pref_u, cv - pref_v)
        score = -config.area_penalty * area - config.center_distance_penalty * dist
        if score > best_score:
            best_score = score
            best = cnt
            best_area = area

    return best, best_area


def _collect_debug_stats(
    contours: list[np.ndarray],
    frame_w: int,
    frame_h: int,
    config: TargetDetectionConfig,
) -> tuple[float, float, int]:
    largest_area = 0.0
    largest_in_roi = 0.0
    candidates_in_roi = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        largest_area = max(largest_area, area)
        cu, cv = _contour_centroid(cnt)
        if not _centroid_in_roi(cu, cv, frame_w, frame_h, config):
            continue
        largest_in_roi = max(largest_in_roi, area)
        if _contour_is_valid_size(area, frame_w, frame_h, config):
            candidates_in_roi += 1

    return largest_area, largest_in_roi, candidates_in_roi


def detect_target(
    frame_bgr: np.ndarray,
    config: TargetDetectionConfig,
    *,
    return_mask: bool = False,
) -> TargetDetectionResult:
    """在单帧图像中检测盒子中心（默认银色图案 + ROI）。"""
    if frame_bgr is None or frame_bgr.size == 0:
        return TargetDetectionResult(found=False, mask=None if not return_mask else np.array([]))

    frame_h, frame_w = frame_bgr.shape[:2]
    mask = _build_binary_mask(frame_bgr, config)
    mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_area, largest_in_roi, candidates_in_roi = _collect_debug_stats(contours, frame_w, frame_h, config)
    best, best_area = _select_best_contour(contours, frame_w, frame_h, config)

    debug_kwargs = dict(
        debug_mask_ratio=mask_ratio,
        debug_contour_count=len(contours),
        debug_largest_area=largest_area,
        debug_largest_area_in_roi=largest_in_roi,
        debug_candidates_in_roi=candidates_in_roi,
    )

    if best is None:
        return TargetDetectionResult(found=False, mask=mask if return_mask else None, **debug_kwargs)

    x, y, w, h = cv2.boundingRect(best)
    cu, cv = _contour_centroid(best)

    return TargetDetectionResult(
        found=True,
        center_u=int(cu),
        center_v=int(cv),
        bbox=(x, y, w, h),
        contour_area=best_area,
        mask=mask if return_mask else None,
        **debug_kwargs,
    )


def _draw_roi_guide(frame_bgr: np.ndarray, config: TargetDetectionConfig) -> None:
    """绘制 ROI 黄框（搜索范围，不是检测结果）。"""
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _roi_pixel_bounds(w, h, config)
    cv2.rectangle(frame_bgr, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        frame_bgr,
        "ROI search",
        (x0 + 4, max(y0 + 16, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_target_overlay(
    frame_bgr: np.ndarray,
    result: TargetDetectionResult,
    config: TargetDetectionConfig,
) -> np.ndarray:
    """在画面上绘制蓝色目标框、中心圆点及像素坐标。"""
    out = frame_bgr.copy()
    if config.draw_roi_guide:
        _draw_roi_guide(out, config)

    if not result.found or result.bbox is None:
        lines = [
            config.lost_text,
            f"mask={result.debug_mask_ratio * 100:.1f}%  contours={result.debug_contour_count}",
            f"roi_largest={result.debug_largest_area_in_roi:.0f}  "
            f"cands={result.debug_candidates_in_roi}",
            f"need area [{config.min_contour_area:.0f}, "
            f"{_effective_max_contour_area(frame_bgr.shape[1], frame_bgr.shape[0], config):.0f}]",
        ]
        if result.debug_mask_ratio < 0.003:
            lines.append("hint: widen V range (silver: --hsv-lower 0 0 40)")
        elif result.debug_largest_area_in_roi < config.min_contour_area:
            lines.append("hint: lower --min-contour-area or widen ROI")
        elif result.debug_candidates_in_roi == 0 and result.debug_largest_area_in_roi > 0:
            lines.append("hint: blob too large — tighten HSV or lower --max-contour-area")
        elif result.debug_largest_area_in_roi == 0:
            lines.append("hint: adjust ROI (--roi-y-min) or color preset")
        y0 = 24
        for i, line in enumerate(lines):
            cv2.putText(
                out,
                line,
                (12, y0 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.text_scale,
                config.text_color_bgr,
                config.text_thickness,
                cv2.LINE_AA,
            )
        return out

    x, y, w, h = result.bbox
    cv2.rectangle(out, (x, y), (x + w - 1, y + h - 1), config.box_color_bgr, config.box_thickness)

    if result.center_u is not None and result.center_v is not None:
        cv2.circle(
            out,
            (result.center_u, result.center_v),
            config.center_radius,
            config.center_color_bgr,
            -1,
            lineType=cv2.LINE_AA,
        )
        label = f"target ({result.center_u}, {result.center_v})"
        cv2.putText(
            out,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.text_scale,
            config.text_color_bgr,
            config.text_thickness,
            cv2.LINE_AA,
        )

    return out


class TargetSmoother:
    """对目标中心与包围框做指数滑动平均，抑制颜色阈值带来的逐帧抖动。"""

    def __init__(self, config: TargetDetectionConfig):
        self._config = config
        self._ready = False
        self._cu = self._cv = 0.0
        self._bx = self._by = self._bw = self._bh = 0.0
        self._lost_streak = 0

    def reset(self) -> None:
        self._ready = False
        self._lost_streak = 0

    def _pack(self, template: TargetDetectionResult) -> TargetDetectionResult:
        label_area = template.contour_area
        return TargetDetectionResult(
            found=True,
            center_u=int(round(self._cu)),
            center_v=int(round(self._cv)),
            bbox=(
                int(round(self._bx)),
                int(round(self._by)),
                int(round(self._bw)),
                int(round(self._bh)),
            ),
            contour_area=label_area,
            mask=template.mask,
            debug_mask_ratio=template.debug_mask_ratio,
            debug_contour_count=template.debug_contour_count,
            debug_largest_area=template.debug_largest_area,
            debug_largest_area_in_roi=template.debug_largest_area_in_roi,
            debug_candidates_in_roi=template.debug_candidates_in_roi,
        )

    def smooth(self, raw: TargetDetectionResult) -> TargetDetectionResult:
        if not self._config.enable_temporal_smooth:
            return raw

        if raw.found and raw.bbox is not None and raw.center_u is not None and raw.center_v is not None:
            x, y, w, h = raw.bbox
            if raw.contour_area > 0 and raw.contour_area > self._config.max_contour_area:
                self._lost_streak += 1
                if self._ready and self._lost_streak <= self._config.max_hold_frames:
                    return self._pack(raw)
                self._ready = False
                return raw
            self._lost_streak = 0
            if not self._ready:
                self._cu, self._cv = float(raw.center_u), float(raw.center_v)
                self._bx, self._by, self._bw, self._bh = float(x), float(y), float(w), float(h)
                self._ready = True
            else:
                ac = self._config.center_ema_alpha
                ab = self._config.bbox_ema_alpha
                self._cu = ac * raw.center_u + (1.0 - ac) * self._cu
                self._cv = ac * raw.center_v + (1.0 - ac) * self._cv
                self._bx = ab * x + (1.0 - ab) * self._bx
                self._by = ab * y + (1.0 - ab) * self._by
                self._bw = ab * w + (1.0 - ab) * self._bw
                self._bh = ab * h + (1.0 - ab) * self._bh
            return self._pack(raw)

        self._lost_streak += 1
        if self._ready and self._lost_streak <= self._config.max_hold_frames:
            return self._pack(raw)
        if self._lost_streak > self._config.max_hold_frames:
            self._ready = False
        return raw


def detect_and_draw(
    frame_bgr: np.ndarray,
    config: TargetDetectionConfig,
    *,
    return_mask: bool = False,
    smoother: TargetSmoother | None = None,
) -> tuple[np.ndarray, TargetDetectionResult]:
    """检测目标并在原图上叠加可视化，供调试脚本和 VR 推流复用。"""
    result = detect_target(frame_bgr, config, return_mask=return_mask)
    if smoother is not None:
        result = smoother.smooth(result)
    overlay = draw_target_overlay(frame_bgr, result, config)
    return overlay, result
