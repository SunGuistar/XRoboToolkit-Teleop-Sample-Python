"""夹爪检测：颜色轮廓 + ROI。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from xrobotoolkit_teleop.shared_autonomy.gripper_color_config import GripperColorConfig


@dataclass
class GripperDetectionResult:
    found: bool
    gripper_u: int | None = None
    gripper_v: int | None = None
    bbox: tuple[int, int, int, int] | None = None
    contour_area: float = 0.0
    mask: np.ndarray | None = None
    debug_mask_ratio: float = 0.0
    debug_contour_count: int = 0
    debug_largest_area_in_roi: float = 0.0


def _roi_bounds(frame_w: int, frame_h: int, x0r: float, y0r: float, x1r: float, y1r: float) -> tuple[int, int, int, int]:
    x0 = int(x0r * frame_w)
    x1 = int(x1r * frame_w)
    y0 = int(y0r * frame_h)
    y1 = int(y1r * frame_h)
    return x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)


def _apply_roi_mask(mask: np.ndarray, frame_w: int, frame_h: int, x0r: float, y0r: float, x1r: float, y1r: float) -> np.ndarray:
    x0, y0, x1, y1 = _roi_bounds(frame_w, frame_h, x0r, y0r, x1r, y1r)
    out = np.zeros_like(mask)
    out[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
    return out


def _contour_centroid(cnt: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(cnt)
    if m["m00"] > 1e-6:
        return m["m10"] / m["m00"], m["m01"] / m["m00"]
    x, y, w, h = cv2.boundingRect(cnt)
    return x + w / 2.0, y + h / 2.0


def _build_color_mask(frame_bgr: np.ndarray, config: GripperColorConfig) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(config.hsv_lower, np.uint8), np.array(config.hsv_upper, np.uint8))
    k = max(1, config.morph_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    if config.morph_open_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.morph_open_iterations)
    if config.morph_close_iterations > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.morph_close_iterations)
    return _apply_roi_mask(
        mask, w, h, config.roi_x_min_ratio, config.roi_y_min_ratio, config.roi_x_max_ratio, config.roi_y_max_ratio
    )


def _effective_max_area(frame_w: int, frame_h: int, config: GripperColorConfig) -> float:
    x0, y0, x1, y1 = _roi_bounds(
        frame_w, frame_h, config.roi_x_min_ratio, config.roi_y_min_ratio, config.roi_x_max_ratio, config.roi_y_max_ratio
    )
    roi_area = float((x1 - x0) * (y1 - y0))
    return min(config.max_contour_area, config.max_contour_roi_area_ratio * roi_area)


def detect_gripper(
    frame_bgr: np.ndarray,
    config: GripperColorConfig,
    *,
    return_mask: bool = False,
) -> GripperDetectionResult:
    """颜色阈值 + 右上 ROI 内最大深色轮廓 → 夹爪中心。"""
    if frame_bgr is None or frame_bgr.size == 0:
        return GripperDetectionResult(found=False)

    h, w = frame_bgr.shape[:2]
    mask = _build_color_mask(frame_bgr, config)
    mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pref_u = config.prefer_center_u_ratio * w
    pref_v = config.prefer_center_v_ratio * h
    max_area = _effective_max_area(w, h, config)
    largest_in_roi = 0.0
    best = None
    best_area = 0.0
    best_score = -math.inf

    for cnt in contours:
        area = cv2.contourArea(cnt)
        cu, cv = _contour_centroid(cnt)
        if not (
            config.roi_x_min_ratio * w <= cu <= config.roi_x_max_ratio * w
            and config.roi_y_min_ratio * h <= cv <= config.roi_y_max_ratio * h
        ):
            continue
        largest_in_roi = max(largest_in_roi, area)
        if area < config.min_contour_area or area > max_area:
            continue
        dist = math.hypot(cu - pref_u, cv - pref_v)
        score = area - config.center_distance_penalty * dist
        if score > best_score:
            best_score = score
            best = cnt
            best_area = area

    debug = dict(
        debug_mask_ratio=mask_ratio,
        debug_contour_count=len(contours),
        debug_largest_area_in_roi=largest_in_roi,
    )

    if best is None:
        return GripperDetectionResult(found=False, mask=mask if return_mask else None, **debug)

    x, y, bw, bh = cv2.boundingRect(best)
    if config.use_bbox_bottom_center:
        ref_u, ref_v = x + bw / 2.0, float(y + bh)
    else:
        ref_u, ref_v = x + bw / 2.0, y + bh / 2.0

    return GripperDetectionResult(
        found=True,
        gripper_u=int(round(ref_u + config.gripper_offset_u)),
        gripper_v=int(round(ref_v + config.gripper_offset_v)),
        bbox=(x, y, bw, bh),
        contour_area=best_area,
        mask=mask if return_mask else None,
        **debug,
    )


class GripperSmoother:
    """夹爪中心 EMA 平滑。"""

    def __init__(self, config: GripperColorConfig):
        self._config = config
        self._ready = False
        self._gu = self._gv = 0.0
        self._lost_streak = 0
        self._last = GripperDetectionResult(found=False)

    def smooth(self, raw: GripperDetectionResult) -> GripperDetectionResult:
        if not self._config.enable_temporal_smooth:
            return raw

        if raw.found and raw.gripper_u is not None and raw.gripper_v is not None:
            self._lost_streak = 0
            if not self._ready:
                self._gu, self._gv = float(raw.gripper_u), float(raw.gripper_v)
                self._ready = True
            else:
                a = self._config.center_ema_alpha
                self._gu = a * raw.gripper_u + (1.0 - a) * self._gu
                self._gv = a * raw.gripper_v + (1.0 - a) * self._gv
            self._last = raw
            return GripperDetectionResult(
                found=True,
                gripper_u=int(round(self._gu)),
                gripper_v=int(round(self._gv)),
                bbox=raw.bbox,
                contour_area=raw.contour_area,
                mask=raw.mask,
                debug_mask_ratio=raw.debug_mask_ratio,
                debug_contour_count=raw.debug_contour_count,
                debug_largest_area_in_roi=raw.debug_largest_area_in_roi,
            )

        self._lost_streak += 1
        if self._ready and self._lost_streak <= self._config.max_hold_frames:
            return GripperDetectionResult(
                found=True,
                gripper_u=int(round(self._gu)),
                gripper_v=int(round(self._gv)),
                bbox=self._last.bbox,
                contour_area=self._last.contour_area,
            )
        if self._lost_streak > self._config.max_hold_frames:
            self._ready = False
        return raw


def _draw_gripper_roi_guide(frame_bgr: np.ndarray, config: GripperColorConfig) -> None:
    h, w = frame_bgr.shape[:2]
    x0, y0, x1, y1 = _roi_bounds(
        w, h, config.roi_x_min_ratio, config.roi_y_min_ratio, config.roi_x_max_ratio, config.roi_y_max_ratio
    )
    cv2.rectangle(frame_bgr, (x0, y0), (x1 - 1, y1 - 1), config.roi_guide_color_bgr, 1, cv2.LINE_AA)
    cv2.putText(
        frame_bgr,
        "GRIPPER ROI",
        (x0 + 4, max(y0 + 16, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        config.roi_guide_color_bgr,
        1,
        cv2.LINE_AA,
    )


def draw_gripper_overlay(
    frame_bgr: np.ndarray,
    result: GripperDetectionResult,
    config: GripperColorConfig,
) -> np.ndarray:
    """绘制夹爪 ROI、绿框与夹爪中心。"""
    out = frame_bgr.copy()
    if config.draw_roi_guide:
        _draw_gripper_roi_guide(out, config)

    if not result.found or result.bbox is None:
        lines = [
            config.lost_text,
            f"grip_mask={result.debug_mask_ratio * 100:.1f}%  roi_largest={result.debug_largest_area_in_roi:.0f}",
        ]
        if result.debug_mask_ratio < 0.003:
            lines.append("hint: raise V in --grip-hsv-upper (e.g. 80)")
        elif result.debug_largest_area_in_roi < config.min_contour_area:
            lines.append("hint: lower --grip-min-area or widen GRIPPER ROI")
        y0 = out.shape[0] - 16 - (len(lines) - 1) * 22
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

    x, y, bw, bh = result.bbox
    cv2.rectangle(out, (x, y), (x + bw - 1, y + bh - 1), config.box_color_bgr, config.box_thickness)
    if result.gripper_u is not None and result.gripper_v is not None:
        cv2.circle(
            out,
            (result.gripper_u, result.gripper_v),
            config.gripper_center_radius,
            config.gripper_center_color_bgr,
            -1,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"gripper ({result.gripper_u}, {result.gripper_v})",
            (max(4, x), max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.text_scale,
            config.text_color_bgr,
            config.text_thickness,
            cv2.LINE_AA,
        )
    return out
