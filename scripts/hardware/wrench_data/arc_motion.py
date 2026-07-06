"""Shared horizontal arc motion around a fixed world-Z axis through a tool_tcp record (x, y)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from xrobotoolkit_teleop.hardware.soft_constraint import rotation_matrix_to_rotvec

WRENCH_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_TOOL_TCP_FILE = WRENCH_DATA_DIR / "tool_tcp.json"
DEFAULT_HOME_FILE = WRENCH_DATA_DIR / "home.json"
DEFAULT_RECORD_NAME = "RECORD_假定的工具TCP"


class OrientationMode(str, Enum):
    KEEP = "keep"
    FOLLOW_ARC = "follow_arc"


@dataclass
class ArcPlan:
    record_name: str
    axis_record_tcp_pose: np.ndarray
    x_axis: float
    y_axis: float
    p0: np.ndarray
    radius_m: float
    angle_deg: float
    pose_mid: np.ndarray
    pose_end: np.ndarray
    orientation_mode: OrientationMode


def rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rv / angle
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c
    return np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )


def matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    return rotation_matrix_to_rotvec(rotation)


def rotation_matrix_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def arc_position_xy(
    theta_rad: float,
    x_axis: float,
    y_axis: float,
    x0: float,
    y0: float,
) -> tuple[float, float]:
    dx = x0 - x_axis
    dy = y0 - y_axis
    x = x_axis + np.cos(theta_rad) * dx - np.sin(theta_rad) * dy
    y = y_axis + np.sin(theta_rad) * dx + np.cos(theta_rad) * dy
    return float(x), float(y)


def orientation_at_theta(rotvec0: np.ndarray, theta_rad: float, mode: OrientationMode) -> np.ndarray:
    if mode == OrientationMode.KEEP:
        return np.asarray(rotvec0, dtype=np.float64).reshape(3).copy()
    r0 = rotvec_to_matrix(rotvec0)
    r_theta = rotation_matrix_z(theta_rad) @ r0
    return matrix_to_rotvec(r_theta)


def build_pose(
    x: float,
    y: float,
    z: float,
    rotvec0: np.ndarray,
    theta_rad: float,
    mode: OrientationMode,
) -> np.ndarray:
    pose = np.zeros(6, dtype=np.float64)
    pose[0] = x
    pose[1] = y
    pose[2] = z
    pose[3:] = orientation_at_theta(rotvec0, theta_rad, mode)
    return pose


def _resolve_record_key(records: dict, record_name: str) -> str:
    if record_name in records:
        return record_name
    upper = record_name.upper()
    if upper in records:
        return upper
    lower = record_name.lower()
    for key in records:
        if key.lower() == lower:
            return key
    available = ", ".join(sorted(records.keys())) or "(empty)"
    raise KeyError(f"Record '{record_name}' not found. Available: {available}")


def load_record_tcp_pose(
    tool_tcp_file: Path = DEFAULT_TOOL_TCP_FILE,
    record_name: str = DEFAULT_RECORD_NAME,
) -> tuple[str, np.ndarray]:
    with tool_tcp_file.open(encoding="utf-8") as handle:
        data = json.load(handle)
    records = data.get("records", {})
    key = _resolve_record_key(records, record_name)
    tcp_pose = np.asarray(records[key]["tcp_pose"], dtype=np.float64).reshape(6)
    return key, tcp_pose


def load_axis_xy(
    tool_tcp_file: Path = DEFAULT_TOOL_TCP_FILE,
    record_name: str = DEFAULT_RECORD_NAME,
) -> tuple[str, np.ndarray, float, float]:
    key, tcp_pose = load_record_tcp_pose(tool_tcp_file, record_name)
    return key, tcp_pose, float(tcp_pose[0]), float(tcp_pose[1])


def compute_arc_plan(
    p0: np.ndarray,
    x_axis: float,
    y_axis: float,
    angle_deg: float,
    orientation_mode: OrientationMode,
    *,
    record_name: str = DEFAULT_RECORD_NAME,
    axis_record_tcp_pose: np.ndarray | None = None,
) -> ArcPlan:
    p0 = np.asarray(p0, dtype=np.float64).reshape(6)
    x0, y0, z0 = float(p0[0]), float(p0[1]), float(p0[2])
    rotvec0 = p0[3:].copy()

    radius_m = float(np.hypot(x0 - x_axis, y0 - y_axis))
    if radius_m < 1e-6:
        raise ValueError(
            f"Current TCP xy [{x0:.6f}, {y0:.6f}] is on rotation axis [{x_axis:.6f}, {y_axis:.6f}]; "
            "cannot define a horizontal arc."
        )

    theta_mid = np.deg2rad(angle_deg / 2.0)
    theta_end = np.deg2rad(angle_deg)

    x_mid, y_mid = arc_position_xy(theta_mid, x_axis, y_axis, x0, y0)
    x_end, y_end = arc_position_xy(theta_end, x_axis, y_axis, x0, y0)

    pose_mid = build_pose(x_mid, y_mid, z0, rotvec0, theta_mid, orientation_mode)
    pose_end = build_pose(x_end, y_end, z0, rotvec0, theta_end, orientation_mode)

    return ArcPlan(
        record_name=record_name,
        axis_record_tcp_pose=np.asarray(axis_record_tcp_pose if axis_record_tcp_pose is not None else p0, dtype=np.float64),
        x_axis=x_axis,
        y_axis=y_axis,
        p0=p0,
        radius_m=radius_m,
        angle_deg=float(angle_deg),
        pose_mid=pose_mid,
        pose_end=pose_end,
        orientation_mode=orientation_mode,
    )


def generate_arc_waypoints(
    plan: ArcPlan,
    steps: int,
) -> list[np.ndarray]:
    if steps < 2:
        return [plan.pose_end.copy()]

    thetas = np.linspace(0.0, np.deg2rad(plan.angle_deg), steps)
    x0, y0, z0 = plan.p0[:3]
    rotvec0 = plan.p0[3:]
    waypoints: list[np.ndarray] = []
    for theta in thetas[1:]:
        x, y = arc_position_xy(theta, plan.x_axis, plan.y_axis, x0, y0)
        waypoints.append(build_pose(x, y, z0, rotvec0, theta, plan.orientation_mode))
    return waypoints


def print_pre_motion_info(plan: ArcPlan) -> None:
    print("=== Arc Motion Plan ===")
    print(f"axis_record ({plan.record_name}) tcp_pose: {plan.axis_record_tcp_pose.tolist()}")
    print(f"current_tcp_pose p0: {plan.p0.tolist()}")
    print(f"rotation_axis_xy: [{plan.x_axis:.6f}, {plan.y_axis:.6f}]")
    print(f"radius_m: {plan.radius_m:.6f}")
    print(f"pose_mid: {plan.pose_mid.tolist()}")
    print(f"pose_end: {plan.pose_end.tolist()}")
    print(f"angle_deg: {plan.angle_deg:.3f}")
    print(f"orientation_mode: {plan.orientation_mode.value}")


@dataclass
class ArcMotionErrors:
    radius_error_m: float
    z_error_m: float
    end_position_error_m: float
    actual_tcp_pose: np.ndarray


def compute_post_motion_errors(
    plan: ArcPlan,
    actual_tcp_pose: np.ndarray,
) -> ArcMotionErrors:
    actual = np.asarray(actual_tcp_pose, dtype=np.float64).reshape(6)
    actual_radius = float(np.hypot(actual[0] - plan.x_axis, actual[1] - plan.y_axis))
    radius_error_m = abs(actual_radius - plan.radius_m)
    z_error_m = abs(float(actual[2]) - float(plan.p0[2]))
    end_position_error_m = float(np.linalg.norm(actual[:3] - plan.pose_end[:3]))
    return ArcMotionErrors(
        radius_error_m=radius_error_m,
        z_error_m=z_error_m,
        end_position_error_m=end_position_error_m,
        actual_tcp_pose=actual,
    )


def print_post_motion_errors(plan: ArcPlan, errors: ArcMotionErrors) -> None:
    print("=== Arc Motion Result ===")
    print(f"actual_tcp_pose: {errors.actual_tcp_pose.tolist()}")
    print(f"horizontal_radius_error_m: {errors.radius_error_m:.6f}")
    print(f"z_error_m: {errors.z_error_m:.6f}")
    print(f"end_position_error_m: {errors.end_position_error_m:.6f}")
