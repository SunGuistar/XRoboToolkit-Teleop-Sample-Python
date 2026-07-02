"""World-frame 3-DOF hard constraint for UR teleoperation (wrench turning).

Reference pose is recorded at the flange on A-press, but constraints use world axes:
  free  — x, y, rz (horizontal translation + yaw)
  locked — z, rx, ry (height + pitch/roll)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# Kept for API compatibility with existing controller kwargs (no longer used).
DEFAULT_KZ = 10.0
DEFAULT_KR = 5.0
DEFAULT_DZ_DAMP = 0.2

WORLD_Z = np.array([0.0, 0.0, 1.0], dtype=np.float64)


@dataclass
class SoftConstraintState:
    active: bool = False
    z_ref: float = 0.0
    rx_ref: float = 0.0
    ry_ref: float = 0.0
    T_ref_flange: np.ndarray | None = None


def rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    """Rotation matrix -> UR axis-angle rotation vector [rx, ry, rz]."""
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-9:
        return np.zeros(3, dtype=np.float64)
    axis = np.array(
        [
            r[2, 1] - r[1, 2],
            r[0, 2] - r[2, 0],
            r[1, 0] - r[0, 1],
        ],
        dtype=np.float64,
    )
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    return axis * (angle / axis_norm)


def rotation_matrix_to_euler_zyx(rotation: np.ndarray) -> tuple[float, float, float]:
    """Rotation matrix -> (yaw, pitch, roll) with R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = float(np.hypot(r[0, 0], r[1, 0]))
    if sy >= 1e-6:
        yaw = float(np.arctan2(r[1, 0], r[0, 0]))
        pitch = float(np.arctan2(-r[2, 0], sy))
        roll = float(np.arctan2(r[2, 1], r[2, 2]))
    else:
        yaw = float(np.arctan2(-r[0, 1], r[1, 1]))
        pitch = float(np.arctan2(-r[2, 0], sy))
        roll = 0.0
    return yaw, pitch, roll


def euler_zyx_to_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """(yaw, pitch, roll) -> rotation matrix, R = Rz @ Ry @ Rx."""
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def transform_to_pose6(transform: np.ndarray) -> np.ndarray:
    """4x4 world transform -> [x, y, z, rx, ry, rz] (m, rad)."""
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    pos = t[:3, 3]
    rotvec = rotation_matrix_to_rotvec(t[:3, :3])
    return np.concatenate([pos, rotvec])


def locked_dof_errors(T_current: np.ndarray, T_ref: np.ndarray) -> tuple[float, float, float]:
    """World-frame errors on locked z, rx, ry (should stay near zero)."""
    pose_curr = transform_to_pose6(T_current)
    pose_ref = transform_to_pose6(T_ref)
    return (
        float(pose_curr[2] - pose_ref[2]),
        float(pose_curr[3] - pose_ref[3]),
        float(pose_curr[4] - pose_ref[4]),
    )


def project_pose_3dof_world(
    T_user_world: np.ndarray,
    T_grip_start_world: np.ndarray,
    T_ref_world: np.ndarray,
) -> np.ndarray:
    """Project full user pose: free x/y/rz; locked z/rx/ry in world frame."""
    t_user = np.asarray(T_user_world, dtype=np.float64).reshape(4, 4)
    t_grip = np.asarray(T_grip_start_world, dtype=np.float64).reshape(4, 4)
    t_ref = np.asarray(T_ref_world, dtype=np.float64).reshape(4, 4)

    dp = t_user[:3, 3] - t_grip[:3, 3]
    dp_horizontal = dp - np.dot(dp, WORLD_Z) * WORLD_Z
    p_cmd = t_ref[:3, 3] + dp_horizontal

    yaw_ref, pitch_ref, roll_ref = rotation_matrix_to_euler_zyx(t_ref[:3, :3])
    yaw_grip, _, _ = rotation_matrix_to_euler_zyx(t_grip[:3, :3])
    yaw_user, _, _ = rotation_matrix_to_euler_zyx(t_user[:3, :3])
    yaw_cmd = yaw_ref + (yaw_user - yaw_grip)
    r_cmd = euler_zyx_to_rotation_matrix(yaw_cmd, pitch_ref, roll_ref)

    t_cmd = np.eye(4, dtype=np.float64)
    t_cmd[:3, :3] = r_cmd
    t_cmd[:3, 3] = p_cmd
    return t_cmd


class SoftConstraintTeleopHelper:
    """3-DOF world constraint: user controls x/y/rz; z/rx/ry locked to A-button reference."""

    def __init__(self, **_kwargs):
        self.state = SoftConstraintState()
        self._prev_a_button = False
        self._prev_b_button = False
        self._T_grip_start: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self.state.active

    def reset_motion_state(self) -> None:
        self._T_grip_start = None

    def set_grip_start(self, T_world: np.ndarray) -> None:
        """Record flange pose when grip activates (baseline for user x/y/rz delta)."""
        self._T_grip_start = np.asarray(T_world, dtype=np.float64).reshape(4, 4).copy()

    def clear_all(self) -> None:
        self.state = SoftConstraintState()
        self.reset_motion_state()
        self._prev_a_button = False
        self._prev_b_button = False

    def update_buttons(
        self,
        *,
        a_pressed: bool,
        b_pressed: bool,
        placo_robot,
        resync_active_grip: Callable[[], None] | None = None,
    ) -> None:
        if a_pressed and not self._prev_a_button:
            t_flange = placo_robot.get_T_world_frame("flange")
            pose6 = transform_to_pose6(t_flange)
            self.state.active = True
            self.state.T_ref_flange = t_flange.copy()
            self.state.z_ref = float(pose6[2])
            self.state.rx_ref = float(pose6[3])
            self.state.ry_ref = float(pose6[4])
            self.reset_motion_state()
            print(
                "3DOF 约束已启用（世界系）— 仅 x/y/rz 可动，z/rx/ry 锁定 — "
                f"z_ref={pose6[2]:.4f}, rx_ref={pose6[3]:.4f}, ry_ref={pose6[4]:.4f}"
            )

        if b_pressed and not self._prev_b_button:
            if self.state.active:
                print("3DOF 约束已关闭，参考已清除。")
            self.state.active = False
            self.state.T_ref_flange = None
            self.state.z_ref = 0.0
            self.state.rx_ref = 0.0
            self.state.ry_ref = 0.0
            self.reset_motion_state()
            if resync_active_grip is not None:
                resync_active_grip()

        self._prev_a_button = a_pressed
        self._prev_b_button = b_pressed

    def project_target(self, T_user_world: np.ndarray) -> np.ndarray:
        """Project cumulative VR target onto world x/y/rz DOFs only."""
        t_user = np.asarray(T_user_world, dtype=np.float64).reshape(4, 4)
        if self.state.T_ref_flange is None or self._T_grip_start is None:
            return t_user
        return project_pose_3dof_world(t_user, self._T_grip_start, self.state.T_ref_flange)
