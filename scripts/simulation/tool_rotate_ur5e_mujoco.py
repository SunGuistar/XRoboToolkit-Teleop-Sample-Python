#!/usr/bin/env python3
"""MuJoCo + Placo 仿真 tool_rotate 路径：虚拟工具中心固定，绕 base Z 旋转。

不依赖 VR。用 Placo IK 跟踪 compute_target_pose() 生成的 TCP 路点，在 MuJoCo 中播放。

Usage:
  python scripts/simulation/tool_rotate_ur5e_mujoco.py
  python scripts/simulation/tool_rotate_ur5e_mujoco.py --angle-deg 10 --pose-name POSE_1
  python scripts/simulation/tool_rotate_ur5e_mujoco.py --angle-deg -30 --steps 40

Viewer (after first play):
  Space / R  replay rotation
  H          jump to rotation start pose
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import placo
import tyro
from meshcat import transformations as tf
from mujoco import viewer as mj_viewer

from xrobotoolkit_teleop.utils.mujoco_utils import (
    calc_mujoco_ctrl_from_qpos,
    set_mujoco_joint_pos_by_name,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH
from xrobotoolkit_teleop.utils.ur_pose import matrix_to_rotvec, rotvec_to_matrix

_HARDWARE_TOOL_ROTATE = Path(__file__).resolve().parent.parent / "hardware" / "tool_rotate.py"
_DEFAULT_POSE_FILE = Path(__file__).resolve().parent.parent / "hardware" / "tool_tcp_rotate.json"
_EE_LINK = "tool0"
_MOCAP_TOOL_CENTER = "target"
_UR5E_JOINT_MAP = (
    ("shoulder_pan_joint", "shoulder_pan"),
    ("shoulder_lift_joint", "shoulder_lift"),
    ("elbow_joint", "elbow"),
    ("wrist_1_joint", "wrist_1"),
    ("wrist_2_joint", "wrist_2"),
    ("wrist_3_joint", "wrist_3"),
)


def _load_tool_rotate_module():
    spec = importlib.util.spec_from_file_location("tool_rotate", _HARDWARE_TOOL_ROTATE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {_HARDWARE_TOOL_ROTATE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["tool_rotate"] = module
    spec.loader.exec_module(module)
    return module


def load_recorded_pose(pose_file: Path, pose_name: str) -> np.ndarray:
    with pose_file.open(encoding="utf-8") as handle:
        data = json.load(handle)
    poses = data.get("poses", {})
    if pose_name not in poses:
        available = ", ".join(sorted(poses.keys())) or "(empty)"
        raise KeyError(f"Pose '{pose_name}' not in {pose_file}. Available: {available}")
    return np.asarray(poses[pose_name], dtype=float).reshape(6)


def mujoco_body_to_ur_pose(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model.")

    T = tf.quaternion_matrix(mj_data.xquat[body_id].copy())
    T[:3, 3] = mj_data.xpos[body_id].copy()
    pose = np.zeros(6, dtype=float)
    pose[:3] = T[:3, 3]
    pose[3:] = matrix_to_rotvec(T[:3, :3])
    return pose


def ur_pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    pose = np.asarray(pose, dtype=float).reshape(6)
    T = np.eye(4)
    T[:3, :3] = rotvec_to_matrix(pose[3:])
    T[:3, 3] = pose[:3]
    return T


def sync_mujoco_from_placo(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, placo_robot: placo.RobotWrapper) -> None:
    qpos = mj_data.qpos.copy()
    for placo_joint, mujoco_joint in _UR5E_JOINT_MAP:
        set_mujoco_joint_pos_by_name(mj_model, qpos, mujoco_joint, placo_robot.get_joint(placo_joint))
    mj_data.qpos[:] = qpos
    mj_data.ctrl[:] = calc_mujoco_ctrl_from_qpos(mj_model, qpos)
    mujoco.mj_forward(mj_model, mj_data)


def sync_placo_from_mujoco(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, placo_robot: placo.RobotWrapper) -> None:
    placo_robot.state.q[:7] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    for placo_joint, mujoco_joint in _UR5E_JOINT_MAP:
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, mujoco_joint)
        qpos_addr = mj_model.jnt_qposadr[joint_id]
        placo_robot.set_joint(placo_joint, mj_data.qpos[qpos_addr])
    placo_robot.update_kinematics()


def set_mocap_position(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, body_name: str, position: np.ndarray) -> None:
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        return
    mocap_id = mj_model.body_mocapid[body_id]
    if mocap_id == -1:
        return
    mj_data.mocap_pos[mocap_id] = np.asarray(position, dtype=float).reshape(3)


def solve_tcp_ik(
    solver: placo.KinematicsSolver,
    frame_task,
    joints_task,
    placo_robot: placo.RobotWrapper,
    tcp_pose: np.ndarray,
    ee_link: str = _EE_LINK,
    ik_iterations: int = 5,
    position_tol_m: float = 2e-2,
    orientation_tol_rad: float = 0.15,
) -> bool:
    frame_task.T_world_frame = ur_pose_to_matrix(tcp_pose)
    joints_task.set_joints({joint: placo_robot.get_joint(joint) for joint in placo_robot.joint_names()})

    for _ in range(ik_iterations):
        solver.solve(True)
        placo_robot.update_kinematics()

    achieved = placo_robot.get_T_world_frame(ee_link)
    target = frame_task.T_world_frame
    pos_err = float(np.linalg.norm(achieved[:3, 3] - target[:3, 3]))
    R_err = achieved[:3, :3].T @ target[:3, :3]
    trace = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    ori_err = float(np.arccos(trace))
    return pos_err <= position_tol_m and ori_err <= orientation_tol_rad


def follow_tcp_path(
    solver: placo.KinematicsSolver,
    frame_task,
    joints_task,
    placo_robot: placo.RobotWrapper,
    tcp_path: list[np.ndarray],
) -> int:
    """Track a TCP path with Placo IK. Returns number of failed waypoints."""
    failed = 0
    for target_pose in tcp_path:
        if not solve_tcp_ik(solver, frame_task, joints_task, placo_robot, target_pose):
            failed += 1
    return failed


def interpolate_tcp_poses(start_pose: np.ndarray, end_pose: np.ndarray, steps: int) -> list[np.ndarray]:
    """Linear position + slerp-like rotvec interpolation between two UR poses."""
    if steps < 2:
        return [np.asarray(end_pose, dtype=float).reshape(6)]

    start_pose = np.asarray(start_pose, dtype=float).reshape(6)
    end_pose = np.asarray(end_pose, dtype=float).reshape(6)
    path: list[np.ndarray] = []

    T0 = ur_pose_to_matrix(start_pose)
    T1 = ur_pose_to_matrix(end_pose)
    R0 = T0[:3, :3]
    R1 = T1[:3, :3]
    R_delta = R0.T @ R1
    delta_rotvec = matrix_to_rotvec(R_delta)

    for alpha in np.linspace(0.0, 1.0, steps):
        pose = np.zeros(6, dtype=float)
        pose[:3] = (1.0 - alpha) * start_pose[:3] + alpha * end_pose[:3]
        R_step = R0 @ rotvec_to_matrix(alpha * delta_rotvec)
        pose[3:] = matrix_to_rotvec(R_step)
        path.append(pose)

    return path


def tcp_positions_from_waypoints(waypoints: list[np.ndarray]) -> np.ndarray:
    return np.asarray([np.asarray(wp, dtype=float).reshape(6)[:3] for wp in waypoints], dtype=float)


def update_trajectory_scene(
    viewer,
    planned_tcp: np.ndarray,
    executed_tcp: np.ndarray,
    tool_center: np.ndarray,
    *,
    line_width: float = 0.006,
    show_tool_center_ring: bool = True,
) -> None:
    """Draw planned/executed TCP paths in the passive viewer overlay."""
    viewer.user_scn.ngeom = 0
    geom_index = 0
    max_geoms = viewer.user_scn.maxgeom

    def add_segment(p0: np.ndarray, p1: np.ndarray, rgba: np.ndarray) -> None:
        nonlocal geom_index
        if geom_index >= max_geoms:
            return
        geom = viewer.user_scn.geoms[geom_index]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            line_width,
            np.asarray(p0, dtype=float).reshape(3),
            np.asarray(p1, dtype=float).reshape(3),
        )
        geom_index += 1

    def add_sphere(position: np.ndarray, radius: float, rgba: np.ndarray) -> None:
        nonlocal geom_index
        if geom_index >= max_geoms:
            return
        geom = viewer.user_scn.geoms[geom_index]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([radius, 0.0, 0.0]),
            pos=np.asarray(position, dtype=float).reshape(3),
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )
        geom_index += 1

    planned_rgba = np.array([1.0, 0.55, 0.1, 0.85], dtype=float)
    executed_rgba = np.array([0.15, 0.85, 0.35, 0.9], dtype=float)
    center_rgba = np.array([0.2, 0.45, 1.0, 0.75], dtype=float)

    for i in range(len(planned_tcp) - 1):
        add_segment(planned_tcp[i], planned_tcp[i + 1], planned_rgba)

    for i in range(len(executed_tcp) - 1):
        add_segment(executed_tcp[i], executed_tcp[i + 1], executed_rgba)

    if len(planned_tcp) > 0:
        add_sphere(planned_tcp[0], 0.012, np.array([0.2, 0.9, 0.2, 1.0]))
        add_sphere(planned_tcp[-1], 0.012, np.array([0.95, 0.2, 0.2, 1.0]))

    if len(executed_tcp) > 0:
        add_sphere(executed_tcp[-1], 0.010, np.array([0.1, 1.0, 0.4, 1.0]))

    add_sphere(tool_center, 0.014, center_rgba)

    if show_tool_center_ring and len(planned_tcp) >= 2:
        radius_xy = float(np.linalg.norm((planned_tcp[0] - tool_center)[:2]))
        if radius_xy > 1e-6:
            ring_points = []
            for theta in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
                ring_points.append(
                    tool_center
                    + np.array([radius_xy * np.cos(theta), radius_xy * np.sin(theta), 0.0])
                )
            ring_rgba = np.array([0.2, 0.45, 1.0, 0.35], dtype=float)
            for i in range(len(ring_points)):
                add_segment(ring_points[i], ring_points[(i + 1) % len(ring_points)], ring_rgba)

    viewer.user_scn.ngeom = geom_index


@dataclass
class ViewerControls:
    replay_requested: bool = False
    jump_to_start_requested: bool = False
    is_playing: bool = False


def make_viewer_key_callback(controls: ViewerControls):
    def key_callback(keycode: int) -> None:
        key = chr(keycode)
        if key in {" ", "r", "R"}:
            controls.replay_requested = True
        elif key in {"h", "H"}:
            controls.jump_to_start_requested = True

    return key_callback


def snapshot_mujoco_qpos(mj_data: mujoco.MjData) -> np.ndarray:
    return mj_data.qpos.copy()


def apply_mujoco_qpos(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, qpos: np.ndarray) -> None:
    mj_data.qpos[:] = qpos
    mj_data.ctrl[:] = calc_mujoco_ctrl_from_qpos(mj_model, qpos)
    mujoco.mj_forward(mj_model, mj_data)


def setup_viewer_camera(viewer, tool_center_base: np.ndarray) -> None:
    scene_center = np.array([0.25, 0.0, 0.35])
    lookat = 0.35 * scene_center + 0.65 * tool_center_base
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -28
    viewer.cam.distance = max(3.5, float(np.linalg.norm(lookat) + 2.0))
    viewer.cam.lookat = lookat


def animate_qpos_path(
    viewer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    qpos_path: list[np.ndarray],
    start_tcp_xyz: np.ndarray,
    planned_tcp_path: np.ndarray,
    tool_center_base: np.ndarray,
    *,
    step_pause_s: float,
    sim_substeps: int,
    show_trajectory: bool,
    show_tool_center_ring: bool,
) -> list[np.ndarray]:
    """Play back a recorded joint-space path and return executed TCP xyz samples."""
    executed_tcp_path: list[np.ndarray] = [start_tcp_xyz.copy()]

    for qpos in qpos_path[1:]:
        if not viewer.is_running():
            break

        apply_mujoco_qpos(mj_model, mj_data, qpos)
        executed_tcp_path.append(mujoco_body_to_ur_pose(mj_model, mj_data, _EE_LINK)[:3].copy())

        for _ in range(sim_substeps):
            if not viewer.is_running():
                break
            mujoco.mj_step(mj_model, mj_data)
            if show_trajectory:
                update_trajectory_scene(
                    viewer,
                    planned_tcp_path,
                    np.asarray(executed_tcp_path),
                    tool_center_base,
                    show_tool_center_ring=show_tool_center_ring,
                )
            viewer.sync()
            time.sleep(step_pause_s)

    return executed_tcp_path


def main(
    xml_path: str = os.path.join(ASSET_PATH, "universal_robots_ur5e/scene.xml"),
    robot_urdf_path: str = os.path.join(ASSET_PATH, "universal_robots_ur5e/ur5e.urdf"),
    pose_file: Path = _DEFAULT_POSE_FILE,
    pose_name: str | None = "POSE_1",
    angle_deg: float = 10.0,
    steps: int = 30,
    step_pause_s: float = 0.05,
    tool_offset_tcp_m: tuple[float, float, float] | None = None,
    seed_steps: int = 20,
    use_home: bool = False,
    show_trajectory: bool = True,
    show_tool_center_ring: bool = True,
    auto_play: bool = True,
    sim_substeps: int = 3,
) -> None:
    """Simulate virtual-tool-center rotation in MuJoCo.

    Args:
        pose_name: Recorded pose from tool_tcp_rotate.json used as rotation start.
        use_home: If True, ignore pose_file and start from MJCF home keyframe.
        tool_offset_tcp_m: Override TOOL_OFFSET_TCP_M from tool_rotate.py.
        auto_play: Play once when the viewer opens. Use Space/R to replay later.
    """
    tool_rotate = _load_tool_rotate_module()
    tool_offset = (
        np.asarray(tool_offset_tcp_m, dtype=float)
        if tool_offset_tcp_m is not None
        else tool_rotate.TOOL_OFFSET_TCP_M.copy()
    )

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, mj_model.key("home").id)
    mujoco.mj_forward(mj_model, mj_data)

    placo_robot = placo.RobotWrapper(robot_urdf_path)
    solver = placo.KinematicsSolver(placo_robot)
    solver.mask_fbase(True)
    solver.dt = step_pause_s

    sync_placo_from_mujoco(mj_model, mj_data, placo_robot)

    frame_task = solver.add_frame_task(_EE_LINK, placo_robot.get_T_world_frame(_EE_LINK))
    frame_task.configure("tool0", "soft", 10.0)
    manipulability = solver.add_manipulability_task(_EE_LINK, "both", 1.0)
    manipulability.configure("manipulability", "soft", 1e-2)

    joints_task = solver.add_joints_task()
    joints_task.set_joints({joint: 0.0 for joint in placo_robot.joint_names()})
    joints_task.configure("joints_regularization", "soft", 1e-4)

    if use_home or pose_name is None:
        start_tcp_pose = mujoco_body_to_ur_pose(mj_model, mj_data, _EE_LINK)
        print("Start pose: MuJoCo home keyframe")
    else:
        recorded_pose = load_recorded_pose(pose_file, pose_name)
        current_pose = mujoco_body_to_ur_pose(mj_model, mj_data, _EE_LINK)
        seed_path = interpolate_tcp_poses(current_pose, recorded_pose, seed_steps)
        seed_failed = follow_tcp_path(solver, frame_task, joints_task, placo_robot, seed_path[1:])
        sync_mujoco_from_placo(mj_model, mj_data, placo_robot)
        start_tcp_pose = mujoco_body_to_ur_pose(mj_model, mj_data, _EE_LINK)
        print(f"Start pose: {pose_name} from {pose_file}")
        if seed_failed:
            print(
                f"WARNING: {seed_failed}/{len(seed_path) - 1} seed waypoints missed IK tolerance.",
                file=sys.stderr,
            )

    angle_rad = float(np.deg2rad(angle_deg))
    waypoint_angles = np.linspace(0.0, angle_rad, max(steps, 2))
    waypoints: list[np.ndarray] = []
    tool_center_base: np.ndarray | None = None

    for angle in waypoint_angles:
        target_pose, center = tool_rotate.compute_target_pose(start_tcp_pose, tool_offset, float(angle))
        waypoints.append(target_pose)
        tool_center_base = center

    assert tool_center_base is not None
    final_pose = waypoints[-1]
    planned_drift_mm = tool_rotate.verify_tool_center(final_pose, tool_offset, tool_center_base)

    p_tcp = start_tcp_pose[:3]
    target_p = final_pose[:3]

    print("=== Tool Rotate MuJoCo Sim ===")
    print(f"angle_deg: {angle_deg:+.2f}")
    print(f"steps: {steps}")
    print(f"tool_offset_tcp_m: {tool_offset}")
    print(f"tool_center_base: {tool_center_base}")
    print()
    print("Current TCP:", p_tcp)
    print("Tool Center:", tool_center_base)
    print("TCP->Tool =", tool_center_base - p_tcp)
    print("Radius XY =", np.linalg.norm((tool_center_base - p_tcp)[:2]))
    print("Target TCP:", target_p)
    print("TCP displacement:", np.linalg.norm(target_p - p_tcp))
    print()
    print(f"planned tool center drift mm: {planned_drift_mm:.6f}")
    print(f"waypoints: {len(waypoints)}")
    print("Trajectory colors: orange=planned TCP, green=executed TCP, blue=tool center")
    print()

    planned_tcp_path = tcp_positions_from_waypoints(waypoints)
    rotation_qpos_keyframes: list[np.ndarray] = []

    set_mocap_position(mj_model, mj_data, _MOCAP_TOOL_CENTER, tool_center_base)

    # Pre-compute IK and record joint keyframes before opening the viewer.
    rotation_start_qpos = snapshot_mujoco_qpos(mj_data)
    rotation_qpos_keyframes.append(rotation_start_qpos.copy())

    failed_steps = 0
    for index, target_pose in enumerate(waypoints[1:], start=1):
        if not solve_tcp_ik(solver, frame_task, joints_task, placo_robot, target_pose):
            failed_steps += 1
            print(f"WARNING: IK failed at waypoint {index}/{len(waypoints) - 1}", file=sys.stderr)
        sync_mujoco_from_placo(mj_model, mj_data, placo_robot)
        rotation_qpos_keyframes.append(snapshot_mujoco_qpos(mj_data))

    final_qpos = rotation_qpos_keyframes[-1]
    apply_mujoco_qpos(mj_model, mj_data, final_qpos)
    actual_pose = mujoco_body_to_ur_pose(mj_model, mj_data, _EE_LINK)
    actual_drift_mm = tool_rotate.verify_tool_center(actual_pose, tool_offset, tool_center_base)

    print("=== Planned Result (IK) ===")
    print(f"final_tcp_pose: {actual_pose}")
    print(f"planned TCP xyz delta mm: {np.linalg.norm(actual_pose[:3] - start_tcp_pose[:3]) * 1000.0:.3f}")
    print(f"planned tool center drift mm: {actual_drift_mm:.3f}")
    if failed_steps:
        print(f"IK failures: {failed_steps}/{len(waypoints) - 1}", file=sys.stderr)
    print()
    print("Viewer controls: Space/R = replay, H = jump to start pose")
    print("Close viewer window to exit.")
    print()

    controls = ViewerControls()
    executed_tcp_path: list[np.ndarray] = [start_tcp_pose[:3].copy()]

    with mj_viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=make_viewer_key_callback(controls),
    ) as viewer:
        setup_viewer_camera(viewer, tool_center_base)
        viewer.sync()

        if auto_play:
            controls.is_playing = True
            executed_tcp_path = animate_qpos_path(
                viewer,
                mj_model,
                mj_data,
                rotation_qpos_keyframes,
                start_tcp_pose[:3],
                planned_tcp_path,
                tool_center_base,
                step_pause_s=step_pause_s,
                sim_substeps=sim_substeps,
                show_trajectory=show_trajectory,
                show_tool_center_ring=show_tool_center_ring,
            )
            controls.is_playing = False

            actual_pose = mujoco_body_to_ur_pose(mj_model, mj_data, _EE_LINK)
            actual_drift_mm = tool_rotate.verify_tool_center(actual_pose, tool_offset, tool_center_base)
            print("=== Playback Result ===")
            print(f"final_tcp_pose: {actual_pose}")
            print(f"actual TCP xyz delta mm: {np.linalg.norm(actual_pose[:3] - start_tcp_pose[:3]) * 1000.0:.3f}")
            print(f"actual tool center drift mm: {actual_drift_mm:.3f}")
            print("Press Space/R to replay.")
        else:
            apply_mujoco_qpos(mj_model, mj_data, rotation_start_qpos)
            if show_trajectory:
                update_trajectory_scene(
                    viewer,
                    planned_tcp_path,
                    np.asarray(executed_tcp_path),
                    tool_center_base,
                    show_tool_center_ring=show_tool_center_ring,
                )
            viewer.sync()
            print("Auto-play disabled. Press Space/R to play.")

        while viewer.is_running():
            if controls.jump_to_start_requested and not controls.is_playing:
                controls.jump_to_start_requested = False
                apply_mujoco_qpos(mj_model, mj_data, rotation_start_qpos)
                executed_tcp_path = [start_tcp_pose[:3].copy()]

            if controls.replay_requested and not controls.is_playing:
                controls.replay_requested = False
                controls.is_playing = True
                apply_mujoco_qpos(mj_model, mj_data, rotation_start_qpos)
                executed_tcp_path = animate_qpos_path(
                    viewer,
                    mj_model,
                    mj_data,
                    rotation_qpos_keyframes,
                    start_tcp_pose[:3],
                    planned_tcp_path,
                    tool_center_base,
                    step_pause_s=step_pause_s,
                    sim_substeps=sim_substeps,
                    show_trajectory=show_trajectory,
                    show_tool_center_ring=show_tool_center_ring,
                )
                controls.is_playing = False
                print("Replay finished. Press Space/R to replay again.")

            if show_trajectory and not controls.is_playing:
                update_trajectory_scene(
                    viewer,
                    planned_tcp_path,
                    np.asarray(executed_tcp_path),
                    tool_center_base,
                    show_tool_center_ring=show_tool_center_ring,
                )

            viewer.sync()
            time.sleep(0.03)


if __name__ == "__main__":
    tyro.cli(main)
