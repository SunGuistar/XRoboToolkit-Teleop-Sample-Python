#!/usr/bin/env python3
"""MuJoCo simulation of horizontal arc motion.

Uses Placo IK to track arc waypoints.

Viewer overlay:
  blue    = axis record (e.g. RECORD_1) tcp_pose in tool_tcp.json
  magenta = world-Z rotation axis through record xy
  orange  = planned TCP arc path
  cyan    = radius from axis to arc start
  green   = arc start (p0)
  red     = arc end

Viewer: Space/R replay, H jump to start.
"""

from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arc_motion import (
    DEFAULT_HOME_FILE,
    DEFAULT_TOOL_TCP_FILE,
    ArcPlan,
    OrientationMode,
    arc_position_xy,
    compute_arc_plan,
    compute_post_motion_errors,
    generate_arc_waypoints,
    load_axis_xy,
    print_post_motion_errors,
    print_pre_motion_info,
    rotvec_to_matrix,
)

_EE_LINK = "tool0"
_UR5E_JOINT_MAP = (
    ("shoulder_pan_joint", "shoulder_pan"),
    ("shoulder_lift_joint", "shoulder_lift"),
    ("elbow_joint", "elbow"),
    ("wrist_1_joint", "wrist_1"),
    ("wrist_2_joint", "wrist_2"),
    ("wrist_3_joint", "wrist_3"),
)
def load_record_joint_q(tool_tcp_file: Path, record_key: str) -> np.ndarray:
    with tool_tcp_file.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return np.asarray(data["records"][record_key]["joint_q_rad"], dtype=float).reshape(6)

def mujoco_body_to_ur_pose(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model.")
    T = tf.quaternion_matrix(mj_data.xquat[body_id].copy())
    T[:3, 3] = mj_data.xpos[body_id].copy()
    pose = np.zeros(6, dtype=float)
    pose[:3] = T[:3, 3]
    pose[3:] = _matrix_to_rotvec(T[:3, :3])
    return pose


def _matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    from xrobotoolkit_teleop.hardware.soft_constraint import rotation_matrix_to_rotvec

    return rotation_matrix_to_rotvec(rotation)


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


def set_mujoco_home_joints(mj_model: mujoco.MjModel, mj_data: mujoco.MjData, joint_q_rad: np.ndarray) -> None:
    q = np.asarray(joint_q_rad, dtype=float).reshape(6)
    qpos = mj_data.qpos.copy()
    for index, (_, mujoco_joint) in enumerate(_UR5E_JOINT_MAP):
        set_mujoco_joint_pos_by_name(mj_model, qpos, mujoco_joint, q[index])
    mj_data.qpos[:] = qpos
    mj_data.ctrl[:] = calc_mujoco_ctrl_from_qpos(mj_model, qpos)
    mujoco.mj_forward(mj_model, mj_data)


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


def tcp_positions_from_waypoints(waypoints: list[np.ndarray]) -> np.ndarray:
    return np.asarray([np.asarray(wp, dtype=float).reshape(6)[:3] for wp in waypoints], dtype=float)


def update_arc_scene(
    viewer,
    plan: ArcPlan,
    planned_tcp: np.ndarray,
    *,
    line_width: float = 0.006,
    axis_z_margin_m: float = 0.08,
) -> None:
    """Draw axis record pose, world-Z rotation axis, and planned arc in the viewer."""
    viewer.user_scn.ngeom = 0
    geom_index = 0
    max_geoms = viewer.user_scn.maxgeom

    def add_segment(p0: np.ndarray, p1: np.ndarray, rgba: np.ndarray, width: float | None = None) -> None:
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
            line_width if width is None else width,
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

    record_rgba = np.array([0.2, 0.45, 1.0, 0.95], dtype=float)
    axis_rgba = np.array([0.85, 0.2, 0.85, 0.85], dtype=float)
    radius_rgba = np.array([0.2, 0.85, 0.95, 0.75], dtype=float)
    arc_rgba = np.array([1.0, 0.55, 0.1, 0.85], dtype=float)
    ring_rgba = np.array([1.0, 0.55, 0.1, 0.35], dtype=float)
    start_rgba = np.array([0.2, 0.9, 0.2, 1.0], dtype=float)
    end_rgba = np.array([0.95, 0.2, 0.2, 1.0], dtype=float)

    record_xyz = np.asarray(plan.axis_record_tcp_pose[:3], dtype=float)
    p0_xyz = np.asarray(plan.p0[:3], dtype=float)
    z_ref = float(p0_xyz[2])
    axis_on_arc_plane = np.array([plan.x_axis, plan.y_axis, z_ref], dtype=float)

    # Blue sphere: recorded TCP pose (e.g. RECORD_1) from tool_tcp.json.
    add_sphere(record_xyz, 0.016, record_rgba)

    # Magenta vertical line: world-Z axis through the record xy.
    z_min = min(record_xyz[2], z_ref) - axis_z_margin_m
    z_max = max(record_xyz[2], z_ref) + axis_z_margin_m + 0.12
    axis_bottom = np.array([plan.x_axis, plan.y_axis, z_min], dtype=float)
    axis_top = np.array([plan.x_axis, plan.y_axis, z_max], dtype=float)
    add_segment(axis_bottom, axis_top, axis_rgba, width=0.008)
    add_sphere(axis_on_arc_plane, 0.010, axis_rgba)

    # Cyan segment: horizontal radius from axis to arc start p0.
    add_segment(axis_on_arc_plane, p0_xyz, radius_rgba)

    # Orange segments: dense planned TCP path.
    for i in range(len(planned_tcp) - 1):
        add_segment(planned_tcp[i], planned_tcp[i + 1], arc_rgba)

    if len(planned_tcp) > 0:
        add_sphere(planned_tcp[0], 0.012, start_rgba)
        add_sphere(planned_tcp[-1], 0.012, end_rgba)

    # Faint orange arc on the horizontal plane (same z as p0).
    x0, y0 = float(p0_xyz[0]), float(p0_xyz[1])
    ring_points = []
    n_ring = max(16, int(abs(plan.angle_deg)))
    for theta in np.linspace(0.0, np.deg2rad(plan.angle_deg), n_ring):
        x, y = arc_position_xy(theta, plan.x_axis, plan.y_axis, x0, y0)
        ring_points.append(np.array([x, y, z_ref], dtype=float))
    for i in range(len(ring_points) - 1):
        add_segment(ring_points[i], ring_points[i + 1], ring_rgba)

    viewer.user_scn.ngeom = geom_index


def animate_qpos_path(
    viewer,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    qpos_path: list[np.ndarray],
    *,
    plan: ArcPlan,
    planned_tcp: np.ndarray,
    step_pause_s: float,
    sim_substeps: int,
    show_trajectory: bool,
) -> None:
    for qpos in qpos_path[1:]:
        if not viewer.is_running():
            break
        apply_mujoco_qpos(mj_model, mj_data, qpos)
        for _ in range(sim_substeps):
            if not viewer.is_running():
                break
            mujoco.mj_step(mj_model, mj_data)
            if show_trajectory:
                update_arc_scene(viewer, plan, planned_tcp)
            viewer.sync()
            time.sleep(step_pause_s)


def main(
    xml_path: str = os.path.join(ASSET_PATH, "universal_robots_ur5e/scene.xml"),
    robot_urdf_path: str = os.path.join(ASSET_PATH, "universal_robots_ur5e/ur5e.urdf"),
    tool_tcp_file: Path = DEFAULT_TOOL_TCP_FILE,
    home_file: Path = DEFAULT_HOME_FILE,
    record_name: str = "RECORD_1",
    angle_deg: float = 60.0,
    orientation_mode: OrientationMode = OrientationMode.FOLLOW_ARC,
    steps: int = 30,
    step_pause_s: float = 0.05,
    sim_substeps: int = 3,
    use_home: bool = True,
    auto_play: bool = True,
    show_trajectory: bool = True,
) -> None:
    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, mj_model.key("home").id)
    mujoco.mj_forward(mj_model, mj_data)

    if use_home:
        if not home_file.exists():
            raise FileNotFoundError(
                f"Home file not found: {home_file}\n"
                "Run on hardware: python scripts/hardware/wrench_data/record_home.py"
            )
        with home_file.open(encoding="utf-8") as handle:
            home_data = json.load(handle)
        set_mujoco_home_joints(mj_model, mj_data, np.asarray(home_data["joint_q_rad"], dtype=float))

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

    record_key, record_tcp_hw, _, _ = load_axis_xy(tool_tcp_file, record_name)

    # 当前 MuJoCo home 状态下的 TCP，作为圆弧起点
    p0 = mujoco_body_to_ur_pose(mj_model, mj_data, _EE_LINK)
    home_qpos_snapshot = snapshot_mujoco_qpos(mj_data)

    # 用 record 的关节角在 MuJoCo 中重建 record pose
    record_q_rad = load_record_joint_q(tool_tcp_file, record_key)
    set_mujoco_home_joints(mj_model, mj_data, record_q_rad)

    axis_record_tcp_pose = mujoco_body_to_ur_pose(mj_model, mj_data, _EE_LINK)
    x_axis = float(axis_record_tcp_pose[0])
    y_axis = float(axis_record_tcp_pose[1])

    # 恢复到 home 起点
    apply_mujoco_qpos(mj_model, mj_data, home_qpos_snapshot)

    print("=== MuJoCo Frame Debug ===")
    print(f"record_key: {record_key}")
    print(f"record_tcp_hw: {record_tcp_hw.tolist()}")
    print(f"p0_mujoco_home: {p0.tolist()}")
    print(f"axis_record_tcp_pose_mujoco: {axis_record_tcp_pose.tolist()}")
    print(f"radius_mujoco: {np.hypot(p0[0] - x_axis, p0[1] - y_axis):.6f}")

    plan = compute_arc_plan(
        p0,
        x_axis,
        y_axis,
        angle_deg,
        orientation_mode,
        record_name=record_key,
        axis_record_tcp_pose=axis_record_tcp_pose,
    )
    print_pre_motion_info(plan)

    arc_waypoints = generate_arc_waypoints(plan, steps=max(steps, 2))
    movec_waypoints = [plan.pose_mid, plan.pose_end]
    planned_tcp_path = tcp_positions_from_waypoints([plan.p0, *arc_waypoints])
    print(f"sim_waypoints: {len(arc_waypoints)} (dense arc), moveC via/end: {len(movec_waypoints)}")
    if show_trajectory:
        print(
            f"Overlay: blue={record_key} tcp_pose, "
            "magenta=world-Z rotation axis, "
            "cyan=radius, orange=planned arc (green=start, red=end)"
        )

    start_qpos = snapshot_mujoco_qpos(mj_data)
    qpos_keyframes = [start_qpos.copy()]

    failed = 0
    for target_pose in arc_waypoints:
        if not solve_tcp_ik(solver, frame_task, joints_task, placo_robot, target_pose):
            failed += 1
        sync_mujoco_from_placo(mj_model, mj_data, placo_robot)
        qpos_keyframes.append(snapshot_mujoco_qpos(mj_data))

    apply_mujoco_qpos(mj_model, mj_data, qpos_keyframes[-1])
    actual_tcp = mujoco_body_to_ur_pose(mj_model, mj_data, _EE_LINK)
    errors = compute_post_motion_errors(plan, actual_tcp)
    print_post_motion_errors(plan, errors)
    if failed:
        print(f"WARNING: {failed}/{len(arc_waypoints)} IK waypoints missed tolerance.", file=sys.stderr)

    controls = ViewerControls()
    with mj_viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=make_viewer_key_callback(controls),
    ) as viewer:
        viewer.sync()
        print("Viewer: Space/R replay, H jump to start, close window to exit.")

        if show_trajectory:
            update_arc_scene(viewer, plan, planned_tcp_path)

        if auto_play:
            controls.is_playing = True
            animate_qpos_path(
                viewer,
                mj_model,
                mj_data,
                qpos_keyframes,
                plan=plan,
                planned_tcp=planned_tcp_path,
                step_pause_s=step_pause_s,
                sim_substeps=sim_substeps,
                show_trajectory=show_trajectory,
            )
            controls.is_playing = False
            actual_tcp = mujoco_body_to_ur_pose(mj_model, mj_data, _EE_LINK)
            errors = compute_post_motion_errors(plan, actual_tcp)
            print_post_motion_errors(plan, errors)

        while viewer.is_running():
            if controls.jump_to_start_requested and not controls.is_playing:
                controls.jump_to_start_requested = False
                apply_mujoco_qpos(mj_model, mj_data, start_qpos)

            if controls.replay_requested and not controls.is_playing:
                controls.replay_requested = False
                controls.is_playing = True
                apply_mujoco_qpos(mj_model, mj_data, start_qpos)
                animate_qpos_path(
                    viewer,
                    mj_model,
                    mj_data,
                    qpos_keyframes,
                    plan=plan,
                    planned_tcp=planned_tcp_path,
                    step_pause_s=step_pause_s,
                    sim_substeps=sim_substeps,
                    show_trajectory=show_trajectory,
                )
                controls.is_playing = False

            if show_trajectory:
                update_arc_scene(viewer, plan, planned_tcp_path)
            viewer.sync()
            time.sleep(0.03)


if __name__ == "__main__":
    tyro.cli(main)
