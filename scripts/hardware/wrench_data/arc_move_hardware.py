#!/usr/bin/env python3
"""Execute horizontal arc motion on UR5 hardware via RTDE movePath (moveL + blend).

The rotation axis is vertical (world Z) and passes through RECORD_假定的工具TCP xy in tool_tcp.json.

Usage:
  python scripts/hardware/wrench_data/arc_move_hardware.py --angle-deg 60
  python scripts/hardware/wrench_data/arc_move_hardware.py --orientation-mode FOLLOW_ARC
  python scripts/hardware/wrench_data/arc_move_hardware.py --blend 0.012
  python scripts/hardware/wrench_data/arc_move_hardware.py --move-home-first
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rtde_control
import tyro

from xrobotoolkit_teleop.hardware.interface.universal_robots import connect_rtde, DEFAULT_ROBOT_IP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arc_motion import (
    DEFAULT_HOME_FILE,
    DEFAULT_RECORD_NAME,
    DEFAULT_TOOL_TCP_FILE,
    OrientationMode,
    compute_arc_plan,
    compute_post_motion_errors,
    generate_arc_waypoints,
    load_axis_xy,
    print_post_motion_errors,
    print_pre_motion_info,
)
from record_home import load_home_joints


def wait_until_stopped(rtde_r, timeout_s: float = 120.0, poll_s: float = 0.05) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        speed = np.linalg.norm(rtde_r.getActualTCPSpeed()[:3])
        if speed < 1e-3:
            return
        time.sleep(poll_s)
    raise TimeoutError("Timed out waiting for robot to stop.")


def build_movel_path(
    waypoints: list[np.ndarray],
    velocity: float,
    acceleration: float,
    blend: float,
) -> rtde_control.Path:
    """Build a moveL path with blend on interior waypoints (last waypoint blend=0)."""
    path = rtde_control.Path()
    rows: list[list[float]] = []
    last_index = len(waypoints) - 1
    for index, pose in enumerate(waypoints):
        blend_radius = 0.0 if index == last_index else blend
        rows.append([*pose.tolist(), velocity, acceleration, blend_radius])
    path.appendMovelPath(rows)
    return path


def main(
    robot_ip: str = DEFAULT_ROBOT_IP,
    tool_tcp_file: Path = DEFAULT_TOOL_TCP_FILE,
    home_file: Path = DEFAULT_HOME_FILE,
    home_record_name: str | None = None,
    record_name: str = DEFAULT_RECORD_NAME,
    angle_deg: float = 60.0,
    orientation_mode: OrientationMode = OrientationMode.FOLLOW_ARC,
    acceleration: float = 0.2,
    velocity: float = 0.03,
    steps: int = 30,
    blend: float = 0.012,
    move_home_first: bool = False,
    dry_run: bool = False,
) -> None:
    rtde_c = None
    rtde_r = None

    try:
        rtde_c, rtde_r = connect_rtde(robot_ip)

        if move_home_first:
            if not home_file.exists():
                raise FileNotFoundError(
                    f"Home file not found: {home_file}\n"
                    "Run: python scripts/hardware/wrench_data/record_home.py"
                )
            home_q = load_home_joints(home_file, home_record_name)
            print(f"Moving to home joints (rad): {home_q}")
            if not rtde_c.moveJ(home_q.tolist()):
                raise RuntimeError("moveJ to home failed.")
            wait_until_stopped(rtde_r)
            print("Reached home.")

        record_key, record_tcp, x_axis, y_axis = load_axis_xy(tool_tcp_file, record_name)
        p0 = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64).reshape(6)

        plan = compute_arc_plan(
            p0,
            x_axis,
            y_axis,
            angle_deg,
            orientation_mode,
            record_name=record_key,
            axis_record_tcp_pose=record_tcp,
        )
        print_pre_motion_info(plan)

        waypoints = generate_arc_waypoints(plan, steps=max(steps, 2))
        movel_path = build_movel_path(waypoints, velocity, acceleration, blend)
        print(f"movePath waypoints: {len(waypoints)}, blend_m: {blend:.4f} (interior), 0.0 (final)")

        if dry_run:
            print("Dry run: movePath skipped.")
            return

        if not rtde_c.movePath(movel_path):
            raise RuntimeError("movePath failed.")

        wait_until_stopped(rtde_r)
        actual_tcp = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64).reshape(6)
        errors = compute_post_motion_errors(plan, actual_tcp)
        print_post_motion_errors(plan, errors)

    finally:
        if rtde_c is not None:
            try:
                rtde_c.stopScript()
            except Exception:
                pass
        if rtde_r is not None:
            try:
                rtde_r.disconnect()
            except Exception:
                pass


if __name__ == "__main__":
    try:
        tyro.cli(main)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
