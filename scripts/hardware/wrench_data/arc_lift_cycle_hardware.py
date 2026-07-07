#!/usr/bin/env python3
"""Repeat arc rotate -> lift -> reverse arc -> lower on UR5 hardware.

Each cycle:
  0. moveJ to home
  1. Rotate -angle_deg around world-Z through tool_tcp record xy
  2. moveL up by lift_m (world Z)
  3. Rotate +angle_deg
  4. moveL down by lift_m

Usage:
  python scripts/hardware/wrench_data/arc_lift_cycle_hardware.py
  python scripts/hardware/wrench_data/arc_lift_cycle_hardware.py --cycles 48 --angle-deg 60 --lift-m 0.05
  python scripts/hardware/wrench_data/arc_lift_cycle_hardware.py --reset
  python scripts/hardware/wrench_data/arc_lift_cycle_hardware.py --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
from arc_move_hardware import build_movel_path, wait_until_stopped
from record_home import load_home_joints


def move_to_home(
    rtde_c,
    rtde_r,
    home_q: np.ndarray,
    *,
    home_velocity: float,
    home_acceleration: float,
    label: str,
    verbose: bool,
) -> None:
    if verbose:
        print(f"\n=== {label} ===")
        print(f"home joints (rad): {home_q.tolist()}")
    else:
        print(f"{label}...", end=" ", flush=True)
    if not rtde_c.moveJ(home_q.tolist(), home_velocity, home_acceleration):
        raise RuntimeError(f"{label}: moveJ to home failed.")
    wait_until_stopped(rtde_r)
    if verbose:
        print(f"actual_tcp_pose: {[float(v) for v in rtde_r.getActualTCPPose()]}")
    else:
        print("done")


def execute_arc(
    rtde_c,
    rtde_r,
    *,
    x_axis: float,
    y_axis: float,
    record_key: str,
    record_tcp: np.ndarray,
    angle_deg: float,
    orientation_mode: OrientationMode,
    velocity: float,
    acceleration: float,
    steps: int,
    blend: float,
    label: str,
    verbose: bool,
) -> None:
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
    if verbose:
        print(f"\n=== {label} ===")
        print_pre_motion_info(plan)
    else:
        print(f"{label}...", end=" ", flush=True)

    step_count = max(steps, 2)
    waypoints = generate_arc_waypoints(plan, steps=step_count)
    movel_path = build_movel_path(waypoints, velocity, acceleration, blend)
    if verbose:
        print(f"movePath waypoints: {len(waypoints)}, blend_m: {blend:.4f} (interior), 0.0 (final)")

    if not rtde_c.movePath(movel_path):
        raise RuntimeError(f"{label}: movePath failed.")
    wait_until_stopped(rtde_r)

    if verbose:
        actual_tcp = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64).reshape(6)
        errors = compute_post_motion_errors(plan, actual_tcp)
        print_post_motion_errors(plan, errors)
    else:
        print("done")


def execute_z_move(
    rtde_c,
    rtde_r,
    *,
    dz_m: float,
    velocity: float,
    acceleration: float,
    label: str,
    verbose: bool,
) -> None:
    tcp = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64).reshape(6).copy()
    tcp[2] += dz_m
    if verbose:
        print(f"\n=== {label} ===")
        print(f"dz_m: {dz_m:+.4f}, target_tcp_pose: {tcp.tolist()}")
    else:
        print(f"{label}...", end=" ", flush=True)

    if not rtde_c.moveL(tcp.tolist(), velocity, acceleration):
        raise RuntimeError(f"{label}: moveL failed.")
    wait_until_stopped(rtde_r)
    if verbose:
        print(f"actual_tcp_pose: {[float(v) for v in rtde_r.getActualTCPPose()]}")
    else:
        print("done")


def main(
    robot_ip: str = DEFAULT_ROBOT_IP,
    tool_tcp_file: Path = DEFAULT_TOOL_TCP_FILE,
    home_file: Path = DEFAULT_HOME_FILE,
    home_record_name: str | None = None,
    record_name: str = DEFAULT_RECORD_NAME,
    angle_deg: float = 60.0,
    lift_m: float = 0.05,
    cycles: int = 48,
    orientation_mode: OrientationMode = OrientationMode.FOLLOW_ARC,
    acceleration: float = 0.5,
    velocity: float = 0.06,
    home_velocity: float = 1.2,
    home_acceleration: float = 1.5,
    steps: int = 18,
    blend: float = 0.02,
    reset: bool = False,
    verbose: bool = False,
    dry_run: bool = False,
) -> None:
    if not home_file.exists():
        raise FileNotFoundError(
            f"Home file not found: {home_file}\n"
            "Run: python scripts/hardware/wrench_data/record_home.py"
        )

    rtde_c = None
    rtde_r = None

    try:
        rtde_c, rtde_r = connect_rtde(robot_ip)
        home_q = load_home_joints(home_file, home_record_name)

        if reset:
            move_to_home(
                rtde_c,
                rtde_r,
                home_q,
                home_velocity=home_velocity,
                home_acceleration=home_acceleration,
                label="Reset: move to home",
                verbose=True,
            )
            return

        if cycles < 1:
            raise ValueError(f"cycles must be >= 1, got {cycles}")
        if lift_m <= 0.0:
            raise ValueError(f"lift_m must be > 0, got {lift_m}")

        record_key, record_tcp, x_axis, y_axis = load_axis_xy(tool_tcp_file, record_name)
        print("=== Arc Lift Cycle Plan ===")
        print(f"cycles: {cycles}")
        print(f"angle_deg: -{angle_deg:.1f} / +{angle_deg:.1f}")
        print(f"lift_m: {lift_m:.4f}")
        print(f"velocity_m_s: {velocity:.3f}, acceleration_m_s2: {acceleration:.3f}")
        print(f"home_velocity_rad_s: {home_velocity:.3f}, home_acceleration_rad_s2: {home_acceleration:.3f}")
        print(f"steps: {steps}, blend_m: {blend:.4f}, verbose: {verbose}")
        print(f"home_file: {home_file}")
        print(f"home_record: {home_record_name or '(default)'}")
        print(f"home joints (rad): {home_q.tolist()}")

        if dry_run:
            print("Dry run: motion skipped.")
            return

        cycle_start_tcps: list[np.ndarray] = []

        for cycle in range(1, cycles + 1):
            print(f"\n######## Cycle {cycle}/{cycles} ########")
            move_to_home(
                rtde_c,
                rtde_r,
                home_q,
                home_velocity=home_velocity,
                home_acceleration=home_acceleration,
                label=f"Cycle {cycle}: move to home",
                verbose=verbose,
            )
            cycle_start_tcps.append(np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64).reshape(6))
            execute_arc(
                rtde_c,
                rtde_r,
                x_axis=x_axis,
                y_axis=y_axis,
                record_key=record_key,
                record_tcp=record_tcp,
                angle_deg=-angle_deg,
                orientation_mode=orientation_mode,
                velocity=velocity,
                acceleration=acceleration,
                steps=steps,
                blend=blend,
                label=f"Cycle {cycle}: rotate -{angle_deg:.1f} deg",
                verbose=verbose,
            )
            execute_z_move(
                rtde_c,
                rtde_r,
                dz_m=lift_m,
                velocity=velocity,
                acceleration=acceleration,
                label=f"Cycle {cycle}: lift +{lift_m * 100.0:.1f} cm",
                verbose=verbose,
            )
            execute_arc(
                rtde_c,
                rtde_r,
                x_axis=x_axis,
                y_axis=y_axis,
                record_key=record_key,
                record_tcp=record_tcp,
                angle_deg=angle_deg,
                orientation_mode=orientation_mode,
                velocity=velocity,
                acceleration=acceleration,
                steps=steps,
                blend=blend,
                label=f"Cycle {cycle}: rotate +{angle_deg:.1f} deg",
                verbose=verbose,
            )
            execute_z_move(
                rtde_c,
                rtde_r,
                dz_m=-lift_m,
                velocity=velocity,
                acceleration=acceleration,
                label=f"Cycle {cycle}: lower -{lift_m * 100.0:.1f} cm",
                verbose=verbose,
            )

        end_tcp = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64).reshape(6)
        print("\n=== All Cycles Complete ===")
        print(f"last_cycle_end_tcp_pose: {end_tcp.tolist()}")
        if len(cycle_start_tcps) >= 2:
            ref_tcp = cycle_start_tcps[0]
            max_cycle_start_delta_m = max(
                float(np.linalg.norm(tcp[:3] - ref_tcp[:3])) for tcp in cycle_start_tcps[1:]
            )
            print(f"cycle_start_tcp_pose (cycle 1): {ref_tcp.tolist()}")
            print(f"max_cycle_start_position_delta_m (vs cycle 1): {max_cycle_start_delta_m:.6f}")

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
