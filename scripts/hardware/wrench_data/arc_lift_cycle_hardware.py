#!/usr/bin/env python3
"""Repeat arc rotate -> lift -> reverse arc -> lower on UR5 hardware.

Each cycle:
  0. moveL to session home TCP (updated in-script; drops every N cycles)
  1. Rotate -angle_deg via movePath (multi-waypoint arc)
  2. moveL up by lift_m (world Z)
  3. Rotate +angle_deg via direct moveL to arc endpoint
  4. moveL down by lift_m

On start: moveJ to home.json once, then record initial/session home in memory.
Every 6 cycles session home Z drops by 3 mm (world Z). --reset returns to initial home.

Fast defaults: neg arc uses movePath (multi-waypoint); pos arc uses direct moveL.

Usage:
  python scripts/hardware/wrench_data/arc_lift_cycle_hardware.py
  python scripts/hardware/wrench_data/arc_lift_cycle_hardware.py --cycles 48 --angle-deg 60 --lift-m 0.05
  python scripts/hardware/wrench_data/arc_lift_cycle_hardware.py --reset
  python scripts/hardware/wrench_data/arc_lift_cycle_hardware.py --dry-run
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro

from xrobotoolkit_teleop.hardware.interface.universal_robots import connect_rtde, DEFAULT_ROBOT_IP

sys.path.insert(0, str(Path(__file__).resolve().parent))


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


import arc_motion

arc_motion.matrix_to_rotvec = rotation_matrix_to_rotvec

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


@dataclass
class SessionHome:
    initial_home_q: np.ndarray
    initial_home_tcp: np.ndarray
    session_home_tcp: np.ndarray


def record_session_home(rtde_r) -> tuple[np.ndarray, np.ndarray]:
    home_q = np.asarray(rtde_r.getActualQ(), dtype=np.float64).reshape(6)
    home_tcp = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64).reshape(6)
    return home_q, home_tcp


def max_session_z_drop_m(cycles: int, *, every_n_cycles: int, drop_m: float) -> float:
    if cycles < 1:
        return 0.0
    return float((cycles - 1) // every_n_cycles) * drop_m


def move_to_home_joints(
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


def move_to_session_home(
    rtde_c,
    rtde_r,
    home_tcp: np.ndarray,
    *,
    velocity: float,
    acceleration: float,
    label: str,
    verbose: bool,
) -> None:
    tcp = np.asarray(home_tcp, dtype=np.float64).reshape(6)
    if verbose:
        print(f"\n=== {label} ===")
        print(f"target_tcp_pose: {tcp.tolist()}")
    else:
        print(f"{label}...", end=" ", flush=True)
    if not rtde_c.moveL(tcp.tolist(), velocity, acceleration):
        raise RuntimeError(f"{label}: moveL to session home failed.")
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

    if steps <= 1:
        target_pose = plan.pose_end
        if verbose:
            print("mode: direct moveL to arc endpoint (steps<=1)")
            print(f"target_tcp_pose: {target_pose.tolist()}")
        if not rtde_c.moveL(target_pose.tolist(), velocity, acceleration):
            raise RuntimeError(f"{label}: moveL to arc endpoint failed.")
    else:
        waypoints = generate_arc_waypoints(plan, steps=steps)
        movel_path = build_movel_path(waypoints, velocity, acceleration, blend)
        if verbose:
            print(f"mode: movePath, waypoints: {len(waypoints)}, blend_m: {blend:.4f} (interior), 0.0 (final)")
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
    acceleration: float = 1.2,
    velocity: float = 0.20,
    home_velocity: float = 1.5,
    home_acceleration: float = 2.0,
    neg_arc_steps: int = 18,
    pos_arc_steps: int = 1,
    blend: float = 0.02,
    start_z_drop_every_n_cycles: int = 6,
    start_z_drop_m: float = 0.003,
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
        file_home_q = load_home_joints(home_file, home_record_name)

        if reset:
            move_to_home_joints(
                rtde_c,
                rtde_r,
                file_home_q,
                home_velocity=home_velocity,
                home_acceleration=home_acceleration,
                label="Reset: move to file home",
                verbose=True,
            )
            return

        if cycles < 1:
            raise ValueError(f"cycles must be >= 1, got {cycles}")
        if lift_m <= 0.0:
            raise ValueError(f"lift_m must be > 0, got {lift_m}")
        if start_z_drop_every_n_cycles < 1:
            raise ValueError(f"start_z_drop_every_n_cycles must be >= 1, got {start_z_drop_every_n_cycles}")
        if start_z_drop_m < 0.0:
            raise ValueError(f"start_z_drop_m must be >= 0, got {start_z_drop_m}")
        if neg_arc_steps < 2:
            raise ValueError(f"neg_arc_steps must be >= 2 for movePath, got {neg_arc_steps}")
        if pos_arc_steps < 1:
            raise ValueError(f"pos_arc_steps must be >= 1, got {pos_arc_steps}")

        record_key, record_tcp, x_axis, y_axis = load_axis_xy(tool_tcp_file, record_name)
        print("=== Arc Lift Cycle Plan ===")
        print(f"cycles: {cycles}")
        print(f"angle_deg: -{angle_deg:.1f} / +{angle_deg:.1f}")
        print(f"lift_m: {lift_m:.4f}")
        print(f"velocity_m_s: {velocity:.3f}, acceleration_m_s2: {acceleration:.3f}")
        print(
            f"neg_arc_steps: {neg_arc_steps} (movePath), "
            f"pos_arc_steps: {pos_arc_steps} ({'direct moveL' if pos_arc_steps <= 1 else 'movePath'})"
        )
        print(f"home_velocity_rad_s: {home_velocity:.3f}, home_acceleration_rad_s2: {home_acceleration:.3f}")
        print(f"blend_m: {blend:.4f}, verbose: {verbose}")
        print(
            f"session_home_z_drop: -{start_z_drop_m * 1000.0:.1f} mm every {start_z_drop_every_n_cycles} cycles "
            f"(max -{max_session_z_drop_m(cycles, every_n_cycles=start_z_drop_every_n_cycles, drop_m=start_z_drop_m) * 1000.0:.1f} mm)"
        )
        print(f"bootstrap_home_file: {home_file}")
        print(f"bootstrap_home_record: {home_record_name or '(default)'}")

        if dry_run:
            print("Dry run: motion skipped.")
            return

        print("\n=== Record Session Home ===")
        move_to_home_joints(
            rtde_c,
            rtde_r,
            file_home_q,
            home_velocity=home_velocity,
            home_acceleration=home_acceleration,
            label="Bootstrap: move to file home",
            verbose=verbose,
        )
        initial_home_q, initial_home_tcp = record_session_home(rtde_r)
        session = SessionHome(
            initial_home_q=initial_home_q,
            initial_home_tcp=initial_home_tcp,
            session_home_tcp=initial_home_tcp.copy(),
        )
        print(f"initial_home_tcp: {session.initial_home_tcp.tolist()}")
        print(f"initial_home_q (rad): {session.initial_home_q.tolist()}")

        cycle_start_tcps: list[np.ndarray] = []
        cycle_durations_s: list[float] = []
        run_t0 = time.monotonic()

        for cycle in range(1, cycles + 1):
            cycle_t0 = time.monotonic()
            if cycle > 1 and (cycle - 1) % start_z_drop_every_n_cycles == 0:
                session.session_home_tcp[2] -= start_z_drop_m
                print(
                    f"Updated session home Z: -{(session.initial_home_tcp[2] - session.session_home_tcp[2]) * 1000.0:.1f} mm "
                    f"from initial"
                )

            print(f"\n######## Cycle {cycle}/{cycles} ########")
            print(f"session_home_tcp: {session.session_home_tcp.tolist()}")
            move_to_session_home(
                rtde_c,
                rtde_r,
                session.session_home_tcp,
                velocity=velocity,
                acceleration=acceleration,
                label=f"Cycle {cycle}: move to session home",
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
                steps=neg_arc_steps,
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
                steps=pos_arc_steps,
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

            cycle_elapsed_s = time.monotonic() - cycle_t0
            cycle_durations_s.append(cycle_elapsed_s)
            print(f"Cycle {cycle} elapsed: {cycle_elapsed_s:.1f} s")

        total_elapsed_s = time.monotonic() - run_t0
        end_tcp = np.asarray(rtde_r.getActualTCPPose(), dtype=np.float64).reshape(6)
        print("\n=== All Cycles Complete ===")
        print(f"total_elapsed_s: {total_elapsed_s:.1f}")
        if cycle_durations_s:
            print(
                f"cycle_time_s: min={min(cycle_durations_s):.1f}, "
                f"max={max(cycle_durations_s):.1f}, "
                f"avg={sum(cycle_durations_s) / len(cycle_durations_s):.1f}"
            )
        print(f"last_cycle_end_tcp_pose: {end_tcp.tolist()}")
        print(f"final_session_home_tcp: {session.session_home_tcp.tolist()}")
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
