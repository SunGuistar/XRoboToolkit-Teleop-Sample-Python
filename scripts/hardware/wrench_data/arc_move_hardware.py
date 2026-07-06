#!/usr/bin/env python3
"""Execute horizontal arc motion on UR5 hardware via RTDE movePath (moveL + blend).

The rotation axis is vertical (world Z) and passes through RECORD_假定的工具TCP xy in tool_tcp.json.

Usage:
  python scripts/hardware/wrench_data/arc_move_hardware.py --angle-deg 60
  python scripts/hardware/wrench_data/arc_move_hardware.py --orientation-mode FOLLOW_ARC
  python scripts/hardware/wrench_data/arc_move_hardware.py --blend 0.012
  python scripts/hardware/wrench_data/arc_move_hardware.py --move-home-first
  python scripts/hardware/wrench_data/arc_move_hardware.py --record-waypoint-joints
  python scripts/hardware/wrench_data/arc_move_hardware.py --record-waypoint-joints --waypoint-joints-mode visit
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
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
    WRENCH_DATA_DIR,
    ArcPlan,
    OrientationMode,
    compute_arc_plan,
    compute_post_motion_errors,
    generate_arc_waypoints,
    load_axis_xy,
    print_post_motion_errors,
    print_pre_motion_info,
)
from record_home import load_home_joints, resolve_output_file

DEFAULT_WAYPOINT_JOINTS_FILE = WRENCH_DATA_DIR / "arc_waypoint_joints.json"
WAYPOINT_JOINTS_NOTE = (
    "Arc waypoint joint snapshots from UR5 hardware. RTDE getActualQ/getActualTCPPose. Units: m, rad."
)


class WaypointJointsMode(str, Enum):
    """How to capture joint angles at arc waypoints."""

    TRACE = "trace"
    VISIT = "visit"


@dataclass
class RtdeSample:
    t: float
    joint_q_rad: list[float]
    tcp_pose: list[float]


class RtdeSampler:
    """Background RTDE poller used while movePath is executing."""

    def __init__(self, rtde_r, poll_s: float = 0.008) -> None:
        self._rtde_r = rtde_r
        self._poll_s = poll_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[RtdeSample] = []

    def start(self) -> None:
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, name="rtde-waypoint-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.samples.append(
                RtdeSample(
                    t=time.monotonic(),
                    joint_q_rad=[float(v) for v in self._rtde_r.getActualQ()],
                    tcp_pose=[float(v) for v in self._rtde_r.getActualTCPPose()],
                )
            )
            time.sleep(self._poll_s)


def wait_until_stopped(rtde_r, timeout_s: float = 120.0, poll_s: float = 0.05) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        speed = np.linalg.norm(rtde_r.getActualTCPSpeed()[:3])
        if speed < 1e-3:
            return
        time.sleep(poll_s)
    raise TimeoutError("Timed out waiting for robot to stop.")


def waypoint_thetas_deg(plan: ArcPlan, steps: int) -> np.ndarray:
    return np.linspace(0.0, plan.angle_deg, max(steps, 2))[1:]


def tcp_position_error_m(actual_tcp: list[float] | np.ndarray, planned_pose: np.ndarray) -> float:
    actual = np.asarray(actual_tcp, dtype=np.float64).reshape(6)
    planned = np.asarray(planned_pose, dtype=np.float64).reshape(6)
    return float(np.linalg.norm(actual[:3] - planned[:3]))


def match_samples_to_waypoints(
    samples: list[RtdeSample],
    waypoints: list[np.ndarray],
    thetas_deg: np.ndarray,
) -> list[dict]:
    if not samples:
        raise RuntimeError("No RTDE samples collected during motion.")

    records: list[dict] = []
    for index, (planned_pose, theta_deg) in enumerate(zip(waypoints, thetas_deg, strict=True)):
        best_sample = min(samples, key=lambda sample: tcp_position_error_m(sample.tcp_pose, planned_pose))
        q_rad = best_sample.joint_q_rad
        records.append(
            {
                "index": index + 1,
                "theta_deg": float(theta_deg),
                "planned_tcp_pose": [float(v) for v in np.asarray(planned_pose, dtype=np.float64).reshape(6)],
                "joint_q_rad": q_rad,
                "joint_q_deg": np.degrees(np.asarray(q_rad, dtype=np.float64)).tolist(),
                "tcp_pose": best_sample.tcp_pose,
                "tcp_position_error_m": tcp_position_error_m(best_sample.tcp_pose, planned_pose),
                "sample_time_s": best_sample.t,
            }
        )
    return records


def collect_waypoint_snapshot(
    rtde_r,
    index: int,
    theta_deg: float,
    planned_pose: np.ndarray,
) -> dict:
    q_rad = [float(v) for v in rtde_r.getActualQ()]
    tcp_pose = [float(v) for v in rtde_r.getActualTCPPose()]
    return {
        "index": index,
        "theta_deg": float(theta_deg),
        "planned_tcp_pose": [float(v) for v in np.asarray(planned_pose, dtype=np.float64).reshape(6)],
        "joint_q_rad": q_rad,
        "joint_q_deg": np.degrees(np.asarray(q_rad, dtype=np.float64)).tolist(),
        "tcp_pose": tcp_pose,
        "target_tcp_pose": [float(v) for v in rtde_r.getTargetTCPPose()],
        "tcp_position_error_m": tcp_position_error_m(tcp_pose, planned_pose),
    }


def record_waypoints_by_visit(
    rtde_c,
    rtde_r,
    waypoints: list[np.ndarray],
    thetas_deg: np.ndarray,
    velocity: float,
    acceleration: float,
) -> list[dict]:
    records: list[dict] = []
    for index, (planned_pose, theta_deg) in enumerate(zip(waypoints, thetas_deg, strict=True), start=1):
        if not rtde_c.moveL(planned_pose.tolist(), velocity, acceleration):
            raise RuntimeError(f"moveL to waypoint {index}/{len(waypoints)} failed.")
        wait_until_stopped(rtde_r)
        records.append(collect_waypoint_snapshot(rtde_r, index, theta_deg, planned_pose))
    return records


def save_waypoint_joints_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def print_waypoint_joints_summary(records: list[dict], *, mode: WaypointJointsMode) -> None:
    print(f"=== Arc Waypoint Joint Angles ({mode.value}) ===")
    print(f"waypoints: {len(records)}")
    print("  #  theta   J1_pan  J2_lift J3_elbow   J4_w1    J5_w2    J6_w3  tcp_err_mm")
    print("-" * 78)
    max_tcp_err_m = 0.0
    for record in records:
        q_deg = record["joint_q_deg"]
        tcp_err_m = float(record["tcp_position_error_m"])
        max_tcp_err_m = max(max_tcp_err_m, tcp_err_m)
        print(
            f"{record['index']:3d} {record['theta_deg']:6.2f} "
            f"{q_deg[0]:8.2f} {q_deg[1]:8.2f} {q_deg[2]:8.2f} "
            f"{q_deg[3]:8.2f} {q_deg[4]:8.2f} {q_deg[5]:8.2f} "
            f"{tcp_err_m * 1000.0:8.2f}"
        )
    print(f"max_tcp_position_error_mm: {max_tcp_err_m * 1000.0:.2f}")


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
    record_waypoint_joints: bool = False,
    waypoint_joints_mode: WaypointJointsMode = WaypointJointsMode.TRACE,
    waypoint_joints_file: Path = DEFAULT_WAYPOINT_JOINTS_FILE,
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

        step_count = max(steps, 2)
        waypoints = generate_arc_waypoints(plan, steps=step_count)
        thetas_deg = waypoint_thetas_deg(plan, step_count)
        movel_path = build_movel_path(waypoints, velocity, acceleration, blend)
        print(f"movePath waypoints: {len(waypoints)}, blend_m: {blend:.4f} (interior), 0.0 (final)")

        if dry_run:
            if record_waypoint_joints:
                print(f"Would record {len(waypoints)} waypoint joint snapshots ({waypoint_joints_mode.value}).")
            print("Dry run: movePath skipped.")
            return

        waypoint_records: list[dict] | None = None

        if record_waypoint_joints and waypoint_joints_mode == WaypointJointsMode.VISIT:
            print(f"Recording {len(waypoints)} waypoint joint snapshots via sequential moveL stops.")
            waypoint_records = record_waypoints_by_visit(
                rtde_c,
                rtde_r,
                waypoints,
                thetas_deg,
                velocity,
                acceleration,
            )
        else:
            sampler: RtdeSampler | None = None
            if record_waypoint_joints:
                print(f"Recording {len(waypoints)} waypoint joint snapshots during movePath (trace).")
                sampler = RtdeSampler(rtde_r)
                sampler.start()

            if not rtde_c.movePath(movel_path):
                if sampler is not None:
                    sampler.stop()
                raise RuntimeError("movePath failed.")

            if sampler is not None:
                sampler.stop()
                waypoint_records = match_samples_to_waypoints(sampler.samples, waypoints, thetas_deg)

            wait_until_stopped(rtde_r)

        if waypoint_records is not None:
            output_file = resolve_output_file(waypoint_joints_file)
            payload = {
                "robot_ip": robot_ip,
                "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "note": WAYPOINT_JOINTS_NOTE,
                "record_name": record_key,
                "angle_deg": float(angle_deg),
                "orientation_mode": orientation_mode.value,
                "steps": step_count,
                "waypoint_count": len(waypoint_records),
                "capture_mode": waypoint_joints_mode.value,
                "velocity_m_s": float(velocity),
                "acceleration_m_s2": float(acceleration),
                "blend_m": float(blend),
                "waypoints": waypoint_records,
            }
            save_waypoint_joints_file(output_file, payload)
            print_waypoint_joints_summary(waypoint_records, mode=waypoint_joints_mode)
            print(f"Saved waypoint joints: {output_file}")

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
