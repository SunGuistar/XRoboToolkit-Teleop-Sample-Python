#!/usr/bin/env python3
"""Record current UR5 state as home pose into wrench_data/home.json.

Usage:
  python scripts/hardware/wrench_data/record_home.py
  python scripts/hardware/wrench_data/record_home.py --robot-ip 192.168.56.10
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rtde_receive
import tyro

from xrobotoolkit_teleop.hardware.interface.universal_robots import DEFAULT_ROBOT_IP

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arc_motion import DEFAULT_HOME_FILE

DEFAULT_NOTE = (
    "Home pose for wrench_data scripts. RTDE getActualQ/getActualTCPPose. Units: m, rad."
)


def collect_home_snapshot(rtde_r: rtde_receive.RTDEReceiveInterface) -> dict:
    q = [float(v) for v in rtde_r.getActualQ()]
    return {
        "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "note": DEFAULT_NOTE,
        "joint_q_rad": q,
        "joint_q_deg": np.degrees(np.array(q)).tolist(),
        "tcp_pose": [float(v) for v in rtde_r.getActualTCPPose()],
        "target_tcp_pose": [float(v) for v in rtde_r.getTargetTCPPose()],
    }


def save_home_file(path: Path, robot_ip: str, snapshot: dict) -> None:
    payload = {"robot_ip": robot_ip, **snapshot}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_home_file(path: Path = DEFAULT_HOME_FILE) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main(
    robot_ip: str = DEFAULT_ROBOT_IP,
    output_file: Path = DEFAULT_HOME_FILE,
) -> None:
    rtde_r: rtde_receive.RTDEReceiveInterface | None = None

    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        snapshot = collect_home_snapshot(rtde_r)
        save_home_file(output_file, robot_ip, snapshot)

        print("=== Record UR5 Home ===")
        print(f"robot_ip: {robot_ip}")
        print(f"output_file: {output_file}")
        print(f"recorded_at: {snapshot['recorded_at']}")
        print(f"joint_q_deg: {[round(v, 2) for v in snapshot['joint_q_deg']]}")
        print(f"tcp_pose: {snapshot['tcp_pose']}")
        print("Done.")

    finally:
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
