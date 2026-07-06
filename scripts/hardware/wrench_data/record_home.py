#!/usr/bin/env python3
"""Record current UR5 state as home pose into wrench_data/home.json.

Appends each snapshot as a new entry; existing records are never overwritten.

Usage:
  python scripts/hardware/wrench_data/record_home.py
  python scripts/hardware/wrench_data/record_home.py --robot-ip 192.168.56.10
  python scripts/hardware/wrench_data/record_home.py --record-name HOME_arc_start
  python scripts/hardware/wrench_data/record_home.py --output-file toolCenter.json
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
from arc_motion import DEFAULT_HOME_FILE, WRENCH_DATA_DIR, _resolve_record_key

DEFAULT_NOTE = (
    "Home pose for wrench_data scripts. RTDE getActualQ/getActualTCPPose. Units: m, rad."
)


def resolve_output_file(path: Path) -> Path:
    """Resolve relative output paths against scripts/hardware/wrench_data/."""
    if path.is_absolute():
        return path
    relative = path
    if path.parts and path.parts[0] == "wrench_data" and len(path.parts) > 1:
        relative = Path(*path.parts[1:])
    return WRENCH_DATA_DIR / relative


def next_record_name(records: dict) -> str:
    index = 1
    while f"HOME_{index}" in records:
        index += 1
    return f"HOME_{index}"


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


def _snapshot_from_flat(data: dict) -> dict | None:
    """Convert legacy single-snapshot JSON to one record entry."""
    if "joint_q_rad" not in data:
        return None
    return {
        "recorded_at": data.get("recorded_at"),
        "note": data.get("note", DEFAULT_NOTE),
        "joint_q_rad": data["joint_q_rad"],
        "joint_q_deg": data.get("joint_q_deg"),
        "tcp_pose": data.get("tcp_pose"),
        "target_tcp_pose": data.get("target_tcp_pose"),
    }


def load_home_file(path: Path = DEFAULT_HOME_FILE) -> dict:
    path = resolve_output_file(path)
    if not path.exists():
        return {
            "robot_ip": DEFAULT_ROBOT_IP,
            "note": DEFAULT_NOTE,
            "records": {},
        }

    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    if "records" in data:
        data.setdefault("robot_ip", DEFAULT_ROBOT_IP)
        data.setdefault("note", DEFAULT_NOTE)
        data["records"] = dict(data["records"])
        return data

    snapshot = _snapshot_from_flat(data)
    if snapshot is None:
        return {
            "robot_ip": data.get("robot_ip", DEFAULT_ROBOT_IP),
            "note": data.get("note", DEFAULT_NOTE),
            "records": {},
        }

    return {
        "robot_ip": data.get("robot_ip", DEFAULT_ROBOT_IP),
        "note": data.get("note", DEFAULT_NOTE),
        "records": {"HOME_1": snapshot},
    }


def load_home_record(path: Path = DEFAULT_HOME_FILE, record_name: str | None = None) -> dict:
    data = load_home_file(path)
    records = data.get("records", {})
    if not records:
        raise KeyError(f"No home records in {path}")

    if record_name is None:
        if len(records) == 1:
            return next(iter(records.values()))
        key = _resolve_record_key(records, "HOME_1")
        return records[key]

    key = _resolve_record_key(records, record_name)
    return records[key]


def load_home_joints(path: Path = DEFAULT_HOME_FILE, record_name: str | None = None) -> np.ndarray:
    record = load_home_record(path, record_name)
    if "joint_q_rad" not in record:
        raise KeyError(f"Home record missing joint_q_rad: {path}")
    return np.asarray(record["joint_q_rad"], dtype=np.float64).reshape(6)


def save_home_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def main(
    robot_ip: str = DEFAULT_ROBOT_IP,
    record_name: str | None = None,
    output_file: Path = DEFAULT_HOME_FILE,
) -> None:
    rtde_r: rtde_receive.RTDEReceiveInterface | None = None

    try:
        output_file = resolve_output_file(output_file)
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        snapshot = collect_home_snapshot(rtde_r)

        data = load_home_file(output_file)
        data["robot_ip"] = robot_ip
        records = data.setdefault("records", {})

        name = record_name or next_record_name(records)
        if name in records:
            raise RuntimeError(f"Record name '{name}' already exists in {output_file}")

        records[name] = snapshot
        save_home_file(output_file, data)

        print("=== Record UR5 Home ===")
        print(f"robot_ip: {robot_ip}")
        print(f"saved_as: {name}")
        print(f"output_file: {output_file}")
        print(f"recorded_at: {snapshot['recorded_at']}")
        print(f"joint_q_deg: {[round(v, 2) for v in snapshot['joint_q_deg']]}")
        print(f"tcp_pose: {snapshot['tcp_pose']}")
        print(f"total_records: {len(records)}")
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
