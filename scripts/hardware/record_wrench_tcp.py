#!/usr/bin/env python3
"""Record current UR5 state (joints + TCP) into wrench_data/tool_tcp.json.

Uses RTDE receive only (read-only). Appends each snapshot as a new entry;
existing records are never overwritten.

Usage:
  python scripts/hardware/record_wrench_tcp.py
  python scripts/hardware/record_wrench_tcp.py --record-name home_pose
  python scripts/hardware/record_wrench_tcp.py --label "wrench TCP, table height"
"""

from __future__ import annotations

import json
import re
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rtde_receive
import tyro

from xrobotoolkit_teleop.hardware.interface.universal_robots import DEFAULT_ROBOT_IP

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "wrench_data" / "tool_tcp.json"
DEFAULT_NOTE = (
    "Each record: RTDE getActualQ/getActualTCPPose/getTargetTCPPose/getActualTCPSpeed "
    "and URScript get_tcp_offset(). Units: m, rad."
)


def next_record_name(records: dict) -> str:
    index = 1
    while f"RECORD_{index}" in records:
        index += 1
    return f"RECORD_{index}"


def urscript_tcp_offset(robot_ip: str) -> list[float]:
    script = 'def probe():\n  textmsg("TCP_OFFSET", get_tcp_offset())\nend\n'
    with socket.create_connection((robot_ip, 30001), timeout=5) as sock:
        sock.recv(4096)
        sock.sendall((script + "\n").encode())
        time.sleep(0.5)
        data = sock.recv(65536).decode(errors="replace")
    match = re.search(r"TCP_OFFSETp\[([^\]]+)\]", data)
    if not match:
        return [0.0] * 6
    return [float(x) for x in match.group(1).split(",")]


def _snapshot_from_flat(data: dict) -> dict | None:
    """Convert legacy single-snapshot JSON to one record entry."""
    if "tcp_pose" not in data:
        return None
    return {
        "recorded_at": data.get("recorded_at"),
        "label": data.get("label"),
        "joint_q_rad": data.get("joint_q_rad"),
        "joint_q_deg": data.get("joint_q_deg"),
        "tcp_pose": data["tcp_pose"],
        "target_tcp_pose": data.get("target_tcp_pose"),
        "tcp_offset": data.get("tcp_offset"),
        "tcp_speed": data.get("tcp_speed"),
    }


def load_output_file(path: Path) -> dict:
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
        "records": {"RECORD_1": snapshot},
    }


def save_output_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def collect_snapshot(rtde_r: rtde_receive.RTDEReceiveInterface, robot_ip: str) -> dict:
    q = [float(v) for v in rtde_r.getActualQ()]
    return {
        "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "joint_q_rad": q,
        "joint_q_deg": np.degrees(np.array(q)).tolist(),
        "tcp_pose": [float(v) for v in rtde_r.getActualTCPPose()],
        "target_tcp_pose": [float(v) for v in rtde_r.getTargetTCPPose()],
        "tcp_offset": urscript_tcp_offset(robot_ip),
        "tcp_speed": [float(v) for v in rtde_r.getActualTCPSpeed()],
    }


def main(
    robot_ip: str = DEFAULT_ROBOT_IP,
    record_name: str | None = None,
    label: str | None = None,
    output_file: Path = DEFAULT_OUTPUT,
) -> None:
    rtde_r: rtde_receive.RTDEReceiveInterface | None = None

    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
        snapshot = collect_snapshot(rtde_r, robot_ip)
        if label:
            snapshot["label"] = label

        data = load_output_file(output_file)
        data["robot_ip"] = robot_ip
        records = data.setdefault("records", {})

        name = record_name or next_record_name(records)
        if name in records:
            raise RuntimeError(f"Record name '{name}' already exists in {output_file}")

        records[name] = snapshot
        save_output_file(output_file, data)

        print("=== Record UR5 Wrench/TCP Snapshot ===")
        print(f"robot_ip: {robot_ip}")
        print(f"saved_as: {name}")
        print(f"output_file: {output_file}")
        print(f"recorded_at: {snapshot['recorded_at']}")
        if label:
            print(f"label: {label}")
        print(f"joint_q_deg: {[round(v, 2) for v in snapshot['joint_q_deg']]}")
        print(f"tcp_pose: {snapshot['tcp_pose']}")
        print(f"tcp_offset: {snapshot['tcp_offset']}")
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
