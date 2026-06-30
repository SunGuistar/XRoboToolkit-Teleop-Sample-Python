#!/usr/bin/env python3
"""通过 RTDE getActualTCPPose() 记录当前 TCP 位姿到 JSON。

只使用 RTDE receive（只读），不占用 RTDE control。
每条记录为 getActualTCPPose() 的原始返回值 [x,y,z,rx,ry,rz]。

Usage:
  python scripts/hardware/record_tcp_pose.py
  python scripts/hardware/record_tcp_pose.py --pose-name POSE_2
  python scripts/hardware/record_tcp_pose.py --pose-file scripts/hardware/tool_tcp_rotate.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rtde_receive
import tyro

from xrobotoolkit_teleop.hardware.interface.universal_robots import DEFAULT_ROBOT_IP

DEFAULT_POSE_FILE = Path(__file__).resolve().parent / "tool_tcp_rotate.json"
DEFAULT_POSE_NOTE = (
    "getActualTCPPose() return values: [x,y,z,rx,ry,rz] in meters + axis-angle (rad)."
)


def next_pose_name(poses: dict) -> str:
    index = 1
    while f"POSE_{index}" in poses:
        index += 1
    return f"POSE_{index}"


def load_pose_file(path: Path) -> dict:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    return {
        "robot_ip": DEFAULT_ROBOT_IP,
        "note": DEFAULT_POSE_NOTE,
        "poses": {},
    }


def save_pose_file(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def main(
    robot_ip: str = DEFAULT_ROBOT_IP,
    pose_name: str | None = None,
    pose_file: Path = DEFAULT_POSE_FILE,
) -> None:
    rtde_r: rtde_receive.RTDEReceiveInterface | None = None

    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)

        tcp_pose = [float(v) for v in rtde_r.getActualTCPPose()]

        data = load_pose_file(pose_file)
        data["robot_ip"] = robot_ip
        poses = data.setdefault("poses", {})

        name = pose_name or next_pose_name(poses)
        if name in poses:
            raise RuntimeError(f"Pose name '{name}' already exists in {pose_file}")

        poses[name] = tcp_pose
        save_pose_file(pose_file, data)

        print("=== Record TCP Pose ===")
        print(f"robot_ip: {robot_ip}")
        print("source: getActualTCPPose()")
        print(f"saved_as: {name}")
        print(f"pose_file: {pose_file}")
        print(f"value: {tcp_pose}")
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
