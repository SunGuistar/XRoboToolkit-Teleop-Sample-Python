#!/usr/bin/env python3
"""Preview Intel RealSense (e.g. D435i) color + depth streams with OpenCV.

Usage:
    python scripts/visualization/view_realsense.py
    python scripts/visualization/view_realsense.py --width 1280 --height 720

Press 'q' in the preview window to exit.

Requires: pyrealsense2, opencv-python, numpy
"""

from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np
import pyrealsense2 as rs


def list_devices() -> list[tuple[str, str]]:
    ctx = rs.context()
    devices = []
    for dev in ctx.query_devices():
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        devices.append((name, serial))
    return devices


def main() -> int:
    parser = argparse.ArgumentParser(description="Preview RealSense color and depth.")
    parser.add_argument("--width", type=int, default=640, help="Color/depth stream width")
    parser.add_argument("--height", type=int, default=480, help="Color/depth stream height")
    parser.add_argument("--fps", type=int, default=30, help="Stream frame rate")
    parser.add_argument("--serial", type=str, default=None, help="Camera serial (default: first device)")
    parser.add_argument("--no-depth", action="store_true", help="Show color only")
    args = parser.parse_args()

    devices = list_devices()
    if not devices:
        print("No RealSense camera found. Check USB connection and udev rules.", file=sys.stderr)
        return 1

    for name, serial in devices:
        print(f"  {name}  (serial: {serial})")

    serial = args.serial or devices[0][1]
    if args.serial and args.serial not in {s for _, s in devices}:
        print(f"Serial {args.serial} not found.", file=sys.stderr)
        return 1

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    if not args.no_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    print(f"Starting preview for {serial} ({args.width}x{args.height} @ {args.fps} fps). Press 'q' to quit.")
    pipeline.start(config)
    align = rs.align(rs.stream.color) if not args.no_depth else None
    window = f"RealSense {serial}"

    try:
        while True:
            frames = pipeline.wait_for_frames()
            if align is not None:
                frames = align.process(frames)

            color = np.asanyarray(frames.get_color_frame().get_data())

            if args.no_depth:
                display = color
            else:
                depth = np.asanyarray(frames.get_depth_frame().get_data())
                depth_vis = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth, alpha=0.03),
                    cv2.COLORMAP_JET,
                )
                display = np.hstack((color, depth_vis))

            cv2.imshow(window, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
