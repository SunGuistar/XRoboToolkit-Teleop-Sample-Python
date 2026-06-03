#!/usr/bin/env python3
"""Stream Intel RealSense color video to a PICO VR headset (XRoboToolkit).

Protocol: TCP, 4-byte big-endian length + H.264 Annex-B NALU (same as RobotVisionConsole).

USB PICO (typical):
  - Headset IP: 192.168.3.198  (opens port 12345 only while Listen is active)
  - PC / PC-Service IP: 192.168.181.64  (enter as "camera IP" on the headset)

  Order matters — do NOT tap Listen first (it times out during camera warmup):
  1. PC: python scripts/visualization/realsense_vr_stream.py --ip 192.168.3.198
  2. When the terminal shows "waiting for Listen", on PICO: ZEDMINI → Listen,
     camera IP = 192.168.181.64

Wi‑Fi PICO: use --ip <headset Wi‑Fi IP> instead of 192.168.3.198.

Fallback server mode (rare): --mode server

ZEDMINI needs side-by-side (default): same full image in left & right halves of the
stream — one per eye. Plain --mono often splits one frame across eyes (dizzy/blur).

VR "big screen": use --layout screen (letterboxed panel per eye), then on PICO open
Remote Vision (ZEDMINI → Listen) and press **B** on the right controller to toggle
2D / stereo view. The video panel is a floating window in the scene (Unity client).

Quality: --quality balanced|high|low  (or --width/--height/--bitrate).

Requires: pyrealsense2, opencv-python, av, numpy
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import time

import av
import cv2
import numpy as np
import pyrealsense2 as rs
from fractions import Fraction

QUALITY_PRESETS = {
    "low": {"width": 640, "height": 480, "bitrate": 2_000_000, "preset": "ultrafast"},
    "balanced": {"width": 1280, "height": 720, "bitrate": 6_000_000, "preset": "veryfast"},
    "high": {"width": 1920, "height": 1080, "bitrate": 10_000_000, "preset": "fast"},
}


def list_devices() -> list[tuple[str, str]]:
    devices = []
    for dev in rs.context().query_devices():
        devices.append((dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number)))
    return devices


def get_network_interfaces() -> list[tuple[str, str, str]]:
    """Return (interface_name, ip, kind_hint) for each non-loopback IPv4 address."""
    import re
    import subprocess

    entries: list[tuple[str, str, str]] = []
    try:
        out = subprocess.check_output(["ip", "-4", "-o", "addr", "show"], text=True)
        for line in out.splitlines():
            # lo               inet 127.0.0.1/8 scope host lo
            m = re.match(r"(\S+)\s+inet\s+([\d.]+)/", line)
            if not m:
                continue
            iface, ip = m.group(1), m.group(2)
            if ip.startswith("127."):
                continue
            if iface.startswith("wl"):
                hint = "Wi‑Fi — use this if PICO is on wireless LAN"
            elif iface.startswith(("en", "eth")):
                hint = "wired/USB — only if headset is on this same subnet"
            else:
                hint = "check same subnet as headset"
            entries.append((iface, ip, hint))
    except (OSError, subprocess.CalledProcessError):
        pass

    if not entries:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("8.8.8.8", 80))
                entries.append(("default", s.getsockname()[0], "default route"))
        except OSError:
            pass
    return entries


def print_usb_pico_hint(pc_service_ip: str, headset_ip: str, port: int) -> None:
    print("\n=== USB PICO — correct order ===")
    print("  1. Keep this script running on the PC (you already did).")
    print("  2. When you see 'waiting for Listen' below, THEN on the headset:")
    print(f"     ZEDMINI → Listen → camera IP = {pc_service_ip}")
    print(f"  3. PC will connect to {headset_ip}:{port} and start sending video.")
    print("  Do NOT tap Listen before the script — Listen will time out.")
    print("  Stream: side-by-side (default) or --layout screen for a TV-style panel.")
    print("  In VR: press **B** (right controller) to switch 2D / stereo display.")
    print("===\n")


def wait_for_headset_connection(
    host: str,
    port: int,
    pc_service_ip: str,
    timeout_s: float = 120.0,
) -> socket.socket:
    """Retry until PICO opens port 12345 (Listen mode), or timeout."""
    print(f">>> On PICO: tap **Listen** now (camera IP = {pc_service_ip}).")
    print(f">>> Waiting up to {timeout_s:.0f}s for {host}:{port} ...\n")

    deadline = time.perf_counter() + timeout_s
    attempt = 0
    while time.perf_counter() < deadline:
        attempt += 1
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((host, port))
            sock.settimeout(None)
            tune_socket(sock)
            print(f"[client] Connected to headset (attempt {attempt}).")
            return sock
        except OSError:
            if attempt == 1 or attempt % 5 == 0:
                print(f"  ... waiting for Listen on PICO (attempt {attempt})")
            time.sleep(1.0)

    raise ConnectionError(
        f"Could not connect to {host}:{port} after {timeout_s:.0f}s. "
        "On PICO: ZEDMINI → Listen, camera IP = "
        f"{pc_service_ip}, then run this script again."
    )


def print_pc_ip_hint(port: int) -> None:
    ifaces = get_network_interfaces()
    print("\n=== Server mode: enter one of these on headset as 'camera IP' ===")
    for iface, ip, hint in ifaces:
        print(f"  {ip:<18}  [{iface}]  {hint}")
    print(f"Listening on 0.0.0.0:{port}")
    print("===\n")


def _letterbox_pane(color_bgr: np.ndarray, pane_w: int, pane_h: int) -> np.ndarray:
    """Scale image to fit inside pane_w×pane_h, centered on black (TV-style)."""
    h, w = color_bgr.shape[:2]
    scale = min(pane_w / w, pane_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(color_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((pane_h, pane_w, 3), dtype=np.uint8)
    y0 = (pane_h - new_h) // 2
    x0 = (pane_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def prepare_frame(color_bgr: np.ndarray, stereo_sbs: bool, layout: str = "sbs") -> np.ndarray:
    if not stereo_sbs:
        return color_bgr
    pane_h, pane_w = color_bgr.shape[:2]
    if layout == "screen":
        pane = _letterbox_pane(color_bgr, pane_w, pane_h)
        return np.hstack((pane, pane))
    return np.hstack((color_bgr, color_bgr))


def create_h264_encoder(
    width: int,
    height: int,
    fps: int,
    bitrate: int,
    x264_preset: str = "veryfast",
) -> av.codec.CodecContext:
    codec = av.CodecContext.create("libx264", "w")
    codec.width = width
    codec.height = height
    codec.pix_fmt = "yuv420p"
    codec.time_base = Fraction(1, fps)
    codec.framerate = Fraction(fps, 1)
    codec.bit_rate = bitrate
    codec.gop_size = fps
    codec.max_b_frames = 0
    codec.options = {
        "preset": x264_preset,
        "tune": "zerolatency",
        "profile": "baseline",
        "x264-params": "annexb=1:repeat-headers=1",
    }
    codec.open()
    return codec


def send_h264_packet(sock: socket.socket, packet: bytes) -> None:
    sock.sendall(struct.pack(">I", len(packet)) + packet)


def tune_socket(sock: socket.socket) -> None:
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)


def accept_tcp_server(bind_host: str, port: int) -> socket.socket:
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((bind_host, port))
    server.listen(1)
    print(f"[server] Waiting for headset on {bind_host}:{port} ...")
    print_pc_ip_hint(port)
    conn, peer = server.accept()
    server.close()
    tune_socket(conn)
    print(f"[server] Headset connected from {peer[0]}:{peer[1]}")
    return conn


def warmup_camera(
    serial: str | None,
    capture_width: int,
    capture_height: int,
    fps: int,
) -> tuple[rs.pipeline, rs.config]:
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, capture_width, capture_height, rs.format.bgr8, fps)
    pipeline.start(config)
    for _ in range(5):
        pipeline.wait_for_frames()
    return pipeline, config


def encode_and_send_frame(
    sock: socket.socket,
    codec: av.codec.CodecContext,
    color_bgr: np.ndarray,
    stereo_sbs: bool,
    layout: str = "sbs",
) -> int:
    """Encode one BGR frame and send all resulting H.264 packets. Returns packet count."""
    color_bgr = prepare_frame(color_bgr, stereo_sbs, layout)
    video_frame = av.VideoFrame.from_ndarray(color_bgr, format="bgr24").reformat(format="yuv420p")
    count = 0
    for packet in codec.encode(video_frame):
        payload = bytes(packet)
        if payload:
            send_h264_packet(sock, payload)
            count += 1
    return count


def stream_loop(
    sock: socket.socket,
    pipeline: rs.pipeline,
    codec: av.codec.CodecContext,
    fps: int,
    stereo_sbs: bool,
    layout: str,
    preview: bool,
) -> int:
    frame_interval = 1.0 / fps
    next_frame_time = time.perf_counter()
    packets_sent = 0

    try:
        while True:
            now = time.perf_counter()
            if now < next_frame_time:
                time.sleep(next_frame_time - now)
            next_frame_time += frame_interval

            frames = pipeline.wait_for_frames(timeout_ms=500)
            color = np.asanyarray(frames.get_color_frame().get_data())

            if preview:
                cv2.imshow("RealSense (preview)", prepare_frame(color, stereo_sbs, layout))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            packets_sent += encode_and_send_frame(sock, codec, color, stereo_sbs, layout)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except (BrokenPipeError, ConnectionError, OSError) as exc:
        print(f"Network error (headset disconnected?): {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            for packet in codec.encode(None):
                payload = bytes(packet)
                if payload:
                    send_h264_packet(sock, payload)
        except OSError:
            pass
        sock.close()
        cv2.destroyAllWindows()
        print(f"H.264 packets sent: {packets_sent}")

    return 0


def init_camera_pipeline(
    capture_width: int,
    capture_height: int,
    fps: int,
    serial: str | None,
) -> tuple[rs.pipeline, str]:
    devices = list_devices()
    if not devices:
        raise RuntimeError("No RealSense camera found.")

    for name, sn in devices:
        print(f"  {name}  (serial: {sn})")

    use_serial = serial or devices[0][1]
    if serial and serial not in {s for _, s in devices}:
        raise RuntimeError(f"Serial {serial} not found.")

    print("Warming up RealSense (before connecting to headset)...")
    pipeline, _ = warmup_camera(use_serial, capture_width, capture_height, fps)
    return pipeline, use_serial


def stream_to_vr(
    sock: socket.socket,
    pipeline: rs.pipeline,
    codec: av.codec.CodecContext,
    fps: int,
    stereo_sbs: bool,
    layout: str,
    preview: bool,
) -> int:
    print("Streaming... (Ctrl+C to stop)")

    # Headset closes the socket if no video arrives quickly after TCP connect.
    sent = 0
    for i in range(5):
        frames = pipeline.wait_for_frames(timeout_ms=2000)
        color = np.asanyarray(frames.get_color_frame().get_data())
        sent += encode_and_send_frame(sock, codec, color, stereo_sbs, layout)
        if sent > 0:
            break
    print(f"Sent {sent} H.264 packet(s) immediately after connect.")
    if sent == 0:
        print("Warning: encoder produced no packets; headset may disconnect.", file=sys.stderr)

    try:
        return stream_loop(sock, pipeline, codec, fps, stereo_sbs, layout, preview)
    finally:
        pipeline.stop()


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream RealSense to PICO VR (TCP/H.264).")
    parser.add_argument(
        "--mode",
        choices=("server", "client"),
        default="client",
        help="client (default): PC connects to headset. server: PC listens.",
    )
    parser.add_argument(
        "--ip",
        type=str,
        default="192.168.3.198",
        help="Headset IP (USB default: 192.168.3.198)",
    )
    parser.add_argument(
        "--pc-service-ip",
        type=str,
        default="192.168.181.64",
        help="Shown on headset as camera/PC-Service IP (for your reference only)",
    )
    parser.add_argument("--bind", type=str, default="0.0.0.0", help="Listen address for server mode")
    parser.add_argument("--port", type=int, default=12345, help="TCP port (default: 12345)")
    parser.add_argument(
        "--quality",
        choices=tuple(QUALITY_PRESETS),
        default="balanced",
        help="Resolution/bitrate trade-off (default: balanced)",
    )
    parser.add_argument("--width", type=int, default=None, help="Override capture width")
    parser.add_argument("--height", type=int, default=None, help="Override capture height")
    parser.add_argument(
        "--layout",
        choices=("sbs", "screen"),
        default="sbs",
        help="sbs: full frame per eye (default). screen: letterboxed TV panel per eye.",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Send single frame (no L|R duplicate). May look sharp one-eye-only on ZEDMINI.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frame rate (default: 30)")
    parser.add_argument("--bitrate", type=int, default=None, help="Override H.264 bitrate (bps)")
    parser.add_argument("--serial", type=str, default=None, help="RealSense serial")
    parser.add_argument("--preview", action="store_true", help="Local OpenCV preview")
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=120.0,
        help="Seconds to retry connecting while you enable Listen on PICO (default: 120)",
    )
    args = parser.parse_args()

    preset = QUALITY_PRESETS[args.quality]
    capture_width = args.width if args.width is not None else preset["width"]
    capture_height = args.height if args.height is not None else preset["height"]
    bitrate = args.bitrate if args.bitrate is not None else preset["bitrate"]
    x264_preset = preset["preset"]
    # ZEDMINI maps the left/right halves of the stream to left/right eyes.
    # Mono sends one wide image → each eye often gets half → discomfort when both open.
    stereo_sbs = not args.mono
    encode_width = capture_width * 2 if stereo_sbs else capture_width
    encode_height = capture_height

    if args.mode == "client":
        print_usb_pico_hint(args.pc_service_ip, args.ip, args.port)

    try:
        pipeline, _ = init_camera_pipeline(capture_width, capture_height, args.fps, args.serial)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if not stereo_sbs:
        layout_desc = "mono (experimental — avoid on ZEDMINI)"
    elif args.layout == "screen":
        layout_desc = "SBS + letterbox (TV-style panel per eye)"
    else:
        layout_desc = "SBS (same full image → each eye)"
    print(
        f"Quality={args.quality}  capture {capture_width}x{capture_height} @ {args.fps}fps → "
        f"encode {encode_width}x{encode_height}  bitrate={bitrate // 1_000_000}Mbps  ({layout_desc})"
    )
    codec = create_h264_encoder(encode_width, encode_height, args.fps, bitrate, x264_preset)

    try:
        if args.mode == "client":
            sock = wait_for_headset_connection(
                args.ip, args.port, args.pc_service_ip, timeout_s=args.connect_timeout
            )
        else:
            sock = accept_tcp_server(args.bind, args.port)
    except ConnectionError as exc:
        print(exc, file=sys.stderr)
        pipeline.stop()
        return 1

    return stream_to_vr(
        sock=sock,
        pipeline=pipeline,
        codec=codec,
        fps=args.fps,
        stereo_sbs=stereo_sbs,
        layout=args.layout,
        preview=args.preview,
    )


if __name__ == "__main__":
    raise SystemExit(main())
