#!/usr/bin/env python3
"""Stream Intel RealSense color video to a PICO VR headset (XRoboToolkit).

Protocol: TCP, 4-byte big-endian length + H.264 Annex-B NALU (same as RobotVisionConsole).

USB PICO (typical):
  - Headset IP: 192.168.3.198  (opens port 12345 only while Listen is active)
  - PC / PC-Service IP: 192.168.181.64  (enter as "camera IP" on the headset)

  Order matters — do NOT tap Listen first (it times out during camera warmup):
  1. PC: python scripts/visualization/realsense_vr_stream.py
     (defaults: USB headset IP, layout=screen, quality=high)
  2. When the terminal shows "waiting for Listen", on PICO: ZEDMINI → Listen,
     camera IP = 192.168.181.64

Wi‑Fi PICO: use --ip <headset Wi‑Fi IP> instead of 192.168.3.198.

Fallback server mode (rare): --mode server

ZEDMINI needs side-by-side (default): same full image in left & right halves of the
stream — one per eye. Plain --mono often splits one frame across eyes (dizzy/blur).

VR "big screen": use --layout screen (letterboxed panel per eye), then on PICO open
Remote Vision (ZEDMINI → Listen) and press **B** on the right controller to toggle
2D / stereo view. The video panel is a floating window in the scene (Unity client).

Default: layout=screen, quality=high. Override with --layout sbs, --quality balanced|low.

Dual-camera picture-in-picture (racing-game style: main view + corner inset):
  python scripts/visualization/realsense_vr_stream.py \\
    --main-serial <rear_serial> --pip-serial <side_serial> \\
    --quality balanced --pip-scale 0.28

  Press **A** on the right controller to swap main / PiP cameras (requires PC Service).

Requires: pyrealsense2, opencv-python, av, numpy
"""

from __future__ import annotations

import argparse
import socket
import struct
import sys
import threading
import time
from collections.abc import Callable

import av
import cv2
import numpy as np
import pyrealsense2 as rs
from av.video.frame import PictureType
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


def compose_pip_frame(
    main_bgr: np.ndarray,
    pip_bgr: np.ndarray,
    pip_scale: float = 0.28,
    margin: int = 16,
    border: int = 2,
) -> np.ndarray:
    """Overlay pip_bgr in the bottom-right corner of main_bgr (racing-game PiP)."""
    out = main_bgr.copy()
    frame_h, frame_w = out.shape[:2]
    pip_w = max(1, int(frame_w * pip_scale))
    pip_h = max(1, int(pip_w * pip_bgr.shape[0] / pip_bgr.shape[1]))
    pip_resized = cv2.resize(pip_bgr, (pip_w, pip_h), interpolation=cv2.INTER_AREA)

    x0 = frame_w - pip_w - margin
    y0 = frame_h - pip_h - margin
    if border > 0:
        cv2.rectangle(
            out,
            (x0 - border, y0 - border),
            (x0 + pip_w + border - 1, y0 + pip_h + border - 1),
            (255, 255, 255),
            border,
        )
    out[y0 : y0 + pip_h, x0 : x0 + pip_w] = pip_resized
    return out


class PipFrameSource:
    """Compose two camera feeds; supports swapping which camera is fullscreen vs PiP."""

    def __init__(
        self,
        main_reader: PipelineReader,
        pip_reader: PipelineReader,
        main_label: str,
        pip_label: str,
        pip_scale: float = 0.28,
        pip_margin: int = 16,
    ):
        self._main_reader = main_reader
        self._pip_reader = pip_reader
        self._main_label = main_label
        self._pip_label = pip_label
        self._pip_scale = pip_scale
        self._pip_margin = pip_margin
        self._swapped = False
        self._lock = threading.Lock()

    def toggle_swap(self) -> bool:
        with self._lock:
            self._swapped = not self._swapped
            swapped = self._swapped
        if swapped:
            print(f"[swap] fullscreen={self._pip_label}  PiP={self._main_label}")
        else:
            print(f"[swap] fullscreen={self._main_label}  PiP={self._pip_label}")
        return swapped

    def get_frame(self) -> np.ndarray:
        main_color = self._main_reader.get_latest()
        pip_color = self._pip_reader.get_latest()
        with self._lock:
            swapped = self._swapped
        if swapped:
            main_color, pip_color = pip_color, main_color
        return compose_pip_frame(main_color, pip_color, self._pip_scale, self._pip_margin)


def prepare_frame(color_bgr: np.ndarray, stereo_sbs: bool, layout: str = "sbs") -> np.ndarray:
    if not stereo_sbs:
        return color_bgr
    pane_h, pane_w = color_bgr.shape[:2]
    if layout == "screen":
        pane = _letterbox_pane(color_bgr, pane_w, pane_h)
        return np.hstack((pane, pane))
    return np.hstack((color_bgr, color_bgr))


class PipelineReader:
    """Read RealSense color frames on a background thread (avoids blocking the stream loop)."""

    def __init__(self, pipeline: rs.pipeline, name: str):
        self._pipeline = pipeline
        self._name = name
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._errors = 0
        self._thread = threading.Thread(target=self._loop, name=f"rs-{name}", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                data = np.asanyarray(color.get_data()).copy()
                with self._lock:
                    self._frame = data
            except RuntimeError as exc:
                self._errors += 1
                if self._errors == 1 or self._errors % 30 == 0:
                    print(f"Warning: {self._name} camera: {exc}", file=sys.stderr)
                time.sleep(0.01)

    def get_latest(self, timeout_s: float = 2.0) -> np.ndarray:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy()
            time.sleep(0.005)
        raise RuntimeError(f"No frame from {self._name} camera within {timeout_s:.1f}s")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


class PreviewWindow:
    """Show preview on a background thread so Wayland/GUI issues cannot block VR streaming."""

    def __init__(self, window_name: str = "RealSense (preview)"):
        self._window_name = window_name
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._quit = False
        self._enabled = True
        self._thread = threading.Thread(target=self._loop, name="preview", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            while not self._stop.is_set():
                with self._lock:
                    frame = None if self._frame is None else self._frame.copy()
                if frame is not None:
                    cv2.imshow(self._window_name, frame)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    self._quit = True
                    break
        except cv2.error as exc:
            self._enabled = False
            print(f"Preview disabled (OpenCV GUI): {exc}", file=sys.stderr)
            print("Streaming continues. Tip: omit --preview, or try QT_QPA_PLATFORM=xcb", file=sys.stderr)
        finally:
            try:
                cv2.destroyWindow(self._window_name)
            except cv2.error:
                pass

    def update(self, frame: np.ndarray) -> None:
        if self._enabled:
            with self._lock:
                self._frame = frame

    def should_quit(self) -> bool:
        return self._quit

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


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
    force_keyframe: bool = False,
) -> int:
    """Encode one BGR frame and send all resulting H.264 packets. Returns packet count."""
    color_bgr = prepare_frame(color_bgr, stereo_sbs, layout)
    video_frame = av.VideoFrame.from_ndarray(color_bgr, format="bgr24").reformat(format="yuv420p")
    if force_keyframe:
        video_frame.pict_type = PictureType.I
    count = 0
    for packet in codec.encode(video_frame):
        payload = bytes(packet)
        if payload:
            send_h264_packet(sock, payload)
            count += 1
    return count


def stream_loop(
    sock: socket.socket,
    get_frame: Callable[[], np.ndarray],
    codec: av.codec.CodecContext,
    fps: int,
    stereo_sbs: bool,
    layout: str,
    preview: PreviewWindow | None,
    before_frame: Callable[[], bool] | None = None,
) -> int:
    frame_interval = 1.0 / fps
    next_frame_time = time.perf_counter()
    packets_sent = 0
    frames_sent = 0
    stat_interval_s = 5.0
    last_stat_time = time.perf_counter()
    stat_frames = 0
    stat_packets = 0

    try:
        while True:
            now = time.perf_counter()
            if now < next_frame_time:
                time.sleep(next_frame_time - now)
            next_frame_time += frame_interval

            force_keyframe = frames_sent == 0
            if before_frame is not None:
                force_keyframe = force_keyframe or before_frame()

            color = get_frame()

            if preview is not None:
                preview.update(prepare_frame(color, stereo_sbs, layout))
                if preview.should_quit():
                    break

            frame_packets = encode_and_send_frame(
                sock, codec, color, stereo_sbs, layout, force_keyframe=force_keyframe
            )
            packets_sent += frame_packets
            frames_sent += 1
            stat_frames += 1
            stat_packets += frame_packets

            if time.perf_counter() - last_stat_time >= stat_interval_s:
                print(
                    f"[stream] {stat_frames} frames, {stat_packets} packets "
                    f"in last {stat_interval_s:.0f}s (total packets: {packets_sent})"
                )
                stat_frames = 0
                stat_packets = 0
                last_stat_time = time.perf_counter()

    except KeyboardInterrupt:
        print("\nStopped by user.")
    except RuntimeError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    except (BrokenPipeError, ConnectionError, OSError) as exc:
        print(f"Network error (headset disconnected?): {exc}", file=sys.stderr)
        return 1
    finally:
        if preview is not None:
            preview.close()
        try:
            for packet in codec.encode(None):
                payload = bytes(packet)
                if payload:
                    send_h264_packet(sock, payload)
        except OSError:
            pass
        sock.close()
        print(f"Stream finished: {frames_sent} frames, {packets_sent} H.264 packets")

    return 0


def _device_label(serial: str) -> str:
    for name, sn in list_devices():
        if sn == serial:
            return name.replace("Intel RealSense ", "")
    return serial


def _print_devices() -> list[tuple[str, str]]:
    devices = list_devices()
    if not devices:
        raise RuntimeError("No RealSense camera found.")
    for name, sn in devices:
        print(f"  {name}  (serial: {sn})")
    return devices


def pick_default_camera_serial(devices: list[tuple[str, str]]) -> str:
    """When multiple cameras are connected, prefer D435I over D435."""
    for name, serial in devices:
        if "D435I" in name.upper().replace(" ", ""):
            return serial
    return devices[0][1]


def _validate_serial(serial: str, devices: list[tuple[str, str]], label: str) -> None:
    if serial not in {s for _, s in devices}:
        raise RuntimeError(f"{label} serial {serial} not found.")


def init_camera_pipeline(
    capture_width: int,
    capture_height: int,
    fps: int,
    serial: str | None,
) -> tuple[rs.pipeline, str]:
    devices = _print_devices()
    if serial:
        _validate_serial(serial, devices, "Camera")
        use_serial = serial
    else:
        use_serial = pick_default_camera_serial(devices)
        print(f"Auto-selected camera: {_device_label(use_serial)}  (serial: {use_serial})")

    print("Warming up RealSense (before connecting to headset)...")
    pipeline, _ = warmup_camera(use_serial, capture_width, capture_height, fps)
    return pipeline, use_serial


def init_pip_camera_pipelines(
    capture_width: int,
    capture_height: int,
    fps: int,
    main_serial: str,
    pip_serial: str,
) -> tuple[rs.pipeline, rs.pipeline, str, str]:
    devices = _print_devices()
    _validate_serial(main_serial, devices, "Main")
    _validate_serial(pip_serial, devices, "PiP")
    if main_serial == pip_serial:
        raise RuntimeError("Main and PiP serial numbers must be different.")

    print("Warming up dual RealSense cameras (before connecting to headset)...")
    main_pipeline, _ = warmup_camera(main_serial, capture_width, capture_height, fps)
    pip_pipeline, _ = warmup_camera(pip_serial, capture_width, capture_height, fps)
    return main_pipeline, pip_pipeline, main_serial, pip_serial


def make_frame_source(
    main_pipeline: rs.pipeline,
    pip_pipeline: rs.pipeline | None = None,
    pip_scale: float = 0.28,
    pip_margin: int = 16,
    main_label: str = "main",
    pip_label: str = "pip",
) -> tuple[Callable[[], np.ndarray], list[PipelineReader], PipFrameSource | None]:
    main_reader = PipelineReader(main_pipeline, "main")
    readers: list[PipelineReader] = [main_reader]
    if pip_pipeline is None:

        def get_frame() -> np.ndarray:
            return main_reader.get_latest()

        return get_frame, readers, None

    pip_reader = PipelineReader(pip_pipeline, "pip")
    readers.append(pip_reader)
    pip_source = PipFrameSource(
        main_reader, pip_reader, main_label, pip_label, pip_scale, pip_margin
    )
    return pip_source.get_frame, readers, pip_source


def init_swap_on_a_button(
    pip_source: PipFrameSource,
    button: str = "A",
) -> tuple[Callable[[], bool], Callable[[], None]]:
    """Poll VR controller button via XRoboToolkit SDK; returns (before_frame, cleanup)."""
    try:
        import xrobotoolkit_sdk as xrt
    except ImportError as exc:
        raise ImportError(
            "xrobotoolkit_sdk not installed in this Python env. "
            "Run: conda activate xrobot"
        ) from exc

    button_getters = {
        "A": xrt.get_A_button,
        "B": xrt.get_B_button,
        "X": xrt.get_X_button,
        "Y": xrt.get_Y_button,
    }
    if button not in button_getters:
        raise ValueError(f"Unsupported swap button: {button}")

    xrt.init()
    print(f"XRoboToolkit SDK ready — press **{button}** to swap main / PiP.")
    prev_pressed = False
    read_button = button_getters[button]

    def before_frame() -> bool:
        nonlocal prev_pressed
        pressed = bool(read_button())
        force_keyframe = False
        if pressed and not prev_pressed:
            pip_source.toggle_swap()
            force_keyframe = True
        prev_pressed = pressed
        return force_keyframe

    def cleanup() -> None:
        xrt.close()

    return before_frame, cleanup


def stream_to_vr(
    sock: socket.socket,
    get_frame: Callable[[], np.ndarray],
    readers: list[PipelineReader],
    pipelines: list[rs.pipeline],
    codec: av.codec.CodecContext,
    fps: int,
    stereo_sbs: bool,
    layout: str,
    preview: bool,
    before_frame: Callable[[], bool] | None = None,
    on_stop: Callable[[], None] | None = None,
) -> int:
    print("Streaming... (Ctrl+C to stop)")

    # Headset closes the socket if no video arrives quickly after TCP connect.
    sent = 0
    for i in range(10):
        color = get_frame()
        sent += encode_and_send_frame(
            sock, codec, color, stereo_sbs, layout, force_keyframe=(i == 0)
        )
        if sent > 0 and i >= 2:
            break
    print(f"Sent {sent} H.264 packet(s) immediately after connect.")
    if sent == 0:
        print("Warning: encoder produced no packets; headset may disconnect.", file=sys.stderr)

    preview_win = PreviewWindow() if preview else None
    try:
        return stream_loop(
            sock, get_frame, codec, fps, stereo_sbs, layout, preview_win, before_frame=before_frame
        )
    finally:
        if on_stop is not None:
            on_stop()
        for reader in readers:
            reader.stop()
        for pipeline in pipelines:
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
        default="high",
        help="Resolution/bitrate trade-off (default: high)",
    )
    parser.add_argument("--width", type=int, default=None, help="Override capture width")
    parser.add_argument("--height", type=int, default=None, help="Override capture height")
    parser.add_argument(
        "--layout",
        choices=("sbs", "screen"),
        default="screen",
        help="screen: letterboxed TV panel per eye (default). sbs: full frame per eye.",
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Send single frame (no L|R duplicate). May look sharp one-eye-only on ZEDMINI.",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frame rate (default: 30)")
    parser.add_argument("--bitrate", type=int, default=None, help="Override H.264 bitrate (bps)")
    parser.add_argument("--serial", type=str, default=None, help="RealSense serial (single camera)")
    parser.add_argument(
        "--main-serial",
        type=str,
        default=None,
        help="Main (full-screen) camera serial for PiP mode, e.g. rear camera",
    )
    parser.add_argument(
        "--pip-serial",
        type=str,
        default=None,
        help="Picture-in-picture camera serial for PiP mode, e.g. side camera",
    )
    parser.add_argument(
        "--pip-scale",
        type=float,
        default=0.28,
        help="PiP width as fraction of main frame (default: 0.28)",
    )
    parser.add_argument(
        "--pip-margin",
        type=int,
        default=16,
        help="PiP inset margin from bottom-right corner in pixels (default: 16)",
    )
    parser.add_argument("--preview", action="store_true", help="Local OpenCV preview")
    parser.add_argument(
        "--no-swap-on-a",
        action="store_true",
        help="Disable controller-button swap of main/PiP cameras (PiP mode only)",
    )
    parser.add_argument(
        "--swap-button",
        choices=("A", "B", "X", "Y"),
        default="A",
        help="Controller button to swap main/PiP (default: A). Requires conda env xrobot + PC Service.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=120.0,
        help="Seconds to retry connecting while you enable Listen on PICO (default: 120)",
    )
    args = parser.parse_args()

    pip_mode = bool(args.main_serial or args.pip_serial)
    if pip_mode and not (args.main_serial and args.pip_serial):
        print("PiP mode requires both --main-serial and --pip-serial.", file=sys.stderr)
        return 1
    if pip_mode and args.serial:
        print("Use either --serial (single camera) or --main-serial/--pip-serial, not both.", file=sys.stderr)
        return 1
    if not 0.1 <= args.pip_scale <= 0.5:
        print("--pip-scale must be between 0.1 and 0.5.", file=sys.stderr)
        return 1

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

    main_pipeline: rs.pipeline | None = None
    pip_pipeline: rs.pipeline | None = None
    pipelines: list[rs.pipeline] = []
    readers: list[PipelineReader] = []
    pip_source: PipFrameSource | None = None
    before_frame: Callable[[], bool] | None = None
    on_stop: Callable[[], None] | None = None
    try:
        if pip_mode:
            main_pipeline, pip_pipeline, main_sn, pip_sn = init_pip_camera_pipelines(
                capture_width, capture_height, args.fps, args.main_serial, args.pip_serial
            )
            pipelines = [main_pipeline, pip_pipeline]
            get_frame, readers, pip_source = make_frame_source(
                main_pipeline,
                pip_pipeline,
                args.pip_scale,
                args.pip_margin,
                main_label=_device_label(main_sn),
                pip_label=_device_label(pip_sn),
            )
            print("Waiting for first frames from both cameras...")
            get_frame()
            print("Both cameras producing frames.")
            if not args.no_swap_on_a:
                try:
                    before_frame, on_stop = init_swap_on_a_button(pip_source, button=args.swap_button)
                except ImportError as exc:
                    print(f"Warning: controller swap unavailable — {exc}", file=sys.stderr)
                    print("  Also ensure XRoboToolkit PC Service is running.", file=sys.stderr)
                except Exception as exc:
                    print(f"Warning: controller swap unavailable ({exc})", file=sys.stderr)
        else:
            main_pipeline, use_serial = init_camera_pipeline(
                capture_width, capture_height, args.fps, args.serial
            )
            pipelines = [main_pipeline]
            get_frame, readers, _ = make_frame_source(main_pipeline)
            get_frame()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        for reader in readers:
            reader.stop()
        for pipeline in pipelines:
            pipeline.stop()
        return 1

    if not stereo_sbs:
        layout_desc = "mono (experimental — avoid on ZEDMINI)"
    elif args.layout == "screen":
        layout_desc = "SBS + letterbox (TV-style panel per eye)"
    else:
        layout_desc = "SBS (same full image → each eye)"
    if pip_mode:
        layout_desc = f"PiP main={main_sn} pip={pip_sn} scale={args.pip_scale} + {layout_desc}"
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
        for reader in readers:
            reader.stop()
        for pipeline in pipelines:
            pipeline.stop()
        return 1

    return stream_to_vr(
        sock=sock,
        get_frame=get_frame,
        readers=readers,
        pipelines=pipelines,
        codec=codec,
        fps=args.fps,
        stereo_sbs=stereo_sbs,
        layout=args.layout,
        preview=args.preview,
        before_frame=before_frame,
        on_stop=on_stop,
    )


if __name__ == "__main__":
    raise SystemExit(main())
