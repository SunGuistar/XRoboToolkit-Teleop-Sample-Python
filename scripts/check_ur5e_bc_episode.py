#!/usr/bin/env python3
"""
Replay and inspect a raw UR5e BC episode (.pkl).

Usage:
  python scripts/check_ur5e_bc_episode.py logs/ur5e_bc/raw/episode_00000.pkl
  python scripts/check_ur5e_bc_episode.py logs/ur5e_bc/raw/episode_00000.pkl --no-gui
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib.pyplot as plt

    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

from xrobotoolkit_teleop.utils.image_utils import decompress_jpg_to_image


def load_episode(path: Path) -> tuple[dict, list[dict]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "metadata" not in data or "frames" not in data:
        raise ValueError(f"Invalid episode format: {path}")
    return data["metadata"], data["frames"]


def decode_image_main(frame: dict) -> np.ndarray | None:
    img = frame.get("image_main")
    if img is None:
        return None
    if isinstance(img, bytes):
        bgr = decompress_jpg_to_image(img)
    elif isinstance(img, np.ndarray):
        bgr = img
    else:
        return None
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_timing_stats(frames: list[dict]) -> dict:
    if len(frames) < 2:
        return {"fps_mean": 0.0, "fps_std": 0.0, "dt_mean": 0.0, "dt_std": 0.0}
    timestamps = np.array([f["timestamp"] for f in frames], dtype=np.float64)
    dts = np.diff(timestamps)
    dts = dts[dts > 0]
    if len(dts) == 0:
        return {"fps_mean": 0.0, "fps_std": 0.0, "dt_mean": 0.0, "dt_std": 0.0}
    return {
        "dt_mean": float(np.mean(dts)),
        "dt_std": float(np.std(dts)),
        "fps_mean": float(1.0 / np.mean(dts)),
        "fps_std": float(np.std(1.0 / dts)),
    }


def print_summary(metadata: dict, frames: list[dict], timing: dict) -> None:
    active_ratio = float(np.mean([f["control_active"] for f in frames])) if frames else 0.0
    print("=" * 60)
    print(f"Episode index : {metadata.get('episode_index')}")
    print(f"Success       : {metadata.get('success')}")
    print(f"Task          : {metadata.get('task')}")
    print(f"Num frames    : {metadata.get('num_frames', len(frames))}")
    print(f"Duration      : {metadata.get('duration', 0.0):.2f}s")
    print(f"FPS (nominal) : {metadata.get('fps')}")
    print(f"FPS (measured): {timing['fps_mean']:.2f} ± {timing['fps_std']:.2f}")
    print(f"dt (measured) : {timing['dt_mean']:.4f}s ± {timing['dt_std']:.4f}s")
    print(f"control_active ratio: {active_ratio:.1%}")
    print(f"Image format  : {metadata.get('image_format')}")
    print(f"Action layout : {metadata.get('action_layout', 'target_q(6)+gripper_cmd(1)')}")
    print("=" * 60)


def plot_signals(frames: list[dict], metadata: dict) -> None:
    t = np.array([f["timestamp"] for f in frames], dtype=np.float64)
    qpos = np.stack([f["qpos"] for f in frames])
    target_q = np.stack([f["target_q"] for f in frames])
    gripper_state = np.array([f["gripper_state"] for f in frames])
    gripper_cmd = np.array([f["gripper_cmd"] for f in frames])
    active = np.array([f["control_active"] for f in frames], dtype=bool)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Episode {metadata.get('episode_index')} — success={metadata.get('success')}")

    for j in range(6):
        axes[0].plot(t, qpos[:, j], label=f"q{j}")
    axes[0].set_ylabel("qpos (rad)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.3)

    for j in range(6):
        axes[1].plot(t, target_q[:, j], label=f"tq{j}")
    axes[1].set_ylabel("target_q (rad)")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, gripper_state, label="gripper_state")
    axes[2].plot(t, gripper_cmd, label="gripper_cmd", linestyle="--")
    axes[2].set_ylabel("gripper [0=open,1=close]")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    axes[3].fill_between(t, 0, 1, where=active, alpha=0.3, color="green", label="control_active")
    axes[3].set_ylabel("active")
    axes[3].set_xlabel("timestamp (s)")
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def replay_images(frames: list[dict], fps: float) -> None:
    delay_ms = max(1, int(1000.0 / fps)) if fps > 0 else 100
    window = "BC Episode Replay (q=quit, space=pause)"
    paused = False

    for i, frame in enumerate(frames):
        rgb = decode_image_main(frame)
        if rgb is None:
            continue
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        label = (
            f"frame {i}/{len(frames)-1}  active={frame['control_active']}  "
            f"grip_cmd={frame['gripper_cmd']:.2f}"
        )
        cv2.putText(bgr, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(window, bgr)
        key = cv2.waitKey(0 if paused else delay_ms) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused

    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw UR5e BC episode pickle.")
    parser.add_argument("episode_path", type=Path, help="Path to episode_XXXXX.pkl")
    parser.add_argument("--no-gui", action="store_true", help="Skip image replay and plots")
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib plots")
    parser.add_argument("--no-video", action="store_true", help="Skip image replay")
    args = parser.parse_args()

    if not args.episode_path.exists():
        print(f"File not found: {args.episode_path}", file=sys.stderr)
        raise SystemExit(1)

    metadata, frames = load_episode(args.episode_path)
    if not frames:
        print("Episode has no frames.", file=sys.stderr)
        raise SystemExit(1)

    timing = compute_timing_stats(frames)
    print_summary(metadata, frames, timing)

    if args.no_gui:
        return

    if not args.no_plot:
        if _HAS_MPL:
            plot_signals(frames, metadata)
        else:
            print("matplotlib not installed — skipping plots (pip install matplotlib)")

    if not args.no_video:
        replay_images(frames, float(metadata.get("fps", 10)))


if __name__ == "__main__":
    main()
