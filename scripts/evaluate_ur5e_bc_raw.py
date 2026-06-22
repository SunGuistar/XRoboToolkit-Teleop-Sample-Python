#!/usr/bin/env python3
"""
Batch quality evaluation for raw UR5e BC episode pickles.

Checks timing, control coverage, gripper pick-place pattern, joint/action quality,
and frozen-camera detection (main vs pip).

Usage:
  python scripts/evaluate_ur5e_bc_raw.py
  python scripts/evaluate_ur5e_bc_raw.py --raw-dir logs/ur5e_bc/raw
  python scripts/evaluate_ur5e_bc_raw.py --json-out logs/ur5e_bc/eval_report.json
  python scripts/evaluate_ur5e_bc_raw.py --fast   # sample images (faster)
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from xrobotoolkit_teleop.utils.image_utils import decompress_jpg_to_image


@dataclass
class EpisodeReport:
    file: str
    episode_index: int
    success: bool
    task: str
    num_frames: int
    duration_s: float
    fps_nominal: float
    fps_measured: float
    active_ratio: float
    idle_start: int
    idle_end: int
    tracking_err_mean: float
    tracking_err_max: float
    joint_motion_rad: float
    gripper_transitions: int
    gripper_pattern: str
    main_frozen_ratio: float
    pip_frozen_ratio: float | None
    main_shape: str | None
    pip_shape: str | None
    timing_gaps_gt_150ms: int
    grade: str
    issues: list[str] = field(default_factory=list)
    trainable_main: bool = False
    trainable_pip: bool = False


def load_episode(path: Path) -> tuple[dict, list[dict]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "metadata" not in data or "frames" not in data:
        raise ValueError(f"Invalid episode format: {path}")
    return data["metadata"], data["frames"]


def decode_frame_image(frame: dict, key: str) -> np.ndarray | None:
    img = frame.get(key)
    if img is None:
        return None
    if isinstance(img, bytes):
        return decompress_jpg_to_image(img)
    if isinstance(img, np.ndarray):
        return img
    return None


def compute_timing_stats(timestamps: np.ndarray) -> dict[str, float | int]:
    if len(timestamps) < 2:
        return {"fps_mean": 0.0, "dt_mean": 0.0, "dt_std": 0.0, "dt_max": 0.0, "gaps_gt_150ms": 0}
    dts = np.diff(timestamps)
    dts = dts[dts > 0]
    if len(dts) == 0:
        return {"fps_mean": 0.0, "dt_mean": 0.0, "dt_std": 0.0, "dt_max": 0.0, "gaps_gt_150ms": 0}
    return {
        "dt_mean": float(np.mean(dts)),
        "dt_std": float(np.std(dts)),
        "dt_max": float(np.max(dts)),
        "fps_mean": float(1.0 / np.mean(dts)),
        "gaps_gt_150ms": int(np.sum(dts > 0.15)),
    }


def analyze_image_stream(
    frames: list[dict],
    key: str,
    *,
    fast: bool,
    frozen_threshold: float = 0.5,
) -> tuple[float, str | None]:
    """Return (frozen_pair_ratio, shape_str). frozen_pair_ratio = fraction of consecutive pairs with no change."""
    indices = _sample_indices(len(frames), fast=fast)
    prev: np.ndarray | None = None
    frozen_pairs = 0
    compared = 0
    shape_str: str | None = None

    for i in indices:
        img = decode_frame_image(frames[i], key)
        if img is None:
            continue
        if shape_str is None:
            shape_str = str(tuple(img.shape))
        if prev is not None:
            diff = float(np.mean(np.abs(img.astype(np.float32) - prev.astype(np.float32))))
            compared += 1
            if diff < frozen_threshold:
                frozen_pairs += 1
        prev = img

    ratio = float(frozen_pairs / compared) if compared else 1.0
    return ratio, shape_str


def _sample_indices(n: int, *, fast: bool) -> list[int]:
    if n == 0:
        return []
    if not fast or n <= 20:
        return list(range(n))
    # Always include start/mid/end plus evenly spaced samples.
    mids = np.linspace(0, n - 1, num=min(20, n), dtype=int)
    return sorted(set(mids.tolist()))


def analyze_gripper(gripper_cmd: np.ndarray, timestamps: np.ndarray, active: np.ndarray) -> tuple[int, str]:
    closed = gripper_cmd > 0.5
    transitions = int(np.sum(np.diff(closed.astype(int)) != 0))

    has_close = bool(np.any(closed))
    has_reopen = False
    if has_close:
        first_close = int(np.argmax(closed))
        if np.any(~closed[first_close:]):
            has_reopen = True

    if has_close and has_reopen:
        pattern = "pick+place"
    elif has_close:
        pattern = "grasp_only"
    else:
        pattern = "no_grasp"

    return transitions, pattern


def grade_episode(
    *,
    success: bool,
    num_frames: int,
    active_ratio: float,
    gripper_pattern: str,
    main_frozen_ratio: float,
    pip_frozen_ratio: float | None,
    has_pip: bool,
    joint_motion_rad: float,
    idle_end: int,
    fps_measured: float,
    timing_gaps: int,
) -> tuple[str, list[str], bool, bool]:
    issues: list[str] = []

    if not success:
        issues.append("marked failed")
    if num_frames < 30:
        issues.append(f"too few frames ({num_frames})")
    if active_ratio < 0.5:
        issues.append(f"low active ratio ({active_ratio:.0%})")
    if gripper_pattern != "pick+place":
        issues.append(f"gripper pattern: {gripper_pattern}")
    if joint_motion_rad < 1.0:
        issues.append(f"minimal joint motion ({joint_motion_rad:.2f} rad)")
    if fps_measured < 8.0:
        issues.append(f"low FPS ({fps_measured:.1f})")
    if timing_gaps > 40:
        issues.append(f"many timing gaps >150ms ({timing_gaps})")
    if idle_end > 20:
        issues.append(f"long idle tail ({idle_end} frames)")

    # Static-pair ratio: high values mean the camera stream stopped updating.
    # ~15% is normal when the arm pauses at grasp/place; >=50% is suspicious; >=90% is broken.
    main_ok = main_frozen_ratio < 0.50
    pip_ok = pip_frozen_ratio is not None and pip_frozen_ratio < 0.50

    if main_frozen_ratio >= 0.90:
        issues.append(f"main camera frozen ({main_frozen_ratio:.0%} static pairs)")
    elif main_frozen_ratio >= 0.50:
        issues.append(f"main camera mostly static ({main_frozen_ratio:.0%} static pairs)")

    if has_pip and pip_frozen_ratio is not None and pip_frozen_ratio >= 0.90:
        issues.append(f"pip camera frozen ({pip_frozen_ratio:.0%} static pairs)")
    elif has_pip and pip_frozen_ratio is not None and pip_frozen_ratio >= 0.50:
        issues.append(f"pip camera mostly static ({pip_frozen_ratio:.0%} static pairs)")

    base_ok = success and num_frames >= 30 and active_ratio >= 0.5 and gripper_pattern == "pick+place"
    trainable_main = base_ok and main_ok
    trainable_pip = base_ok and pip_ok

    if trainable_main:
        grade = "A"
    elif trainable_pip and has_pip:
        grade = "C"
    elif base_ok:
        grade = "B"
    elif num_frames >= 10 and joint_motion_rad >= 0.5:
        grade = "D"
    else:
        grade = "F"

    return grade, issues, trainable_main, trainable_pip


def analyze_episode(path: Path, *, fast: bool) -> EpisodeReport:
    metadata, frames = load_episode(path)
    if not frames:
        raise ValueError(f"Episode has no frames: {path}")

    timestamps = np.array([f["timestamp"] for f in frames], dtype=np.float64)
    active = np.array([f["control_active"] for f in frames], dtype=bool)
    qpos = np.stack([f["qpos"] for f in frames])
    target_q = np.stack([f["target_q"] for f in frames])
    gripper_cmd = np.array([f["gripper_cmd"] for f in frames], dtype=np.float64)

    timing = compute_timing_stats(timestamps)
    active_ratio = float(np.mean(active))

    if active.any():
        err = np.linalg.norm(qpos[active] - target_q[active], axis=1)
        tracking_err_mean = float(np.mean(err))
        tracking_err_max = float(np.max(err))
        joint_motion = float(np.sum(np.max(qpos[active], axis=0) - np.min(qpos[active], axis=0)))
    else:
        tracking_err_mean = float("nan")
        tracking_err_max = float("nan")
        joint_motion = 0.0

    if active.any():
        first_active = int(np.argmax(active))
        last_active = len(active) - 1 - int(np.argmax(active[::-1]))
        idle_start = first_active
        idle_end = len(active) - 1 - last_active
    else:
        idle_start = len(frames)
        idle_end = len(frames)

    gripper_transitions, gripper_pattern = analyze_gripper(gripper_cmd, timestamps, active)

    has_pip = any(f.get("image_pip") is not None for f in frames)
    main_frozen, main_shape = analyze_image_stream(frames, "image_main", fast=fast)
    pip_frozen, pip_shape = (
        analyze_image_stream(frames, "image_pip", fast=fast) if has_pip else (None, None)
    )

    duration_s = float(timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0
    fps_nominal = float(metadata.get("fps", 10.0))

    grade, issues, trainable_main, trainable_pip = grade_episode(
        success=bool(metadata.get("success", False)),
        num_frames=len(frames),
        active_ratio=active_ratio,
        gripper_pattern=gripper_pattern,
        main_frozen_ratio=main_frozen,
        pip_frozen_ratio=pip_frozen,
        has_pip=has_pip,
        joint_motion_rad=joint_motion,
        idle_end=idle_end,
        fps_measured=float(timing["fps_mean"]),
        timing_gaps=int(timing["gaps_gt_150ms"]),
    )

    return EpisodeReport(
        file=path.name,
        episode_index=int(metadata.get("episode_index", -1)),
        success=bool(metadata.get("success", False)),
        task=str(metadata.get("task", "")),
        num_frames=len(frames),
        duration_s=duration_s,
        fps_nominal=fps_nominal,
        fps_measured=float(timing["fps_mean"]),
        active_ratio=active_ratio,
        idle_start=idle_start,
        idle_end=idle_end,
        tracking_err_mean=tracking_err_mean,
        tracking_err_max=tracking_err_max,
        joint_motion_rad=joint_motion,
        gripper_transitions=gripper_transitions,
        gripper_pattern=gripper_pattern,
        main_frozen_ratio=main_frozen,
        pip_frozen_ratio=pip_frozen,
        main_shape=main_shape,
        pip_shape=pip_shape,
        timing_gaps_gt_150ms=int(timing["gaps_gt_150ms"]),
        grade=grade,
        issues=issues,
        trainable_main=trainable_main,
        trainable_pip=trainable_pip,
    )


def print_table(reports: list[EpisodeReport]) -> None:
    header = (
        f"{'Ep':>3} {'Grd':>3} {'Succ':>5} {'Frm':>4} {'Dur':>5} "
        f"{'FPS':>5} {'Act%':>5} {'MainFrz':>7} {'PipFrz':>7} "
        f"{'Grip':>10} {'Train':>10}"
    )
    print(header)
    print("-" * len(header))
    for r in reports:
        pip_frz = f"{r.pip_frozen_ratio:6.0%}" if r.pip_frozen_ratio is not None else "    n/a"
        train = "main" if r.trainable_main else ("pip" if r.trainable_pip else "no")
        print(
            f"{r.episode_index:>3} {r.grade:>3} {str(r.success):>5} {r.num_frames:>4} "
            f"{r.duration_s:>5.1f} {r.fps_measured:>5.1f} {r.active_ratio * 100:>4.0f}% "
            f"{r.main_frozen_ratio:>6.0%} {pip_frz} {r.gripper_pattern:>10} {train:>10}"
        )


def print_details(reports: list[EpisodeReport]) -> None:
    print("\n--- Per-episode details ---\n")
    for r in reports:
        print(f"Episode {r.episode_index:05d} ({r.file})  grade={r.grade}")
        print(f"  Task: {r.task}")
        print(
            f"  Frames={r.num_frames}  duration={r.duration_s:.1f}s  "
            f"FPS={r.fps_measured:.1f} (nominal {r.fps_nominal:.0f})  "
            f"gaps>150ms={r.timing_gaps_gt_150ms}"
        )
        print(
            f"  Active={r.active_ratio:.0%}  idle start/end={r.idle_start}/{r.idle_end}  "
            f"joint motion={r.joint_motion_rad:.2f} rad"
        )
        print(
            f"  Tracking err: mean={r.tracking_err_mean:.4f} rad  max={r.tracking_err_max:.4f} rad"
        )
        print(
            f"  Gripper: {r.gripper_pattern} ({r.gripper_transitions} transitions)  "
            f"main frozen={r.main_frozen_ratio:.0%} ({r.main_shape})"
        )
        if r.pip_frozen_ratio is not None:
            print(f"  PiP frozen={r.pip_frozen_ratio:.0%} ({r.pip_shape})")
        if r.issues:
            print(f"  Issues: {', '.join(r.issues)}")
        print()


def print_aggregate(reports: list[EpisodeReport], raw_dir: Path) -> None:
    if not reports:
        print("No episodes to summarize.")
        return

    n = len(reports)
    train_main = sum(1 for r in reports if r.trainable_main)
    train_pip = sum(1 for r in reports if r.trainable_pip)
    success = sum(1 for r in reports if r.success)
    active_frames = sum(int(r.num_frames * r.active_ratio) for r in reports)
    total_frames = sum(r.num_frames for r in reports)
    total_bytes = sum((raw_dir / r.file).stat().st_size for r in reports if (raw_dir / r.file).exists())

    print("=" * 72)
    print("AGGREGATE SUMMARY")
    print("=" * 72)
    print(f"Episodes       : {n}")
    print(f"Success        : {success}/{n}")
    print(f"Grade A (main) : {train_main}/{n}")
    print(f"Grade C (pip)  : {train_pip - train_main}/{n}  (pip-only usable)")
    print(
        f"Duration       : {np.mean([r.duration_s for r in reports]):.1f}s ± "
        f"{np.std([r.duration_s for r in reports]):.1f}s"
    )
    print(
        f"Frames         : {np.mean([r.num_frames for r in reports]):.0f} ± "
        f"{np.std([r.num_frames for r in reports]):.0f}  "
        f"(active ~{active_frames}/{total_frames})"
    )
    print(f"Measured FPS   : {np.mean([r.fps_measured for r in reports]):.2f}")
    print(f"Total size     : {total_bytes / 1e6:.1f} MB")
    print()
    print("Grade legend:")
    print("  A = dual-camera ready (main stream fresh, pick+place, active>=50%)")
    print("  C = pip-only usable (main frozen but wrist camera ok)")
    print("  B = partial issues  D = weak  F = unusable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate raw UR5e BC episodes.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("logs/ur5e_bc/raw"),
        help="Directory containing episode_XXXXX.pkl files",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write JSON report",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Sample images instead of decoding every frame (faster, less precise)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary table and aggregate stats",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    if not raw_dir.is_dir():
        print(f"Directory not found: {raw_dir}", file=sys.stderr)
        raise SystemExit(1)

    episode_paths = sorted(raw_dir.glob("episode_*.pkl"))
    if not episode_paths:
        print(f"No episode_*.pkl files in {raw_dir}", file=sys.stderr)
        raise SystemExit(1)

    reports: list[EpisodeReport] = []
    errors: list[str] = []
    for path in episode_paths:
        try:
            reports.append(analyze_episode(path, fast=args.fast))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    print(f"\nUR5e BC Raw Evaluation — {len(reports)} episodes in {raw_dir}\n")
    print_table(reports)
    if not args.quiet:
        print_details(reports)
    print_aggregate(reports, raw_dir)

    if errors:
        print("\nErrors:")
        for err in errors:
            print(f"  {err}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "raw_dir": str(raw_dir),
            "episodes": [asdict(r) for r in reports],
            "errors": errors,
            "summary": {
                "count": len(reports),
                "trainable_main": sum(1 for r in reports if r.trainable_main),
                "trainable_pip": sum(1 for r in reports if r.trainable_pip),
                "success": sum(1 for r in reports if r.success),
            },
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nJSON report written to {args.json_out.resolve()}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
