#!/usr/bin/env python3
"""
Convert raw UR5e BC episode pickles to LeRobot dataset format.

Uses the installed LeRobotDataset API (v2/v3 compatible import paths).
Images are stored as video via LeRobot; tabular fields go to parquet.

Field mapping:
  image_main           → observation.images.main
  [qpos, gripper_state] → observation.state   (7-dim, 0=open 1=close)
  [target_q, gripper_cmd] → action            (7-dim, 0=open 1=close)

Usage:
  python scripts/convert_ur5e_raw_to_lerobot.py \\
    --raw-dir logs/ur5e_bc/raw \\
    --repo-id local/ur5e_block_pick \\
    --output-dir datasets/ur5e_block_pick

  # Include failed episodes and idle frames:
  python scripts/convert_ur5e_raw_to_lerobot.py --include-failed --include-inactive
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

from xrobotoolkit_teleop.bc.config import BCConfig, DEFAULT_TASK
from xrobotoolkit_teleop.utils.image_utils import decompress_jpg_to_image


def import_lerobot_dataset():
    """Try multiple LeRobot import paths (v2 / v3 layout differences)."""
    candidates = [
        ("lerobot.datasets.lerobot_dataset", "LeRobotDataset"),
        ("lerobot.common.datasets.lerobot_dataset", "LeRobotDataset"),
    ]
    errors = []
    lerobot_version = "unknown"
    try:
        import lerobot

        lerobot_version = getattr(lerobot, "__version__", "unknown")
    except ImportError:
        pass

    for module_path, class_name in candidates:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            return cls, lerobot_version, module_path
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_path}: {exc}")

    msg = (
        "LeRobot is not installed or LeRobotDataset could not be imported.\n"
        "Install with: pip install lerobot\n"
        "Import errors:\n  " + "\n  ".join(errors)
    )
    raise ImportError(msg)


def load_raw_episode(path: Path) -> tuple[dict, list[dict]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["metadata"], data["frames"]


def decode_image_main(frame: dict) -> np.ndarray:
    """Return RGB uint8 image for LeRobot."""
    img = frame["image_main"]
    if isinstance(img, bytes):
        bgr = decompress_jpg_to_image(img)
    else:
        bgr = np.asarray(img)
    if bgr is None:
        raise ValueError("Failed to decode image_main")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def build_features(image_shape: tuple[int, int, int]) -> dict:
    h, w, c = image_shape
    return {
        "observation.images.main": {
            "dtype": "video",
            "shape": (h, w, c),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["j0", "j1", "j2", "j3", "j4", "j5", "gripper"],
        },
    }


def filter_frames(
    frames: list[dict],
    include_inactive: bool,
) -> list[dict]:
    if include_inactive:
        return frames
    return [f for f in frames if f.get("control_active", False)]


def write_success_sidecar(output_dir: Path, success_map: dict[int, bool]) -> None:
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    sidecar = meta_dir / "bc_episode_success.json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(success_map, f, indent=2)
    print(f"Wrote episode success map → {sidecar}")


def convert_episodes(
    raw_dir: Path,
    output_dir: Path,
    repo_id: str,
    fps: int,
    task: str,
    include_failed: bool,
    include_inactive: bool,
    robot_type: str,
) -> None:
    LeRobotDataset, lerobot_version, import_path = import_lerobot_dataset()
    print(f"LeRobot version={lerobot_version}, import={import_path}")

    episode_files = sorted(raw_dir.glob("episode_*.pkl"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.pkl in {raw_dir}")

    # Peek first episode for image shape
    _, first_frames = load_raw_episode(episode_files[0])
    if not first_frames:
        raise ValueError(f"First episode empty: {episode_files[0]}")
    rgb0 = decode_image_main(first_frames[0])
    features = build_features(rgb0.shape)

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"Warning: output_dir {output_dir} is not empty.")

    create_kwargs = dict(
        repo_id=repo_id,
        fps=fps,
        features=features,
        robot_type=robot_type,
        root=output_dir,
        use_videos=True,
    )

    dataset = LeRobotDataset.create(**create_kwargs)
    success_map: dict[int, bool] = {}
    converted = 0

    for ep_path in episode_files:
        metadata, frames = load_raw_episode(ep_path)
        ep_index = int(metadata.get("episode_index", converted))
        success = bool(metadata.get("success", False))
        success_map[ep_index] = success

        if not include_failed and not success:
            print(f"Skip failed episode {ep_index} ({ep_path.name})")
            continue

        kept = filter_frames(frames, include_inactive)
        if not kept:
            print(f"Skip episode {ep_index}: no frames after filtering")
            continue

        for frame in kept:
            rgb = decode_image_main(frame)
            qpos = np.asarray(frame["qpos"], dtype=np.float32).reshape(6)
            target_q = np.asarray(frame["target_q"], dtype=np.float32).reshape(6)
            gripper_state = np.float32(frame["gripper_state"])
            gripper_cmd = np.float32(frame["gripper_cmd"])

            observation_state = np.concatenate([qpos, [gripper_state]]).astype(np.float32)
            action = np.concatenate([target_q, [gripper_cmd]]).astype(np.float32)

            lerobot_frame = {
                "observation.images.main": rgb,
                "observation.state": observation_state,
                "action": action,
                "task": metadata.get("task", task),
            }

            dataset.add_frame(lerobot_frame)

        dataset.save_episode()
        converted += 1
        print(f"Converted episode {ep_index} ({len(kept)} frames, success={success})")

    finalize = getattr(dataset, "finalize", None)
    if callable(finalize):
        finalize()
        print("Called dataset.finalize()")

    consolidate = getattr(dataset, "consolidate", None)
    if callable(consolidate):
        consolidate()
        print("Called dataset.consolidate() (legacy API)")

    write_success_sidecar(output_dir, success_map)
    print(f"Done: {converted} episodes → {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw UR5e BC pickles to LeRobot format.")
    parser.add_argument("--raw-dir", type=Path, default=Path("logs/ur5e_bc/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/ur5e_block_pick"))
    parser.add_argument("--repo-id", type=str, default=BCConfig().lerobot_repo_id)
    parser.add_argument(
        "--fps",
        type=int,
        default=7,
        help="LeRobot 数据集 fps（建议与实测采样率一致，当前 raw 数据约 7Hz）",
    )
    parser.add_argument("--task", type=str, default=DEFAULT_TASK)
    parser.add_argument("--robot-type", type=str, default="ur5e")
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include episodes marked success=False (default: only success=True)",
    )
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include frames with control_active=False (default: filter them out)",
    )
    args = parser.parse_args()

    if not args.raw_dir.is_dir():
        print(f"raw-dir not found: {args.raw_dir}", file=sys.stderr)
        raise SystemExit(1)

    try:
        convert_episodes(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            repo_id=args.repo_id,
            fps=args.fps,
            task=args.task,
            include_failed=args.include_failed,
            include_inactive=args.include_inactive,
            robot_type=args.robot_type,
        )
    except ImportError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
