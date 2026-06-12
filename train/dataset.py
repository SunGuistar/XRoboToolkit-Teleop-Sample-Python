"""Load LeRobot v3 parquet + video for UR5e BC (minimal, no LeRobot policy dependency)."""

from __future__ import annotations

import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMAGE_KEY_CANDIDATES = (
    "observation.images.main",
    "observation.image",
    "observation.images",
)


def resolve_image_key(features: dict) -> str:
    for key in IMAGE_KEY_CANDIDATES:
        if key in features:
            return key
    for key, spec in features.items():
        if key.startswith("observation.images.") or key == "observation.image":
            return key
        if spec.get("dtype") == "video":
            return key
    raise KeyError(f"No image/video feature found in: {list(features.keys())}")


def _video_path(root: Path, info: dict, image_key: str, chunk_index: int = 0, file_index: int = 0) -> Path:
    template = info["video_path"]
    rel = template.format(video_key=image_key, chunk_index=chunk_index, file_index=file_index)
    return root / rel


def _decode_video_rgb(path: Path, target_size: int) -> np.ndarray:
    """Decode full MP4 to (T, H, W, 3) uint8 RGB, resized to square target_size."""
    import cv2

    container = av.open(str(path))
    stream = container.streams.video[0]
    frames: list[np.ndarray] = []
    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")
        if rgb.shape[0] != target_size or rgb.shape[1] != target_size:
            rgb = cv2.resize(rgb, (target_size, target_size), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    container.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0)


class LeRobotBCDataset(Dataset):
    """
    Reads tabular fields from parquet and images from LeRobot video MP4.

    observation.state: (7,)  [qpos(6), gripper(1)]
    action: (7,)            [target_q(6), gripper_cmd(1)]
    """

    def __init__(
        self,
        dataset_root: str | Path,
        episode_indices: list[int] | None = None,
        image_size: int = 224,
        preload_images: bool = True,
    ):
        self.root = Path(dataset_root).resolve()
        self.image_size = image_size

        with open(self.root / "meta/info.json", encoding="utf-8") as f:
            self.info = json.load(f)

        self.image_key = resolve_image_key(self.info["features"])
        self.state_dim = int(self.info["features"]["observation.state"]["shape"][0])
        self.action_dim = int(self.info["features"]["action"]["shape"][0])
        self.fps = int(self.info.get("fps", 10))

        data_glob = sorted((self.root / "data").glob("chunk-*/file-*.parquet"))
        if not data_glob:
            raise FileNotFoundError(f"No parquet under {self.root / 'data'}")
        df = pd.concat([pd.read_parquet(p) for p in data_glob], ignore_index=True)

        if episode_indices is not None:
            df = df[df["episode_index"].isin(episode_indices)].copy()
        df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
        self.df = df

        self.states = np.stack(df["observation.state"].values).astype(np.float32)
        self.actions = np.stack(df["action"].values).astype(np.float32)
        self.episode_indices = df["episode_index"].values.astype(np.int64)
        self.frame_indices = df["frame_index"].values.astype(np.int64)
        self.global_indices = df["index"].values.astype(np.int64)

        video_path = _video_path(self.root, self.info, self.image_key)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        print(f"[dataset] root={self.root}")
        print(f"[dataset] image_key={self.image_key}  video={video_path.name}")
        print(f"[dataset] samples={len(self.df)}  state_dim={self.state_dim}  action_dim={self.action_dim}")
        print(f"[dataset] states shape={self.states.shape}  actions shape={self.actions.shape}")

        if preload_images:
            print(f"[dataset] preloading video frames (resize {image_size}x{image_size})...")
            all_frames = _decode_video_rgb(video_path, image_size)
            print(f"[dataset] decoded video shape={all_frames.shape}")
            row_idx = self.global_indices
            if row_idx.max() >= len(all_frames):
                raise IndexError(
                    f"parquet index max={row_idx.max()} >= video frames={len(all_frames)}"
                )
            self.images = all_frames[row_idx]  # (N, H, W, 3) uint8
            print(f"[dataset] indexed images shape={self.images.shape}")
        else:
            self._video_path = video_path
            self.images = None

        # normalization stats for state/action (train set should recompute; here per-subset)
        self.state_mean = self.states.mean(axis=0)
        self.state_std = self.states.std(axis=0).clip(min=1e-6)
        self.action_mean = self.actions.mean(axis=0)
        self.action_std = self.actions.std(axis=0).clip(min=1e-6)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        state = self.states[idx]
        action = self.actions[idx]
        if self.images is not None:
            image = self.images[idx]
        else:
            import cv2

            gidx = int(self.global_indices[idx])
            container = av.open(str(self._video_path))
            stream = container.streams.video[0]
            image = None
            for i, frame in enumerate(container.decode(stream)):
                if i == gidx:
                    image = frame.to_ndarray(format="rgb24")
                    break
            container.close()
            if image is None:
                raise IndexError(f"Frame {gidx} not found in video")
            image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        # (3, H, W) float32 [0, 1]
        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return {
            "image": image_t,
            "state": torch.from_numpy(state),
            "action": torch.from_numpy(action),
            "episode_index": int(self.episode_indices[idx]),
            "frame_index": int(self.frame_indices[idx]),
        }

    def episodes(self) -> list[int]:
        return sorted(self.df["episode_index"].unique().tolist())
