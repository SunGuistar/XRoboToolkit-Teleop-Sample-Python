#!/usr/bin/env python3
"""在每个 episode 末尾追加 N 帧静止画面（复制最后一帧）。

用法:
  python scripts/pad_episode_frames.py \
      --src logs/fr3_bc/第三版/raw_correct \
      --dst logs/fr3_bc/第三版/correct \
      --pad-frames 10
"""

import argparse
import copy
import os
import pickle
import sys


def pad_episode(data: dict, pad_n: int) -> dict:
    """在 episode 末尾追加 pad_n 帧（复制最后一帧，递增 frame_index/timestamp）。"""
    frames = data["frames"]
    meta = data["metadata"]
    fps = float(meta.get("fps", 10.0))
    orig_len = len(frames)

    if orig_len == 0 or pad_n <= 0:
        return data

    last = frames[-1]
    last_ts = float(last.get("timestamp", 0.0))
    last_idx = int(last.get("frame_index", orig_len - 1))

    for i in range(pad_n):
        new_frame = copy.deepcopy(last)
        new_frame["frame_index"] = last_idx + 1 + i
        new_frame["timestamp"] = last_ts + (i + 1) / fps
        frames.append(new_frame)

    # 更新 metadata
    meta["num_frames"] = len(frames)
    pad_duration = pad_n / fps
    meta["duration"] = float(meta.get("duration", 0.0)) + pad_duration
    if "end_time" in meta:
        meta["end_time"] = float(meta["end_time"]) + pad_duration

    return data


def main():
    parser = argparse.ArgumentParser(description="在 episode 末尾追加静止帧")
    parser.add_argument("--src", required=True, help="原始 episode 目录")
    parser.add_argument("--dst", required=True, help="输出目录")
    parser.add_argument("--pad-frames", type=int, default=10,
                        help="追加帧数（默认 10）")
    args = parser.parse_args()

    src_dir = args.src
    dst_dir = args.dst
    pad_n = args.pad_frames

    if not os.path.isdir(src_dir):
        print(f"[!] 源目录不存在: {src_dir}")
        sys.exit(1)

    os.makedirs(dst_dir, exist_ok=True)

    pkl_files = sorted(
        f for f in os.listdir(src_dir)
        if f.startswith("episode_") and f.endswith(".pkl")
    )
    if not pkl_files:
        print(f"[!] 源目录无 episode_*.pkl 文件: {src_dir}")
        sys.exit(1)

    print(f"源目录: {src_dir}")
    print(f"输出目录: {dst_dir}")
    print(f"共 {len(pkl_files)} 个 episode，每个追加 {pad_n} 帧\n")

    for fname in pkl_files:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(dst_dir, fname)

        with open(src_path, "rb") as f:
            data = pickle.load(f)

        orig_n = len(data["frames"])
        data = pad_episode(data, pad_n)
        new_n = len(data["frames"])

        with open(dst_path, "wb") as f:
            pickle.dump(data, f)

        print(f"  {fname}: {orig_n} -> {new_n} 帧")

    print(f"\n[✓] 完成，共处理 {len(pkl_files)} 个 episode")


if __name__ == "__main__":
    main()
