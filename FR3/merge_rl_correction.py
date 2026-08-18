#!/usr/bin/env python3
"""
拼接 fail + raw episode，末尾追加 10 帧静止帧，保存到 Rl_correction。

用法：
  python FR3/merge_rl_correction.py

逻辑：
  1. 读取 logs/fr3_bc/raw_RL/fail/ 和 logs/fr3_bc/raw/ 中的 episode_*.pkl
  2. 只处理两个目录都存在的同序号 episode
  3. 拼接顺序：fail 帧在前，raw 帧在后
  4. 重排 frame_index，重算 timestamp
  5. 末尾追加 10 帧静止帧（复制最后一帧）
  6. 保存到 logs/fr3_bc/Rl_correction/

缺失配对的 episode 会报错并跳过。
"""

import argparse
import copy
import os
import pickle
import sys
from pathlib import Path


DEFAULT_FAIL_DIR = "logs/fr3_bc/raw_RL/fail"
DEFAULT_RAW_DIR = "logs/fr3_bc/raw"
DEFAULT_DST_DIR = "logs/fr3_bc/Rl_correction"
PAD_FRAMES = 10


def extract_index(filename: str) -> int | None:
    """从 episode_00000.pkl 中提取序号。"""
    if filename.startswith("episode_") and filename.endswith(".pkl"):
        try:
            return int(filename[len("episode_"):-len(".pkl")])
        except ValueError:
            pass
    return None


def load_episode(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def concatenate_episodes(fail_data: dict, raw_data: dict) -> dict:
    """将 fail 和 raw 两个 episode 拼接，fail 在前，raw 在后。"""
    fail_frames = fail_data["frames"]
    raw_frames = raw_data["frames"]

    # 元数据以 fail 为主（主要是 episode_index、task、fps 等）
    meta = dict(fail_data["metadata"])
    fps = float(meta.get("fps", 10.0))

    # 深拷贝帧，避免修改原始数据
    all_frames = []
    for f in fail_frames:
        all_frames.append(copy.deepcopy(f))
    for f in raw_frames:
        all_frames.append(copy.deepcopy(f))

    # 重排 frame_index 和 timestamp
    for i, frame in enumerate(all_frames):
        frame["frame_index"] = i
        frame["timestamp"] = i / fps
        # episode_index 保持统一
        frame["episode_index"] = meta.get("episode_index", 0)

    data = {"metadata": meta, "frames": all_frames}
    return data


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


def update_metadata(data: dict) -> dict:
    """更新拼接后的 metadata 关键字段。"""
    meta = data["metadata"]
    frames = data["frames"]
    fps = float(meta.get("fps", 10.0))

    meta["num_frames"] = len(frames)
    meta["duration"] = len(frames) / fps if fps > 0 else 0.0
    if frames:
        meta["start_time"] = float(frames[0].get("timestamp", 0.0))
        meta["end_time"] = float(frames[-1].get("timestamp", 0.0))
    # success 标记：来自 fail 的拼接数据统一设为 False
    meta["success"] = False

    return data


def process_pair(fail_path: str, raw_path: str, dst_path: str, pad_n: int = PAD_FRAMES):
    """处理一对 episode：加载、拼接、padding、保存。"""
    fail_data = load_episode(fail_path)
    raw_data = load_episode(raw_path)

    data = concatenate_episodes(fail_data, raw_data)
    data = pad_episode(data, pad_n)
    data = update_metadata(data)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp_path, dst_path)

    return len(fail_data["frames"]), len(raw_data["frames"]), len(data["frames"])


def main():
    parser = argparse.ArgumentParser(
        description="拼接 fail + raw episode，追加静止帧到 Rl_correction"
    )
    parser.add_argument("--fail-dir", default=DEFAULT_FAIL_DIR,
                        help=f"fail episode 目录（默认 {DEFAULT_FAIL_DIR}）")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR,
                        help=f"raw episode 目录（默认 {DEFAULT_RAW_DIR}）")
    parser.add_argument("--dst-dir", default=DEFAULT_DST_DIR,
                        help=f"输出目录（默认 {DEFAULT_DST_DIR}）")
    parser.add_argument("--pad-frames", type=int, default=PAD_FRAMES,
                        help=f"末尾追加静止帧数（默认 {PAD_FRAMES}）")
    args = parser.parse_args()

    fail_dir = Path(args.fail_dir)
    raw_dir = Path(args.raw_dir)
    dst_dir = Path(args.dst_dir)

    if not fail_dir.is_dir():
        print(f"[!] fail 目录不存在: {fail_dir}")
        sys.exit(1)
    if not raw_dir.is_dir():
        print(f"[!] raw 目录不存在: {raw_dir}")
        sys.exit(1)

    dst_dir.mkdir(parents=True, exist_ok=True)

    # 收集两个目录的 episode 序号
    fail_files = {extract_index(f): f for f in os.listdir(fail_dir)
                  if extract_index(f) is not None}
    raw_files = {extract_index(f): f for f in os.listdir(raw_dir)
                 if extract_index(f) is not None}

    all_indices = sorted(set(fail_files.keys()) | set(raw_files.keys()))
    common_indices = sorted(set(fail_files.keys()) & set(raw_files.keys()))

    # 报告缺失配对
    missing = [idx for idx in all_indices if idx not in common_indices]
    if missing:
        print(f"[!] 以下 episode 缺失配对，已跳过：")
        for idx in missing:
            in_fail = "✓" if idx in fail_files else "✗"
            in_raw = "✓" if idx in raw_files else "✗"
            print(f"    episode_{idx:05d}.pkl  fail={in_fail} raw={in_raw}")
        print()

    if not common_indices:
        print("[!] 无共同 episode 可处理")
        sys.exit(1)

    print(f"[*] fail 目录: {fail_dir}")
    print(f"[*] raw 目录: {raw_dir}")
    print(f"[*] 输出目录: {dst_dir}")
    print(f"[*] 共同 episode 数量: {len(common_indices)}")
    print(f"[*] 每个 episode 末尾追加 {args.pad_frames} 帧\n")

    for idx in common_indices:
        fname = f"episode_{idx:05d}.pkl"
        fail_path = str(fail_dir / fail_files[idx])
        raw_path = str(raw_dir / raw_files[idx])
        dst_path = str(dst_dir / fname)

        try:
            fail_n, raw_n, total_n = process_pair(
                fail_path, raw_path, dst_path, pad_n=args.pad_frames
            )
            print(f"  {fname}: fail={fail_n} + raw={raw_n} + pad={args.pad_frames} = {total_n} 帧")
        except Exception as exc:
            print(f"[!] 处理 {fname} 失败: {exc}")

    print(f"\n[✓] 完成，共处理 {len(common_indices)} 个 episode")


if __name__ == "__main__":
    main()
