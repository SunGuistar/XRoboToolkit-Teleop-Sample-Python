#!/usr/bin/env python3
"""将 logs/ur5e_bc/raw 下 episode 文件按 0..N-1 连续重命名，并更新 pkl 内 episode_index。"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path


def renumber_raw_dir(raw_dir: Path, dry_run: bool = False) -> int:
    pattern = re.compile(r"^episode_(\d+)\.pkl$")
    files: list[tuple[int, Path]] = []
    for p in raw_dir.glob("episode_*.pkl"):
        m = pattern.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])

    if not files:
        print(f"未找到 episode 文件：{raw_dir}")
        return 0

    print(f"共 {len(files)} 个文件，将重编号为 episode_00000 .. episode_{len(files)-1:05d}")

    payloads: list[tuple[int, dict]] = []
    for new_idx, (old_idx, path) in enumerate(files):
        with open(path, "rb") as f:
            data = pickle.load(f)
        data["metadata"]["episode_index"] = new_idx
        for frame in data.get("frames", []):
            frame["episode_index"] = new_idx
        payloads.append((new_idx, data))
        print(f"  episode_{old_idx:05d}.pkl → episode_{new_idx:05d}.pkl")
        if dry_run:
            continue

    if dry_run:
        return len(files)

    # 先写临时文件，再删旧文件，最后 rename，避免编号冲突
    tmp_paths: list[Path] = []
    for new_idx, data in payloads:
        tmp = raw_dir / f"episode_{new_idx:05d}.pkl.renumber_tmp"
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_paths.append(tmp)

    for _, old_path in files:
        old_path.unlink()

    for new_idx, tmp in enumerate(tmp_paths):
        final = raw_dir / f"episode_{new_idx:05d}.pkl"
        tmp.replace(final)

    print(f"完成 → {raw_dir.resolve()}")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="连续重命名 UR5e BC raw episode 文件。")
    parser.add_argument("--raw-dir", type=Path, default=Path("logs/ur5e_bc/raw"))
    parser.add_argument("--dry-run", action="store_true", help="只打印计划，不写盘")
    args = parser.parse_args()

    if not args.raw_dir.is_dir():
        print(f"目录不存在：{args.raw_dir}", file=sys.stderr)
        raise SystemExit(1)

    renumber_raw_dir(args.raw_dir.resolve(), dry_run=args.dry_run)


if __name__ == "__main__":
    main()
