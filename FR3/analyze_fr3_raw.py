"""分析 logs/fr3_bc/raw 下的 BC 原始 episode 数据（episode_XXXXX.pkl）。

针对 teleop_single_fr3_hand.py 录制的 V1 schema：
  metadata: episode_index / success / task / fps / num_frames / duration / image_format ...
  frames:   timestamp / frame_index / image_main / image_pip? /
            qpos(6) / target_q(6) / gripper_state / gripper_cmd / control_active

功能：
  1. 总览表：每局帧数、实际 fps、时长、图像信息
  2. 质量检查（默认开启）：
     - frame_index 连续性 / timestamp 单调性 / 丢帧统计
     - qpos/target_q 的 NaN/inf
     - FR3 关节限位越界
     - qpos 与 target_q 跟随误差（|q - target_q| 最大值）
     - control_active（按住 Grip）占比、夹爪开合统计
  3. --plot N：对指定局画 qpos/target_q/gripper 曲线（保存 PNG，不弹窗）

用法：
  python FR3/analyze_fr3_raw.py                       # 全部 episode 总览 + 质检
  python FR3/analyze_fr3_raw.py --episode 0 3 5       # 只看指定局
  python FR3/analyze_fr3_raw.py --plot 0              # 画第 0 局曲线 → logs/fr3_bc/raw/plots/
  python FR3/analyze_fr3_raw.py --raw-dir <dir>       # 指定其他 raw 目录
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

DEFAULT_RAW_DIR = Path("logs/fr3_bc/raw")
PLOT_DIR_NAME = "plots"

# FR3 关节限位（弧度），与录制/部署一致。导入失败时跳过限位检查。
try:
    from xrobotoolkit_teleop.hardware.fr3_config import fr3_joint_limits_arrays

    _LOWER, _UPPER = fr3_joint_limits_arrays()
except Exception:  # noqa: BLE001
    _LOWER, _UPPER = None, None


def _find_episodes(raw_dir: Path, indices: list[int] | None) -> list[Path]:
    files = sorted(raw_dir.glob("episode_*.pkl"))
    if indices is not None:
        wanted = {f"episode_{i:05d}.pkl" for i in indices}
        missing = wanted - {f.name for f in files}
        for name in sorted(missing):
            print(f"警告：未找到 {raw_dir / name}", file=sys.stderr)
        files = [f for f in files if f.name in wanted]
    return files


def _load_episode(path: Path) -> dict:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if "metadata" not in payload or "frames" not in payload:
        raise ValueError(f"文件格式不是 BC raw episode：{path}")
    return payload


def _image_desc(frame: dict) -> str:
    fmt = None
    img = frame.get("image_main")
    if isinstance(img, (bytes, bytearray)):
        fmt = "jpeg"
    elif isinstance(img, np.ndarray):
        fmt = f"raw {img.shape[1]}x{img.shape[0]}"
    else:
        return "未知"
    desc = f"主:{fmt}"
    if "image_pip" in frame:
        pip = frame["image_pip"]
        if isinstance(pip, (bytes, bytearray)):
            desc += " 辅:jpeg"
        elif isinstance(pip, np.ndarray):
            desc += f" 辅:raw {pip.shape[1]}x{pip.shape[0]}"
    return desc


def _stack(frames: list[dict], key: str) -> np.ndarray | None:
    vals = [f.get(key) for f in frames]
    if any(v is None for v in vals):
        return None
    return np.asarray(vals, dtype=np.float64)


def analyze_episode(path: Path, *, quiet: bool = False) -> dict:
    """返回单局质检结果 dict；quiet 模式下仅打印有问题的局。"""
    payload = _load_episode(path)
    meta, frames = payload["metadata"], payload["frames"]
    result: dict = {"path": path, "episode_index": meta.get("episode_index"), "issues": []}
    issues = result["issues"]
    lines: list[str] = []

    def say(msg: str = "") -> None:
        lines.append(msg)

    n = len(frames)
    result["num_frames"] = n
    say(f"\n===== {path.name}（第 {meta.get('episode_index')} 局）=====")
    say(
        f"  task={meta.get('task')!r}  success={meta.get('success')}  "
        f"声明 fps={meta.get('fps')}  声明帧数={meta.get('num_frames')}  "
        f"声明时长={meta.get('duration', 0):.1f}s"
    )

    if n != meta.get("num_frames", n):
        issues.append(f"帧数不一致：metadata={meta.get('num_frames')} 实际={n}")
    if n == 0:
        issues.append("空 episode（0 帧）")
        if not quiet or issues:
            print("\n".join(lines))
        return result

    # ---- 时间轴 ----
    ts = np.array([f["timestamp"] for f in frames], dtype=np.float64)
    result["duration"] = float(ts[-1] - ts[0])
    dts = np.diff(ts)
    if np.any(dts <= 0):
        issues.append(f"timestamp 非单调（{int(np.sum(dts <= 0))} 处）")
    pos_dts = dts[dts > 0]
    if len(pos_dts) > 2:
        med_dt = float(np.median(pos_dts))
        actual_fps = 1.0 / med_dt
        result["actual_fps"] = actual_fps
        # 丢帧判定：dt 超过 2 倍中位周期视为漏采
        gaps = int(np.sum(pos_dts > 2.0 * med_dt))
        result["gaps"] = gaps
        if gaps:
            issues.append(f"疑似丢帧 {gaps} 处（dt > {2.0 * med_dt * 1000:.0f}ms）")
        say(
            f"  时间轴：{result['duration']:.1f}s  实际 fps={actual_fps:.2f}"
            f"（中位 dt={med_dt * 1000:.0f}ms）  dt 范围 "
            f"{pos_dts.min() * 1000:.0f}~{pos_dts.max() * 1000:.0f}ms  丢帧疑似={gaps}"
        )

    # ---- 帧索引连续性 ----
    idx = np.array([f["frame_index"] for f in frames], dtype=np.int64)
    uniq = np.unique(idx)
    if uniq.size != n:
        issues.append(f"frame_index 存在重复（{n - uniq.size} 个）")
    if uniq.size > 1:
        missing = int((uniq[-1] - uniq[0] + 1) - uniq.size)
        if missing:
            issues.append(f"frame_index 不连续（缺 {missing} 个编号）")

    # ---- 关节数据 ----
    qpos = _stack(frames, "qpos")
    target_q = _stack(frames, "target_q")
    for name, arr in (("qpos", qpos), ("target_q", target_q)):
        if arr is None:
            issues.append(f"{name} 存在缺失帧")
            continue
        if arr.shape != (n, 6):
            issues.append(f"{name} 形状异常：{arr.shape}（期望 ({n}, 6)）")
            continue
        bad = int(np.sum(~np.isfinite(arr)))
        if bad:
            issues.append(f"{name} 含 {bad} 个 NaN/inf")

    if qpos is not None and qpos.shape == (n, 6) and np.all(np.isfinite(qpos)):
        say(
            "  qpos 范围（度）：\n"
            f"    min: {np.array2string(np.rad2deg(qpos.min(axis=0)), precision=1, separator=', ')}\n"
            f"    max: {np.array2string(np.rad2deg(qpos.max(axis=0)), precision=1, separator=', ')}"
        )
        if _LOWER is not None and _UPPER is not None:
            tol = np.deg2rad(1.0)  # 1° 容差（软限位裁剪误差）
            over = np.sum((qpos < _LOWER - tol) | (qpos > _UPPER + tol))
            if over:
                issues.append(f"qpos 超出 FR3 关节限位 {over} 个值")

    if qpos is not None and target_q is not None and qpos.shape == target_q.shape:
        err = np.abs(qpos - target_q)
        finite = np.isfinite(err)
        if finite.any():
            max_err = float(np.max(err[finite]))
            result["max_track_err_deg"] = float(np.rad2deg(max_err))
            say(
                f"  跟随误差 |qpos-target_q|：均值 {np.rad2deg(err[finite].mean()):.2f}°  "
                f"最大 {np.rad2deg(max_err):.2f}°"
            )
            if max_err > np.deg2rad(30.0):
                issues.append(f"跟随误差过大（最大 {np.rad2deg(max_err):.1f}°），可能伺服丢包或卡滞")
        else:
            issues.append("跟随误差无法计算（qpos/target_q 全部非有限值）")

    # ---- 夹爪 / 控制状态 ----
    gs = _stack(frames, "gripper_state")
    gc = _stack(frames, "gripper_cmd")
    if gs is not None:
        closed = float(np.mean(gs > 0.5))
        say(f"  夹爪（扳机代理）：state 均值 {gs.mean():.2f}  闭合占比 {closed * 100:.0f}%")
    if gc is not None and gs is not None and np.array_equal(gs, gc):
        say("  注：gripper_state 与 gripper_cmd 完全相同（无物理夹爪，扳机代理）。")
    ca = [bool(f.get("control_active", False)) for f in frames]
    active_ratio = float(np.mean(ca))
    result["control_active_ratio"] = active_ratio
    say(f"  control_active（按住 Grip）占比：{active_ratio * 100:.0f}%")
    if active_ratio == 0.0:
        issues.append("全程未检测到 Grip 控制（可能只是静止录制）")

    # ---- 图像 ----
    say(f"  图像：{_image_desc(frames[0])}  image_format={meta.get('image_format')}")
    if meta.get("image_format") == "jpeg":
        ok = sum(
            1
            for f in frames
            if isinstance(f.get("image_main"), (bytes, bytearray))
            and bytes(f["image_main"][:3]) == b"\xff\xd8\xff"
        )
        if ok != n:
            issues.append(f"JPEG 魔数异常：{n - ok}/{n} 帧不以 FFD8FF 开头")

    # ---- 结论 ----
    if issues:
        say(f"  ⚠ 发现 {len(issues)} 个问题：")
        for msg in issues:
            say(f"    - {msg}")
    else:
        say("  ✓ 质检通过，无异常。")
    if not quiet or issues:
        print("\n".join(lines))
    return result


def plot_episode(path: Path, out_dir: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图。", file=sys.stderr)
        return None

    payload = _load_episode(path)
    frames = payload["frames"]
    if not frames:
        return None
    ts = np.array([f["timestamp"] for f in frames]) - frames[0]["timestamp"]
    qpos = _stack(frames, "qpos")
    target_q = _stack(frames, "target_q")
    gs = _stack(frames, "gripper_state")

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    labels = [f"J{i + 1}" for i in range(6)]
    if qpos is not None:
        for j in range(6):
            axes[0].plot(ts, np.rad2deg(qpos[:, j]), label=labels[j])
    axes[0].set_ylabel("qpos (deg)")
    axes[0].legend(loc="upper right", ncol=6, fontsize=8)
    axes[0].set_title(f"{path.name}  qpos")
    if target_q is not None:
        for j in range(6):
            axes[1].plot(ts, np.rad2deg(target_q[:, j]), label=labels[j])
    axes[1].set_ylabel("target_q (deg)")
    axes[1].legend(loc="upper right", ncol=6, fontsize=8)
    if gs is not None:
        axes[2].plot(ts, gs, label="gripper_state")
        gc = _stack(frames, "gripper_cmd")
        if gc is not None:
            axes[2].plot(ts, gc, "--", label="gripper_cmd")
        axes[2].set_ylabel("gripper (0=open)")
        axes[2].legend(loc="upper right")
    axes[2].set_xlabel("t (s)")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{path.stem}.png"
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 FR3 BC 原始 episode 数据")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="raw episode 目录")
    parser.add_argument("--episode", type=int, nargs="*", default=None, help="只分析指定局号（如 --episode 0 3）")
    parser.add_argument("--plot", type=int, nargs="*", default=None, help="对指定局画曲线并保存 PNG")
    parser.add_argument("--quiet", action="store_true", help="只打印问题局，不打印详细报告")
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    if not raw_dir.is_dir():
        print(f"目录不存在：{raw_dir}", file=sys.stderr)
        sys.exit(1)

    files = _find_episodes(raw_dir, args.episode)
    if not files:
        print(f"未找到 episode_*.pkl：{raw_dir}")
        print("提示：先用 teleop_single_fr3_hand.py 录制（X 开始/停止），停止后自动保存到该目录。")
        return

    print(f"共 {len(files)} 个 episode：{raw_dir}")
    summary = []
    for path in files:
        try:
            result = analyze_episode(path, quiet=args.quiet)
        except Exception as exc:  # noqa: BLE001
            print(f"\n===== {path.name} =====\n  ✗ 读取/解析失败：{exc}", file=sys.stderr)
            continue
        summary.append(result)

    # 汇总表
    print("\n===== 汇总 =====")
    print(f"{'episode':>8} {'帧数':>6} {'实际fps':>8} {'时长s':>7} {'丢帧':>5} {'最大跟随误差':>10} {'Grip占比':>8} {'问题':>4}")
    for r in summary:
        print(
            f"{r['episode_index']:>8} {r.get('num_frames', 0):>6} "
            f"{r.get('actual_fps', float('nan')):>8.2f} {r.get('duration', 0):>7.1f} "
            f"{r.get('gaps', '-'):>5} "
            f"{r.get('max_track_err_deg', float('nan')):>9.1f}° "
            f"{r.get('control_active_ratio', 0) * 100:>7.0f}% {len(r['issues']):>4}"
        )
    bad = [r for r in summary if r["issues"]]
    print(f"\n{len(summary)} 局中 {len(bad)} 局存在问题。")

    if args.plot:
        plot_files = _find_episodes(raw_dir, args.plot)
        out_dir = raw_dir / PLOT_DIR_NAME
        for path in plot_files:
            out = plot_episode(path, out_dir)
            if out is not None:
                print(f"曲线图已保存：{out}")


if __name__ == "__main__":
    main()
