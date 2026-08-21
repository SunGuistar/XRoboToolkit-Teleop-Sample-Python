#!/usr/bin/env python3
"""
FR3 episode 回放脚本

加载 BC 录制的 .pkl episode，把其中记录的 target_q 轨迹
通过 ServoJ 或 MoveJ 下发到 FR3 机械臂。

Episode 格式（由 recorder.py 生成）：
    {
        "metadata": {"fps": 10.0, "num_frames": N, "task": "...", ...},
        "frames": [
            {
                "qpos":          当前关节角 (6,),
                "target_q":      目标关节角 (6,),   ← 用这个
                "gripper_cmd":   夹爪命令,
                "image_main":    JPEG bytes,
                "image_pip":     JPEG bytes,
                "timestamp":     时间戳,
                ...
            },
            ...
        ]
    }

使用方式：
  # 默认：ServoJ 10Hz 连续回放
  python FR3/play_episode.py

  # 指定 episode 文件
  python FR3/play_episode.py --episode logs/fr3_bc/第三版/correct/episode_00000.pkl

  # MoveJ 串行回放（每帧等机械臂到位后再下一步）
  python FR3/play_episode.py --mode movej

  # 从第 10 帧开始回放
  python FR3/play_episode.py --start-frame 10

  # 仅回放前 30 帧
  python FR3/play_episode.py --max-frames 30

  # 不连接机械臂，只打印轨迹信息
  python FR3/play_episode.py --dry-run
"""

import argparse
import math
import os
import pickle
import sys
import time

import numpy as np

# ── SDK 导入（与 simple.py 相同策略） ──────────────────────────────
SDK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fairino-python-sdk", "linux")
if os.path.isdir(SDK_PATH):
    sys.path.insert(0, SDK_PATH)

from fairino import Robot  # noqa: E402

# ── 默认常量 ──────────────────────────────────────────────────────
ROBOT_IP = "192.168.58.2"
DEFAULT_EPISODE = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "logs", "fr3_bc", "第三版修正", "raw", "episode_00001.pkl")
)
EXAXIS_POS = [0.0, 0.0, 0.0, 0.0]
SERVO_CMD_TYPE = 1
MOVEJ_VEL_DEG = 15.0

# FR3 关节软限位（度）
JOINT_LIMITS_DEG = [
    (-175.0, 175.0), (-265.0, 85.0), (-150.0, 150.0),
    (-265.0, 85.0),  (-175.0, 175.0), (-175.0, 175.0),
]
JOINT_LIMITS_LOW_RAD  = np.array([math.radians(lo) for lo, _ in JOINT_LIMITS_DEG])
JOINT_LIMITS_HIGH_RAD = np.array([math.radians(hi) for _, hi in JOINT_LIMITS_DEG])


# ── 工具函数 ──────────────────────────────────────────────────────

def connect_robot(robot_ip: str):
    """连接 + 报警检查 + 使能。"""
    robot = Robot.RPC(robot_ip)
    time.sleep(0.5)
    if not getattr(robot, "is_connect", True):
        raise RuntimeError(f"连接机器人失败（{robot_ip}）")
    print(f"[✓] 已连接 {robot_ip}，SDK 版本: {robot.GetSDKVersion()}")

    _, (main_code, sub_code) = robot.GetRobotErrorCode()
    if main_code != 0 or sub_code != 0:
        print(f"[!] 存在报警（主码 {main_code}，子码 {sub_code}），尝试清除...")
        robot.ResetAllError()
        time.sleep(1.0)

    err = robot.Mode(0)
    if err != 0:
        print(f"[!] Mode(0) 返回错误码 {err}（若示教器已在自动模式可忽略）")
    err = robot.RobotEnable(1)
    if err != 0:
        raise RuntimeError(f"RobotEnable(1) 失败，错误码 {err}")
    print("[✓] 机器人已使能")
    return robot


def wait_move_done(robot, timeout=60.0):
    """等待运动队列清空。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        err, queue_len = robot.GetMotionQueueLength()
        if err == 0 and queue_len == 0:
            return True
        time.sleep(0.05)
    print("[!] 等待运动完成超时")
    return False


def clip_joints(q_rad: np.ndarray) -> np.ndarray:
    """关节角裁剪到软限位。"""
    return np.clip(q_rad, JOINT_LIMITS_LOW_RAD, JOINT_LIMITS_HIGH_RAD)


def load_episode(path: str) -> dict:
    """加载 episode pickle。"""
    print(f"[*] 加载 episode: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    meta = data["metadata"]
    frames = data["frames"]
    print(f"[✓] 任务: {meta.get('task', 'N/A')}")
    print(f"    帧数: {meta.get('num_frames', len(frames))}")
    print(f"    FPS: {meta.get('fps', 'N/A')}")
    print(f"    时长: {meta.get('duration', 'N/A'):.2f}s")
    return data


def extract_trajectory(data: dict, start_frame: int = 0, max_frames: int = 0) -> tuple:
    """从 episode 中提取 target_q 轨迹和帧周期。"""
    frames = data["frames"]
    fps = float(data["metadata"].get("fps", 10.0))
    period = 1.0 / fps

    if start_frame < 0:
        start_frame = 0
    if max_frames > 0:
        end_frame = min(start_frame + max_frames, len(frames))
    else:
        end_frame = len(frames)

    trajectory = []
    for i in range(start_frame, end_frame):
        f = frames[i]
        target_q = np.asarray(f["target_q"], dtype=np.float64)
        gripper_cmd = float(f.get("gripper_cmd", 0.0))
        trajectory.append((i, target_q, gripper_cmd))

    print(f"[*] 回放范围: frame {start_frame} ~ {end_frame - 1}（共 {len(trajectory)} 帧）")
    print(f"    帧周期: {period * 1000:.1f}ms（{fps:.1f}Hz）")
    return trajectory, period


# ── 回放方式 ──────────────────────────────────────────────────────

def play_servo(robot, trajectory: list, period: float, cmd_type: int = SERVO_CMD_TYPE):
    """ServoJ 连续回放（按原 FPS 节奏）。"""
    print("\n[*] ServoJ 连续回放模式")

    err = robot.ServoMoveStart(cmdType=cmd_type)
    if err != 0:
        print(f"[!] ServoMoveStart 失败，错误码 {err}")
        return
    print("[✓] ServoMoveStart 成功")

    ok_count = 0
    fail_count = 0
    t_start = time.perf_counter()

    try:
        for frame_idx, target_q, gripper in trajectory:
            t_cmd = time.perf_counter()
            q_safe = clip_joints(target_q)
            q_deg = np.rad2deg(q_safe).tolist()

            err = robot.ServoJ(
                q_deg, EXAXIS_POS,
                acc=0.0, vel=0.0, cmdT=period,
                filterT=0.0, gain=0.0, id=0,
                cmdType=cmd_type,
            )
            if err == 0:
                ok_count += 1
            else:
                fail_count += 1
                if fail_count <= 3 or fail_count % 20 == 0:
                    print(f"[!] frame {frame_idx}: ServoJ 失败，错误码 {err}")

            # 周期对齐
            elapsed = time.perf_counter() - t_cmd
            sleep_time = max(0.0, period - elapsed)
            time.sleep(sleep_time)

    finally:
        # 回最后一帧再退出伺服
        if trajectory:
            last_q = np.rad2deg(clip_joints(trajectory[-1][1])).tolist()
            robot.ServoJ(last_q, EXAXIS_POS, acc=0.0, vel=0.0, cmdT=period,
                         filterT=0.0, gain=0.0, id=0, cmdType=cmd_type)
            time.sleep(0.2)
        err = robot.ServoMoveEnd(cmdType=cmd_type)
        print(f"[✓] ServoMoveEnd 错误码: {err}")

    total = ok_count + fail_count
    dt = time.perf_counter() - t_start
    print(f"\n[✓] ServoJ 回放结束：{total} 帧，成功 {ok_count}，失败 {fail_count}，"
          f"实际平均周期 {dt / max(total, 1) * 1000:.1f}ms")


def play_movej(robot, trajectory: list, vel_deg: float = MOVEJ_VEL_DEG):
    """MoveJ 串行回放（每帧等运动完成）。"""
    print("\n[*] MoveJ 串行回放模式")

    ok_count = 0
    fail_count = 0
    t_start = time.perf_counter()

    for frame_idx, target_q, gripper in trajectory:
        q_safe = clip_joints(target_q)
        q_deg = np.rad2deg(q_safe).tolist()

        err = robot.MoveJ(q_deg, tool=0, user=0, vel=vel_deg)
        if err != 0:
            print(f"[!] frame {frame_idx}: MoveJ 失败，错误码 {err}")
            fail_count += 1
            continue

        if wait_move_done(robot):
            ok_count += 1
        else:
            fail_count += 1
            print(f"[!] frame {frame_idx}: 等待运动完成超时")

    dt = time.perf_counter() - t_start
    total = len(trajectory)
    print(f"\n[✓] MoveJ 回放结束：{total} 帧，成功 {ok_count}，失败 {fail_count}，"
          f"总耗时 {dt:.1f}s")


def dry_run(trajectory: list, period: float):
    """不连机械臂，仅打印轨迹信息。"""
    print("\n[*] Dry run 模式（不连机械臂）")
    print(f"    第 1 帧 target_q (deg): {np.rad2deg(trajectory[0][1]).round(2).tolist()}")
    print(f"    最后 1 帧 target_q (deg): {np.rad2deg(trajectory[-1][1]).round(2).tolist()}")
    print(f"    预计总时长: {period * len(trajectory):.2f}s")


# ── 入口 ──────────────────────────────────────────────────────────

def confirm_motion(mode: str, yes=False):
    if yes:
        return True
    print(f"\n[安全确认] 即将用 {mode} 回放 episode，电机会运动。"
          "\n  请确认机械臂周围无障碍物、急停按钮可随时触达。")
    answer = input("输入 yes 继续，其他任意键取消: ").strip().lower()
    return answer == "yes"


def main():
    parser = argparse.ArgumentParser(
        description="FR3 episode 轨迹回放（从 .pkl 加载 target_q 并驱动机械臂）"
    )
    parser.add_argument("--episode", default=DEFAULT_EPISODE,
                        help=f"episode pickle 路径（默认 {DEFAULT_EPISODE}）")
    parser.add_argument("--robot-ip", default=ROBOT_IP,
                        help=f"FR3 控制箱 IP（默认 {ROBOT_IP}）")
    parser.add_argument("--mode", choices=["servo", "movej"], default="movej",
                        help="回放方式：movej=每帧等完成（默认），servo=按原 FPS 连续 ServoJ")
    parser.add_argument("--movej-vel", type=float, default=MOVEJ_VEL_DEG,
                        help=f"MoveJ 速度 °/s（默认 {MOVEJ_VEL_DEG}）")
    parser.add_argument("--cmd-type", type=int, default=SERVO_CMD_TYPE, choices=[0, 1],
                        help="ServoJ 传输方式：0=XML-RPC，1=UDP（默认 1）")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="从第几帧开始回放（默认 0）")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多回放多少帧，0 表示全部（默认 0）")
    parser.add_argument("--dry-run", action="store_true",
                        help="不连接机械臂，仅打印轨迹信息")
    parser.add_argument("--yes", action="store_true",
                        help="跳过安全确认")
    args = parser.parse_args()

    # 加载 episode
    data = load_episode(args.episode)
    trajectory, period = extract_trajectory(
        data,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
    )

    if not trajectory:
        print("[!] 轨迹为空")
        sys.exit(1)

    # Dry run
    if args.dry_run:
        dry_run(trajectory, period)
        sys.exit(0)

    # 连接机器人
    robot = connect_robot(args.robot_ip)

    try:
        if args.mode == "servo":
            play_servo(robot, trajectory, period, cmd_type=args.cmd_type)
        else:
            play_movej(robot, trajectory, vel_deg=args.movej_vel)
    except KeyboardInterrupt:
        print("\n[!] 用户中断，停止运动")
        try:
            robot.StopMotion()
            robot.ServoMoveEnd(cmdType=args.cmd_type)
        except Exception:
            pass
    finally:
        try:
            robot.CloseRPC()
        except Exception:
            pass
        print("[✓] 连接已关闭")


if __name__ == "__main__":
    main()
