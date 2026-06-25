"""
Linker Hand L20 简单控制脚本（CAN 通讯）。

使用前准备：
  1. 安装 LinkerHand Python SDK：
       git clone https://github.com/linker-bot/linkerhand-python-sdk.git
       cd linkerhand-python-sdk && pip install -r requirements.txt
  2. 编辑 SDK 内 LinkerHand/config/setting.yaml（左右手型号、CAN 口、sudo 密码）
  3. 连接 USB-CAN 适配器并上电 L20（Linux 下通常为 can0）
  4. 确保没有其他程序同时控制灵巧手（ROS、GUI 等）

常用命令：
  # 连接测试 + 打印状态
  python scripts/hardware/test_linker_l20.py --status

  # 演示：依次执行 张开 → 握拳 → OK → 点赞
  python scripts/hardware/test_linker_l20.py --demo

  # 执行单个预设手势
  python scripts/hardware/test_linker_l20.py --gesture open
  python scripts/hardware/test_linker_l20.py --gesture fist

  # 自定义 20 维关节目标（0-255）
  python scripts/hardware/test_linker_l20.py --pose 255 255 255 255 255 255 10 100 180 240 245 255 255 255 255 255 255 255 255 255

  # 左手 / 指定 CAN 口 / SDK 路径
  python scripts/hardware/test_linker_l20.py --hand-type left --can can0 --sdk-path ~/linkerhand-python-sdk
"""

from __future__ import annotations

import sys
import time
from typing import Literal

import tyro

from xrobotoolkit_teleop.hardware.interface.linker_l20 import (
    L20_JOINT_NAMES,
    L20_PRESET_POSES,
    LinkerL20Interface,
)


def _print_status(hand: LinkerL20Interface) -> None:
    print("\n=== Linker L20 状态 ===")
    print(f"  序列号: {hand.get_serial_number()}")
    print(f"  固件版本: {hand.get_version()}")

    state = hand.get_state()
    if state:
        print(f"  关节位置 ({len(state)}): {state}")
        for name, value in zip(L20_JOINT_NAMES, state, strict=False):
            print(f"    {name:18s}: {value}")

    fault = hand.get_fault()
    if fault:
        print(f"  故障码: {fault}")
        if any(int(f) != 0 for f in fault):
            print("  提示: 存在非零故障码，可尝试 --clear-faults")

    temp = hand.get_temperature()
    if temp:
        print(f"  电机温度: {temp}")

    torque = hand.get_torque()
    if torque:
        print(f"  扭矩限制: {torque}")

    try:
        force = hand.get_force()
        if force:
            print(f"  触觉/力传感: {force}")
    except Exception as exc:
        print(f"  触觉数据不可用: {exc}")


def _run_demo(hand: LinkerL20Interface, hold_s: float, pause_s: float) -> None:
    sequence = ["open", "fist", "ok", "thumbs_up"]
    print("\n=== 手势演示 ===")
    print(f"  顺序: {' → '.join(sequence)}")
    print("  3 秒后开始（Ctrl+C 中止）...")
    try:
        time.sleep(3.0)
    except KeyboardInterrupt:
        print("已取消。")
        return

    for name in sequence:
        print(f"\n-> {name}")
        hand.move_preset(name)
        time.sleep(hold_s)

    print("\n-> 回到张开")
    hand.move_preset("open")
    time.sleep(pause_s)
    print("演示完成。")


def main(
    hand_type: Literal["left", "right"] = "right",
    can: str = "can0",
    sdk_path: str | None = None,
    status: bool = False,
    demo: bool = False,
    gesture: str | None = None,
    pose: tuple[int, ...] | None = None,
    clear_faults: bool = False,
    hold_s: float = 2.0,
    pause_s: float = 0.5,
    reset_on_exit: bool = False,
) -> None:
    """
    Args:
        hand_type: 左手或右手。
        can: CAN 接口名（Linux 通常为 can0）。
        sdk_path: LinkerHand Python SDK 仓库根目录（也可设 LINKERHAND_SDK_PATH）。
        status: 仅连接并打印状态。
        demo: 依次执行预设手势演示。
        gesture: 执行单个预设手势（open / fist / ok / thumbs_up）。
        pose: 自定义 20 维关节目标（0-255）。
        clear_faults: 清除电机故障码（L20 支持）。
        hold_s: 每个手势保持时间（秒）。
        pause_s: 手势切换间隔（秒）。
        reset_on_exit: 脚本退出时是否回到 open（默认 False；demo 演示结束仍会主动张开）。
    """
    if not any([status, demo, gesture, pose, clear_faults]):
        demo = True

    print(f"=== Linker L20 @ {can} ({hand_type}) ===")

    hand: LinkerL20Interface | None = None
    try:
        hand = LinkerL20Interface(hand_type=hand_type, can=can, sdk_path=sdk_path)
        print("OK: 灵巧手已连接。")

        if clear_faults:
            print("正在清除故障码...")
            hand.clear_faults()
            time.sleep(0.2)

        if pose is not None:
            hand.move(list(pose))
            time.sleep(hold_s)
            _print_status(hand)
            return

        if gesture is not None:
            hand.move_preset(gesture)
            time.sleep(hold_s)
            _print_status(hand)
            return

        if status:
            _print_status(hand)
            return

        if demo:
            _run_demo(hand, hold_s=hold_s, pause_s=pause_s)
            _print_status(hand)

    except ImportError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\n用户中断。")
        reset_on_exit = True
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    finally:
        if hand is not None:
            if reset_on_exit:
                try:
                    hand.move_preset("open")
                    time.sleep(0.3)
                except Exception:
                    pass
            hand.close()
            print("连接已关闭。")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--list-gestures":
        print("可用预设手势:")
        for name, pose in L20_PRESET_POSES.items():
            print(f"  {name:12s}  ({len(pose)} joints)")
        raise SystemExit(0)
    tyro.cli(main)
