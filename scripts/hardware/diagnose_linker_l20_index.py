"""
Linker L20 食指诊断脚本。

逐关节检测食指 3 个自由度（根部弯曲 / 侧摆 / 指尖），并与中指对照，
判断是否存在通讯、编码器或电机故障。

用法：
  python scripts/hardware/diagnose_linker_l20_index.py
  python scripts/hardware/diagnose_linker_l20_index.py --hand-type right --settle-s 2.0
  python scripts/hardware/diagnose_linker_l20_index.py --no-motion   # 仅读状态，不运动
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Literal

import tyro

from xrobotoolkit_teleop.hardware.interface.linker_l20 import (
    L20_JOINT_NAMES,
    L20_PRESET_POSES,
    LinkerL20Interface,
)

FAULT_LABELS = {
    0: "正常",
    1: "电流过载",
    2: "温度过高",
    3: "编码错误",
    4: "过压/欠压",
}

# L20 食指 / 中指关节索引（与 L20_JOINT_NAMES 一致）
INDEX_JOINTS = {
    "index_mcp": 1,
    "index_abduction": 6,
    "index_tip": 16,
}
MIDDLE_JOINTS = {
    "middle_mcp": 2,
    "middle_abduction": 7,
    "middle_tip": 17,
}


@dataclass
class JointCheckResult:
    finger: str
    joint_name: str
    joint_index: int
    open_cmd: int
    bent_cmd: int
    open_actual: float
    bent_actual: float
    delta: float
    min_delta: float
    passed: bool
    fault_at_check: int | None
    temp_at_check: float | None
    note: str = ""


def _pose_from_open() -> list[int]:
    return list(L20_PRESET_POSES["open"])


def _set_joint(pose: list[int], index: int, value: int) -> list[int]:
    out = list(pose)
    out[index] = int(value)
    return out


def _read_joint(state: list[float], index: int) -> float:
    return float(state[index])


def _fault_label(code: float | int) -> str:
    code_int = int(code)
    return FAULT_LABELS.get(code_int, f"未知({code_int})")


def _print_finger_snapshot(
    label: str,
    joint_map: dict[str, int],
    state: list[float],
    temp: list[float] | None,
    fault: list[float] | None,
) -> None:
    print(f"\n--- {label} 当前快照 ---")
    for name, idx in joint_map.items():
        s = _read_joint(state, idx)
        t = _read_joint(temp, idx) if temp and len(temp) > idx else None
        f = int(_read_joint(fault, idx)) if fault and len(fault) > idx else None
        extra = ""
        if t is not None:
            extra += f"  温度={t:.0f}°C"
        if f is not None and f != 0:
            extra += f"  故障={_fault_label(f)}"
        print(f"  [{idx:2d}] {name:18s}  位置={s:6.1f}{extra}")


def _check_joint(
    hand: LinkerL20Interface,
    finger: str,
    joint_name: str,
    joint_index: int,
    open_pose: list[int],
    open_cmd: int,
    bent_cmd: int,
    settle_s: float,
    min_delta: float,
) -> JointCheckResult:
    hand.move(_set_joint(open_pose, joint_index, open_cmd))
    time.sleep(settle_s)
    state_open = hand.get_state()
    open_actual = _read_joint(state_open, joint_index)

    bent_pose = _set_joint(open_pose, joint_index, bent_cmd)
    hand.move(bent_pose)
    time.sleep(settle_s)
    state_bent = hand.get_state()
    bent_actual = _read_joint(state_bent, joint_index)

    temp = hand.get_temperature()
    fault = hand.get_fault()
    fault_val = int(_read_joint(fault, joint_index)) if fault and len(fault) > joint_index else None
    temp_val = _read_joint(temp, joint_index) if temp and len(temp) > joint_index else None

    delta = abs(bent_actual - open_actual)
    passed = delta >= min_delta and (fault_val in (None, 0))

    note = ""
    if fault_val not in (None, 0):
        note = f"故障码: {_fault_label(fault_val)}"
    elif delta < min_delta:
        note = f"运动幅度不足 (Δ={delta:.1f} < {min_delta:.1f})"

    return JointCheckResult(
        finger=finger,
        joint_name=joint_name,
        joint_index=joint_index,
        open_cmd=open_cmd,
        bent_cmd=bent_cmd,
        open_actual=open_actual,
        bent_actual=bent_actual,
        delta=delta,
        min_delta=min_delta,
        passed=passed,
        fault_at_check=fault_val,
        temp_at_check=temp_val,
        note=note,
    )


def _print_result(r: JointCheckResult) -> None:
    status = "PASS" if r.passed else "FAIL"
    print(
        f"  [{status}] {r.finger}/{r.joint_name} (J{r.joint_index})  "
        f"cmd {r.open_cmd}->{r.bent_cmd}  "
        f"actual {r.open_actual:.1f}->{r.bent_actual:.1f}  Δ={r.delta:.1f}"
    )
    if r.temp_at_check is not None:
        print(f"         温度={r.temp_at_check:.0f}°C", end="")
    if r.fault_at_check not in (None, 0):
        print(f"  故障={_fault_label(r.fault_at_check)}", end="")
    print()
    if r.note:
        print(f"         -> {r.note}")


def _run_index_curl_test(hand: LinkerL20Interface, settle_s: float) -> JointCheckResult | None:
    """食指整体弯曲：根部+指尖同时收拢，其余保持 open。"""
    open_pose = _pose_from_open()
    curl_pose = _pose_from_open()
    curl_pose[INDEX_JOINTS["index_mcp"]] = 0
    curl_pose[INDEX_JOINTS["index_tip"]] = 0

    hand.move(open_pose)
    time.sleep(settle_s)
    state_open = hand.get_state()
    mcp_open = _read_joint(state_open, INDEX_JOINTS["index_mcp"])
    tip_open = _read_joint(state_open, INDEX_JOINTS["index_tip"])

    hand.move(curl_pose)
    time.sleep(settle_s)
    state_curl = hand.get_state()
    mcp_curl = _read_joint(state_curl, INDEX_JOINTS["index_mcp"])
    tip_curl = _read_joint(state_curl, INDEX_JOINTS["index_tip"])

    mcp_delta = abs(mcp_curl - mcp_open)
    tip_delta = abs(tip_curl - tip_open)
    combined_delta = mcp_delta + tip_delta

    passed = mcp_delta >= 25 and tip_delta >= 25
    return JointCheckResult(
        finger="index",
        joint_name="curl_mcp+tip",
        joint_index=-1,
        open_cmd=255,
        bent_cmd=0,
        open_actual=mcp_open,
        bent_actual=mcp_curl,
        delta=combined_delta,
        min_delta=50.0,
        passed=passed,
        fault_at_check=None,
        temp_at_check=None,
        note=f"根部Δ={mcp_delta:.1f}, 指尖Δ={tip_delta:.1f} (期望各≥25)",
    )


def main(
    hand_type: Literal["left", "right"] = "right",
    can: str = "can0",
    sdk_path: str | None = None,
    settle_s: float = 1.5,
    min_delta: float = 30.0,
    no_motion: bool = False,
    compare_middle: bool = True,
    reset_on_exit: bool = True,
) -> None:
    """
    Args:
        hand_type: 左手或右手。
        can: CAN 接口名。
        sdk_path: LinkerHand SDK 路径。
        settle_s: 每个动作等待稳定时间（秒）。
        min_delta: 单关节判定通过的最小位移（0-255 刻度）。
        no_motion: 仅读取食指/中指状态，不发送运动指令。
        compare_middle: 同时检测中指作为对照。
        reset_on_exit: 结束后回到 open 姿态。
    """
    print(f"=== Linker L20 食指诊断 @ {can} ({hand_type}) ===")
    print("请确保灵巧手周围无障碍，检测时食指/中指会缓慢运动。")
    if not no_motion:
        print("3 秒后开始...")
        try:
            time.sleep(3.0)
        except KeyboardInterrupt:
            print("已取消。")
            return

    hand: LinkerL20Interface | None = None
    results: list[JointCheckResult] = []

    try:
        hand = LinkerL20Interface(hand_type=hand_type, can=can, sdk_path=sdk_path)
        print("OK: 已连接。\n")

        state = hand.get_state()
        temp = hand.get_temperature()
        fault = hand.get_fault()

        _print_finger_snapshot("食指", INDEX_JOINTS, state, temp, fault)
        if compare_middle:
            _print_finger_snapshot("中指(对照)", MIDDLE_JOINTS, state, temp, fault)

        if fault:
            index_faults = [
                (name, idx, int(_read_joint(fault, idx)))
                for name, idx in INDEX_JOINTS.items()
                if len(fault) > idx and int(_read_joint(fault, idx)) != 0
            ]
            if index_faults:
                print("\n[!] 食指相关关节已有故障码：")
                for name, idx, code in index_faults:
                    print(f"    J{idx} {name}: {_fault_label(code)}")
                print("    尝试: python scripts/hardware/test_linker_l20.py --clear-faults")

        if no_motion:
            print("\n(--no-motion) 跳过运动检测。")
            return

        open_pose = _pose_from_open()
        hand.move(open_pose)
        time.sleep(settle_s)

        print("\n=== 单关节运动检测（open 姿态基准，其余关节不变）===")
        test_specs = [
            ("index", "index_mcp", INDEX_JOINTS["index_mcp"], 255, 0),
            ("index", "index_abduction", INDEX_JOINTS["index_abduction"], 100, 200),
            ("index", "index_tip", INDEX_JOINTS["index_tip"], 255, 0),
        ]
        if compare_middle:
            test_specs.extend(
                [
                    ("middle", "middle_mcp", MIDDLE_JOINTS["middle_mcp"], 255, 0),
                    ("middle", "middle_abduction", MIDDLE_JOINTS["middle_abduction"], 100, 200),
                    ("middle", "middle_tip", MIDDLE_JOINTS["middle_tip"], 255, 0),
                ]
            )

        for finger, jname, jidx, open_cmd, bent_cmd in test_specs:
            print(f"\n-> 测试 {finger}/{jname} ...")
            r = _check_joint(
                hand, finger, jname, jidx, open_pose, open_cmd, bent_cmd, settle_s, min_delta
            )
            results.append(r)
            _print_result(r)
            hand.move(open_pose)
            time.sleep(settle_s * 0.5)

        print("\n=== 食指整体弯曲检测（根部+指尖同时收拢）===")
        curl_result = _run_index_curl_test(hand, settle_s)
        if curl_result:
            results.append(curl_result)
            _print_result(curl_result)

        index_results = [r for r in results if r.finger == "index"]
        middle_results = [r for r in results if r.finger == "middle"]
        index_failed = [r for r in index_results if not r.passed]
        middle_failed = [r for r in middle_results if not r.passed]

        print("\n" + "=" * 60)
        print("诊断结论")
        print("=" * 60)
        if not index_failed:
            print("食指：全部检测通过，未发现明显异常。")
        else:
            print(f"食指：{len(index_failed)}/{len(index_results)} 项未通过：")
            for r in index_failed:
                print(f"  - {r.joint_name}: {r.note or '未达标'}")

        if compare_middle and middle_results:
            if not middle_failed:
                print("中指(对照)：全部通过。")
                if index_failed:
                    print("=> 更可能是食指硬件/线路问题，而非 CAN 通讯整体故障。")
            elif index_failed and not middle_failed:
                print("=> 中指正常、食指异常，建议重点检查食指电机/编码器/排线。")
            elif index_failed and middle_failed:
                print("=> 食指与中指均有异常，可能是 CAN/供电或 SDK 配置问题。")

        print("\n关节索引对照：")
        for name, idx in INDEX_JOINTS.items():
            print(f"  J{idx:2d}  {name}  ({L20_JOINT_NAMES[idx]})")

    except ImportError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        print("\n用户中断。")
    finally:
        if hand is not None:
            if reset_on_exit:
                try:
                    hand.move_preset("open")
                    time.sleep(0.3)
                except Exception:
                    pass
            hand.close()
            print("\n连接已关闭。")


if __name__ == "__main__":
    tyro.cli(main)
