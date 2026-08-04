#!/usr/bin/env python3
"""
法奥 FAIRINO FR3 测试脚本（fairino-python-sdk）

用于验证 UR5e -> FR3 遥操作迁移所需的全部底层接口，共 4 类测试：

  1. data   —— 只读数据测试：核对迁移所需的每个 SDK 接口与返回格式
               （GetActualJointPosRadian/Degree、GetActualTCPPose、
                 GetJointSoftLimitDeg、GetRobotErrorCode、GetMotionQueueLength 等）
  2. servo  —— 实时伺服测试：ServoMoveStart + 100Hz ServoJ + ServoMoveEnd
               （等价遥操作 run_robot_thread 的底层链路，默认 J6 微幅摆动）
  3. joints —— 每个关节微动测试：J1~J6 逐个 MoveJ ±delta 度后回位
               （验证 MoveJ/等待完成/软限位裁剪，用于 --reset 回 Home 链路）
  4. ik     —— placo IK 离线测试（不驱动电机）：FK、IK 往返、笛卡尔偏移三种
               （验证 fr3_config.py 的 URDF/joint_names/tcp_frame 是否可用）

使用前准备：
  1. fairino-python-sdk 已解压到本目录下的 fairino-python-sdk/ 文件夹（脚本自动处理导入路径）。
  2. 网线直连控制箱：电脑网卡与控制箱同网段（默认控制箱 IP 为 192.168.58.2）。
  3. 控制箱上电，示教器/APP 上确认无报警、已使能。

单位说明：SDK 位置 mm，姿态 °（度），关节角 °；脚本内部按需换算弧度。

常用命令：
  python FR3/simple.py                     # 默认：只读数据测试（不动电机，安全）
  python FR3/simple.py --test ik           # placo IK 测试（不动电机，需先填 FR3 URDF）
  python FR3/simple.py --test servo        # ServoJ 实时伺服测试（动电机，需确认）
  python FR3/simple.py --test joints       # 每关节微动测试（动电机，需确认）
  python FR3/simple.py --test all          # data -> ik -> joints -> servo
  python FR3/simple.py --test servo --hold # ServoJ 只保持当前位置，不摆动
"""

import argparse
import math
import os
import sys
import time

# 将 SDK 的 linux 目录加入搜索路径（from fairino import Robot 需要）
SDK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fairino-python-sdk", "linux")
if os.path.isdir(SDK_PATH):
    sys.path.insert(0, SDK_PATH)

from fairino import Robot  # noqa: E402

ROBOT_IP = "192.168.58.2"  # 控制箱 IP
EXAXIS_POS = [0.0, 0.0, 0.0, 0.0]  # 无外部轴占位

# 遥操作迁移实际使用的接口清单（data 测试逐一核对）
REQUIRED_INTERFACES = [
    "ServoMoveStart",       # 进入伺服模式（run_robot_thread）
    "ServoJ",               # 实时关节伺服（run_robot_thread，等价 RTDE servoJ）
    "ServoMoveEnd",         # 退出伺服模式（reset_to_home / close）
    "MoveJ",                # 回 Home（reset_to_home）
    "GetActualJointPosRadian",  # get_current_joint_positions（弧度）
    "GetActualJointPosDegree",  # save-home / 数据核对（度）
    "GetActualTCPPose",     # IK 测试 FK 对照
    "GetJointSoftLimitDeg", # 关节软限位（fr3_config joint_limits）
    "GetMotionQueueLength", # 等待 MoveJ 完成
    "GetRobotErrorCode",    # 启动报警检查
    "ResetAllError",        # 启动报警清除
    "RobotEnable",          # 上使能
    "Mode",                 # 自动模式
    "StopMotion",           # 急停兜底（close）
    "CloseRPC",             # 断开连接（close）
]


def wait_move_done(robot, timeout=30.0):
    """等待运动队列清空（即运动完成）"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        err, queue_len = robot.GetMotionQueueLength()
        if err == 0 and queue_len == 0:
            return True
        time.sleep(0.05)
    print("[!] 等待运动完成超时")
    return False


def check(ret, msg):
    """SDK 运动/设置类接口返回错误码，0 表示成功"""
    if isinstance(ret, int) and ret != 0:
        print(f"[!] {msg} 失败，错误码: {ret}")
        return False
    return True


def connect(robot_ip):
    """连接 + 报警检查 + 使能，失败直接退出。"""
    robot = Robot.RPC(robot_ip)
    time.sleep(0.5)  # 等待 CNDE 状态流建立
    if not getattr(robot, "is_connect", True):
        print(f"[!] 连接机器人失败（{robot_ip}），请检查 IP 与网线连接")
        sys.exit(1)
    print(f"[✓] 已连接 {robot_ip}，SDK 版本: {robot.GetSDKVersion()}")

    _, (main_code, sub_code) = robot.GetRobotErrorCode()
    if main_code != 0 or sub_code != 0:
        print(f"[!] 机器人存在报警，主错误码: {main_code}，子错误码: {sub_code}，尝试清除...")
        robot.ResetAllError()
        time.sleep(1)

    err = robot.Mode(0)  # 自动模式
    if err != 0:
        print(f"[!] Mode(0) 返回错误码 {err}（若示教器已在自动模式可忽略）")
    err = robot.RobotEnable(1)
    if err != 0:
        print(f"[!] RobotEnable(1) 失败，错误码 {err}，请确认无急停/报警")
        sys.exit(1)
    return robot


# ======================================================================
# 测试 1：只读数据测试 —— 核对迁移所需的每个接口与返回格式
# ======================================================================

def test_data(robot):
    print("\n" + "=" * 60)
    print("测试 1/4：数据接口测试（只读，不动电机）")
    print("=" * 60)
    failed = []

    # --- 1.1 接口存在性核对（迁移代码用到的全部接口） ---
    print("\n-- 迁移所需接口核对 --")
    for name in REQUIRED_INTERFACES:
        ok = callable(getattr(robot, name, None))
        print(f"  {'[✓]' if ok else '[!]'} Robot.{name}")
        if not ok:
            failed.append(name)

    # --- 1.2 关节角（度） ---
    print("\n-- GetActualJointPosDegree（度） --")
    ret = robot.GetActualJointPosDegree(1)
    if isinstance(ret, tuple) and ret[0] == 0:
        q_deg = ret[1]
        print(f"  [✓] q_deg = {[round(v, 3) for v in q_deg]}")
    else:
        print(f"  [!] 失败，返回: {ret}")
        q_deg = None
        failed.append("GetActualJointPosDegree")

    # --- 1.3 关节角（弧度）—— get_current_joint_positions 的数据源 ---
    print("\n-- GetActualJointPosRadian（弧度，迁移用） --")
    ret = robot.GetActualJointPosRadian(1)
    if isinstance(ret, tuple) and ret[0] == 0 and ret[1] is not None:
        q_rad = ret[1]
        print(f"  [✓] q_rad = {[round(v, 4) for v in q_rad]}")
        # 交叉校验：rad 应等于 deg2rad(deg)
        if q_deg is not None:
            diff = max(abs(math.radians(d) - r) for d, r in zip(q_deg, q_rad))
            if diff < 1e-3:
                print(f"  [✓] 度/弧度交叉校验通过（最大偏差 {diff:.2e} rad）")
            else:
                print(f"  [!] 度/弧度不一致，最大偏差 {diff:.4f} rad")
                failed.append("rad/deg 一致性")
    else:
        print(f"  [!] 失败，返回: {ret}")
        failed.append("GetActualJointPosRadian")

    # --- 1.4 TCP 位姿（IK 测试 FK 对照用） ---
    print("\n-- GetActualTCPPose（mm，°） --")
    ret = robot.GetActualTCPPose()
    if isinstance(ret, tuple) and ret[0] == 0:
        print(f"  [✓] [x,y,z,rx,ry,rz] = {[round(v, 2) for v in ret[1]]}")
    else:
        print(f"  [!] 失败，返回: {ret}")
        failed.append("GetActualTCPPose")

    # --- 1.5 关节软限位（fr3_config joint_limits 数据源） ---
    print("\n-- GetJointSoftLimitDeg（°，共 12 值） --")
    ret = robot.GetJointSoftLimitDeg(1)
    if isinstance(ret, tuple) and ret[0] == 0 and ret[1] is not None:
        lim = ret[1]
        print(f"  [✓] 原始 12 值: {[round(v, 2) for v in lim]}")
        # 实机验证：布局为逐关节交替 [low1, high1, low2, high2, ..., low6, high6]
        low = [lim[2 * i] for i in range(6)]
        high = [lim[2 * i + 1] for i in range(6)]
        print(f"      下限(°): {[round(v, 2) for v in low]}")
        print(f"      上限(°): {[round(v, 2) for v in high]}")
        print("      请将上表换算为弧度后填入 fr3_config.py 的 joint_limits。")
    else:
        print(f"  [!] 失败，返回: {ret}")
        failed.append("GetJointSoftLimitDeg")

    # --- 1.6 运动队列长度（MoveJ 等待完成用） ---
    print("\n-- GetMotionQueueLength --")
    ret = robot.GetMotionQueueLength()
    if isinstance(ret, tuple) and ret[0] == 0:
        print(f"  [✓] 队列长度 = {ret[1]}")
    else:
        print(f"  [!] 失败，返回: {ret}")
        failed.append("GetMotionQueueLength")

    # --- 1.7 错误码（启动自检用） ---
    print("\n-- GetRobotErrorCode --")
    ret = robot.GetRobotErrorCode()
    if isinstance(ret, tuple) and ret[0] == 0:
        main_code, sub_code = ret[1]
        state = "无报警" if (main_code == 0 and sub_code == 0) else f"有报警 {main_code}/{sub_code}"
        print(f"  [✓] {state}")
    else:
        print(f"  [!] 失败，返回: {ret}")
        failed.append("GetRobotErrorCode")

    print("\n" + "-" * 60)
    if failed:
        print(f"数据测试未通过项: {failed}")
        return False
    print("数据测试全部通过。")
    return True


# ======================================================================
# 测试 2：ServoJ 实时伺服测试（遥操作底层链路）
# ======================================================================

def test_servo(robot, hz=100.0, duration=2.0, amplitude_deg=0.5, servo_joint=6,
               hold=False, cmd_type=1):
    print("\n" + "=" * 60)
    print(f"测试 2/4：ServoJ 实时伺服测试（{hz:.0f}Hz，{duration:.1f}s，"
          f"cmdType={cmd_type}{'，保持模式' if hold else f'，J{servo_joint} ±{amplitude_deg}°'}）")
    print("=" * 60)

    ret = robot.GetActualJointPosDegree(1)
    if not (isinstance(ret, tuple) and ret[0] == 0):
        print(f"[!] 读取当前关节角失败: {ret}")
        return False
    q0 = list(ret[1])
    print(f"起始关节角(°): {[round(v, 2) for v in q0]}")

    period = 1.0 / hz
    n_cycles = int(duration * hz)
    err = robot.ServoMoveStart(cmdType=cmd_type)
    if err != 0:
        print(f"[!] ServoMoveStart 失败，错误码 {err}")
        return False
    print("[✓] ServoMoveStart 成功，开始 ServoJ 流...")

    ok_count = 0
    fail_count = 0
    t_start = time.perf_counter()
    try:
        for i in range(n_cycles):
            t_cmd = time.perf_counter()
            q_cmd = q0[:]
            if not hold:
                # 单关节正弦微动（频率 0.5Hz，幅值 amplitude_deg）
                q_cmd[servo_joint - 1] = q0[servo_joint - 1] + amplitude_deg * math.sin(
                    2.0 * math.pi * 0.5 * (i * period)
                )
            err = robot.ServoJ(
                q_cmd, EXAXIS_POS,
                acc=0.0, vel=0.0, cmdT=period, filterT=0.0, gain=0.0,
                id=0, cmdType=cmd_type,
            )
            if err == 0:
                ok_count += 1
            else:
                fail_count += 1
                if fail_count == 1:
                    print(f"[!] ServoJ 首次失败，错误码 {err}")
            # 周期对齐（等价 run_robot_thread 的 pacing）
            elapsed = time.perf_counter() - t_cmd
            time.sleep(max(0.0, period - elapsed))
    finally:
        # 收尾：回到起始位再退出伺服模式
        robot.ServoJ(q0, EXAXIS_POS, acc=0.0, vel=0.0, cmdT=period,
                     filterT=0.0, gain=0.0, id=0, cmdType=cmd_type)
        time.sleep(0.2)
        err = robot.ServoMoveEnd(cmdType=cmd_type)
        print(f"ServoMoveEnd 错误码: {err}")

    total = ok_count + fail_count
    actual_dt = (time.perf_counter() - t_start) / max(total, 1)
    print(f"下发 {total} 条：成功 {ok_count}，失败 {fail_count}，"
          f"平均周期 {actual_dt * 1000:.2f}ms（目标 {period * 1000:.0f}ms）")
    if fail_count == 0 and ok_count == total:
        print("ServoJ 测试通过。")
        return True
    if fail_count == total:
        print("[!] ServoJ 全部失败：检查自动模式/使能/报警，或改 --cmd-type 0（XML-RPC）。")
    return fail_count == 0


# ======================================================================
# 测试 3：每个关节微动测试（MoveJ 链路，--reset 回 Home 用）
# ======================================================================

def test_joints(robot, delta_deg=2.0, vel_deg=10.0):
    print("\n" + "=" * 60)
    print(f"测试 3/4：每关节微动测试（MoveJ ±{delta_deg}°，速度 {vel_deg}°/s）")
    print("=" * 60)

    ret = robot.GetActualJointPosDegree(1)
    if not (isinstance(ret, tuple) and ret[0] == 0):
        print(f"[!] 读取当前关节角失败: {ret}")
        return False
    q0 = list(ret[1])
    print(f"起始关节角(°): {[round(v, 2) for v in q0]}")

    # 软限位裁剪（失败则不裁剪，仅提示）
    low, high = None, None
    ret = robot.GetJointSoftLimitDeg(1)
    if isinstance(ret, tuple) and ret[0] == 0 and ret[1] is not None:
        lim = ret[1]
        # 实机验证布局：逐关节交替 [low1, high1, low2, high2, ...]
        low = [lim[2 * i] for i in range(6)]
        high = [lim[2 * i + 1] for i in range(6)]
    else:
        print("[!] 读取软限位失败，本次不做限位裁剪。")

    all_ok = True
    try:
        for i in range(6):
            q_plus = q0[:]
            q_plus[i] = q0[i] + delta_deg
            if low is not None:
                q_plus[i] = min(max(q_plus[i], low[i]), high[i])

            err = robot.MoveJ(q_plus, tool=0, user=0, vel=vel_deg)
            if err != 0:
                print(f"  [!] J{i + 1} 正向 MoveJ 失败，错误码 {err}")
                all_ok = False
                break
            wait_move_done(robot)
            print(f"  [✓] J{i + 1}: {q0[i]:.2f}° -> {q_plus[i]:.2f}°")

            err = robot.MoveJ(q0, tool=0, user=0, vel=vel_deg)
            if err != 0:
                print(f"  [!] J{i + 1} 回位 MoveJ 失败，错误码 {err}")
                all_ok = False
                break
            wait_move_done(robot)
            print(f"  [✓] J{i + 1}: 回位 {q0[i]:.2f}°")
    except KeyboardInterrupt:
        print("\n[!] 用户中断，停止运动")
        robot.StopMotion()
        return False

    if all_ok:
        print("每关节微动测试通过（6/6 关节）。")
    return all_ok


# ======================================================================
# 测试 4：placo IK 离线测试（不驱动电机）
# ======================================================================

def _rot_angle_error(R_a, R_b):
    """两旋转矩阵间的角度误差（rad）。"""
    import numpy as np
    cos_angle = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def test_ik(robot=None):
    print("\n" + "=" * 60)
    print("测试 4/4：placo IK 离线测试（FK / IK 往返 / 笛卡尔偏移，不驱动电机）")
    print("=" * 60)

    try:
        import numpy as np
        import placo
        from xrobotoolkit_teleop.hardware.fr3_config import FR3_MANIPULATOR_CONFIG
        from xrobotoolkit_teleop.hardware.single_arm_fr3_controller import (
            _resolve_placo_arm_q_slice,
        )
    except ImportError as exc:
        print(f"[!] 依赖缺失（{exc}）。请在 xrobot conda 环境中运行。")
        return False

    urdf_path = FR3_MANIPULATOR_CONFIG["urdf_path"]
    if not urdf_path or not os.path.isfile(urdf_path):
        print(f"[!] FR3 URDF 未配置（当前值: {urdf_path!r}）。")
        print("    请先在 xrobotoolkit_teleop/hardware/fr3_config.py 的")
        print("    FR3_MANIPULATOR_CONFIG['urdf_path'] 填入 FR3 URDF 实际路径，")
        print("    并同步核对 tcp_frame 与 joint_names。")
        return False

    link_name = FR3_MANIPULATOR_CONFIG["tcp_frame"]
    joint_names = FR3_MANIPULATOR_CONFIG["joint_names"]

    robot_model = placo.RobotWrapper(urdf_path)
    q_slice = _resolve_placo_arm_q_slice(robot_model, joint_names)
    solver = placo.KinematicsSolver(robot_model)
    solver.mask_fbase(True)
    solver.dt = 0.01
    frame_task = solver.add_frame_task(link_name, np.eye(4))
    frame_task.configure("frame", "soft", 1.0)
    print(f"  URDF: {urdf_path}")
    print(f"  q 切片: {q_slice}，末端 link: {link_name}")

    # 起始关节角：优先读真机当前角（弧度），离线则用零位
    q0 = np.zeros(6)
    if robot is not None:
        ret = robot.GetActualJointPosRadian(1)
        if isinstance(ret, tuple) and ret[0] == 0 and ret[1] is not None:
            q0 = np.array(ret[1], dtype=float)
            print(f"  使用真机当前关节角(rad): {[round(v, 3) for v in q0]}")
    robot_model.state.q[q_slice] = q0
    robot_model.update_kinematics()

    def fk_pose():
        robot_model.update_kinematics()
        return robot_model.get_T_world_frame(link_name)

    def solve_until_converge(T_target, max_iter=200, pos_tol=1e-4):
        frame_task.T_world_frame = T_target
        for _ in range(max_iter):
            solver.solve(True)
            T = fk_pose()
            pos_err = float(np.linalg.norm(T[:3, 3] - T_target[:3, 3]))
            if pos_err < pos_tol:
                return T, pos_err
        return T, pos_err

    # --- 4.1 FK：正运动学输出 ---
    T0 = fk_pose()
    print(f"\n-- IK-1 正运动学 FK --")
    print(f"  [✓] {link_name} 位置(m): {np.round(T0[:3, 3], 4).tolist()}")
    if robot is not None:
        ret = robot.GetActualTCPPose()
        if isinstance(ret, tuple) and ret[0] == 0:
            tcp_mm = np.array(ret[1][:3]) / 1000.0
            diff = float(np.linalg.norm(tcp_mm - T0[:3, 3]))
            print(f"  真机 TCP(m): {np.round(tcp_mm, 4).tolist()}，与 FK 偏差 {diff * 1000:.1f}mm")
            print("  （TODO: 若偏差为固定偏移，需在 URDF 中配置 TCP 或核对基座对齐）")

    # --- 4.2 IK 往返：扰动关节后解算回原位姿 ---
    print(f"\n-- IK-2 IK 往返（关节扰动 0.05rad 后解回原 FK 位姿） --")
    q_perturbed = q0 + 0.05
    robot_model.state.q[q_slice] = q_perturbed
    robot_model.update_kinematics()
    T, pos_err = solve_until_converge(T0)
    rot_err = _rot_angle_error(T[:3, :3], T0[:3, :3])
    ok2 = pos_err < 1e-3 and rot_err < 1e-2
    print(f"  {'[✓]' if ok2 else '[!]'} 位置误差 {pos_err * 1000:.3f}mm，姿态误差 {math.degrees(rot_err):.3f}°")

    # --- 4.3 笛卡尔偏移：世界系 x 方向 +1cm 目标 ---
    print(f"\n-- IK-3 笛卡尔偏移（目标 = 原 FK 位姿沿世界 x +1cm） --")
    robot_model.state.q[q_slice] = q0
    robot_model.update_kinematics()
    T_target = T0.copy()
    T_target[0, 3] += 0.01
    T, pos_err = solve_until_converge(T_target)
    rot_err = _rot_angle_error(T[:3, :3], T_target[:3, :3])
    ok3 = pos_err < 1e-3 and rot_err < 1e-2
    print(f"  {'[✓]' if ok3 else '[!]'} 位置误差 {pos_err * 1000:.3f}mm，姿态误差 {math.degrees(rot_err):.3f}°")
    q_sol = robot_model.state.q[q_slice].copy()
    print(f"  解算关节角(rad): {[round(v, 3) for v in q_sol]}")

    passed = ok2 and ok3
    print(f"\nIK 测试{'全部通过' if passed else '存在未通过项'}。")
    return passed


# ======================================================================
# 入口
# ======================================================================

def confirm_motion(test_name, yes=False):
    if yes:
        return True
    print(f"\n[安全确认] 即将执行「{test_name}」，电机会运动。"
          "请确认机械臂周围无障碍物、已急停待命。")
    answer = input("输入 yes 继续，其他任意键取消: ").strip().lower()
    return answer == "yes"


def main():
    parser = argparse.ArgumentParser(description="FR3 迁移验证测试（data/servo/joints/ik）")
    parser.add_argument("--test", choices=["data", "servo", "joints", "ik", "all"],
                        default="data", help="测试项（默认 data，只读安全）")
    parser.add_argument("--robot-ip", default=ROBOT_IP, help="控制箱 IP（默认 192.168.58.2）")
    parser.add_argument("--yes", action="store_true", help="跳过运动测试的安全确认")
    # servo 参数
    parser.add_argument("--servo-hz", type=float, default=100.0, help="ServoJ 频率（默认 100）")
    parser.add_argument("--servo-duration", type=float, default=2.0, help="ServoJ 持续秒数")
    parser.add_argument("--servo-amplitude", type=float, default=0.5, help="摆动幅值（度）")
    parser.add_argument("--servo-joint", type=int, default=6, choices=range(1, 7),
                        help="摆动的关节编号 1~6（默认 6，腕部最安全）")
    parser.add_argument("--hold", action="store_true", help="ServoJ 只保持当前位置，不摆动")
    parser.add_argument("--cmd-type", type=int, default=1, choices=[0, 1],
                        help="命令传输方式：0=XML-RPC，1=UDP 透传（默认 1）")
    # joints 参数
    parser.add_argument("--micro-delta", type=float, default=2.0, help="每关节微动幅值（度）")
    args = parser.parse_args()

    results = {}

    # IK 测试可不连真机（离线验证 URDF/placo），其余测试需要连接
    if args.test in ("ik",):
        try:
            robot = connect(args.robot_ip)
        except (SystemExit, Exception) as exc:
            print(f"[!] 连接真机失败（{exc}），IK 测试将以零位离线运行。")
            robot = None
        try:
            results["ik"] = test_ik(robot)
        finally:
            if robot is not None:
                robot.CloseRPC()
        sys.exit(0 if results["ik"] else 1)

    robot = connect(args.robot_ip)
    try:
        if args.test in ("data", "all"):
            results["data"] = test_data(robot)

        if args.test in ("ik", "all"):
            results["ik"] = test_ik(robot)

        if args.test in ("joints", "all"):
            if confirm_motion(f"每关节微动 ±{args.micro_delta}°", args.yes):
                results["joints"] = test_joints(robot, delta_deg=args.micro_delta)
            else:
                print("已取消 joints 测试。")

        if args.test in ("servo", "all"):
            desc = "保持当前位置" if args.hold else f"J{args.servo_joint} ±{args.servo_amplitude}°"
            if confirm_motion(f"ServoJ {args.servo_hz:.0f}Hz {desc}", args.yes):
                results["servo"] = test_servo(
                    robot,
                    hz=args.servo_hz,
                    duration=args.servo_duration,
                    amplitude_deg=args.servo_amplitude,
                    servo_joint=args.servo_joint,
                    hold=args.hold,
                    cmd_type=args.cmd_type,
                )
            else:
                print("已取消 servo 测试。")
    except KeyboardInterrupt:
        print("\n[!] 用户中断，停止运动")
        try:
            robot.StopMotion()
        except Exception:
            pass
    finally:
        robot.CloseRPC()

    print("\n" + "=" * 60)
    print("测试结果汇总：")
    for name, ok in results.items():
        print(f"  {name:<8} {'PASS' if ok else 'FAIL'}")
    if not results:
        print("  （无测试执行）")
    sys.exit(0 if results and all(results.values()) else 1)


if __name__ == "__main__":
    main()
