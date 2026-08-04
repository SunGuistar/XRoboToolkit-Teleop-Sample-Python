#!/usr/bin/env python3
"""
法奥 FAIRINO FR3 简单控制脚本（fairino-python-sdk）

使用前准备：
1. fairino-python-sdk 已解压到本目录下的 fairino-python-sdk/ 文件夹（脚本自动处理导入路径）。
2. 网线直连控制箱：把电脑网卡设为与控制箱同网段（默认控制箱 IP 为 192.168.58.2，
   如有不同请修改下方 ROBOT_IP）。
3. 控制箱上电，示教器/APP 上确认无报警。

单位说明：位置 mm，姿态 °（度），关节角 °。
"""

import os
import sys
import time

# 将 SDK 的 linux 目录加入搜索路径（from fairino import Robot 需要）
SDK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fairino-python-sdk", "linux")
if os.path.isdir(SDK_PATH):
    sys.path.insert(0, SDK_PATH)

from fairino import Robot  # noqa: E402

ROBOT_IP = "192.168.58.2"  # 控制箱 IP


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


def main():
    # 1. 连接机器人（XML-RPC，端口 20003）
    robot = Robot.RPC(ROBOT_IP)
    if not getattr(robot, "is_connect", getattr(robot, "is_conect", True)):
        print("[!] 连接机器人失败，请检查 IP 与网线连接")
        return
    print("连接成功，SDK 版本:", robot.GetSDKVersion())

    # 2. 检查并清除报警
    _, (main_code, sub_code) = robot.GetRobotErrorCode()
    if main_code != 0 or sub_code != 0:
        print(f"[!] 机器人存在报警，主错误码: {main_code}，子错误码: {sub_code}")
        robot.ResetAllError()
        time.sleep(1)

    # 3. 使能机器人
    robot.RobotEnable(1)

    # 4. 读取当前状态
    ret = robot.GetActualJointPosDegree()  # (error, [j1..j6])，单位 °
    if isinstance(ret, tuple):
        print("当前关节角:", [round(q, 2) for q in ret[1]])
    ret = robot.GetActualTCPPose()  # (error, [x,y,z,rx,ry,rz])，mm / °
    if isinstance(ret, tuple):
        print("当前 TCP 位姿:", [round(v, 2) for v in ret[1]])

    # 5. 关节空间运动：MoveJ 回零位（Home），速度 20 mm/s（或°/s）
    home_q = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ret = robot.MoveJ(joint_pos=home_q, tool=0, user=0, vel=20.0)
    if check(ret, "MoveJ 回零位"):
        print("[>] MoveJ 回零位")
        wait_move_done(robot)

    # 6. 笛卡尔空间直线运动：MoveL 在当前位姿基础上沿基坐标 Z 轴抬升 50 mm
    #    如需绝对坐标运动，可自行改为固定 desc_pos
    ret = robot.GetActualTCPPose()
    if isinstance(ret, tuple):
        cur_pose = ret[1]
        target_pose = [cur_pose[0], cur_pose[1], cur_pose[2] + 50.0,
                       cur_pose[3], cur_pose[4], cur_pose[5]]
        ret = robot.MoveL(desc_pos=target_pose, tool=0, user=0, vel=20.0)
        if check(ret, "MoveL 直线运动"):
            print("[>] MoveL 抬升 50mm")
            wait_move_done(robot)
    else:
        print("[!] 获取当前 TCP 位姿失败，跳过 MoveL")

    print("[✓] 演示完成")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] 用户中断，停止运动")
        try:
            Robot.RPC(ROBOT_IP).StopMotion()
        except Exception:
            pass
