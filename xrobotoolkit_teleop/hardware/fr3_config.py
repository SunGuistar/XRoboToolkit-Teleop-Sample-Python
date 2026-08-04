"""FAIRINO FR3 机器人配置。

迁移自 UR5e 版本（single_arm_ur_controller.py）。
所有 FR3 特有的模型/网络参数集中在此文件，URDF 未提供前相关字段为占位值。

TODO(部署前必须确认):
  1. urdf_path   —— 填入 FR3 官方 URDF 路径（FAIRINO 官方资料/售后获取）。
  2. tcp_frame   —— URDF 中末端法兰/TCP 的 link 名称（placo IK 目标帧）。
  3. joint_names —— 与 URDF 中 J1~J6 关节名严格一致（placo q 切片靠它解析）。
  4. joint_limits—— FR3 实际关节软限位（弧度），可用
     robot.GetJointSoftLimitDeg() 从控制箱读取后换算。
"""

from __future__ import annotations

import os

import numpy as np

# FAIRINO 控制箱默认 IP（SDK 默认值），按实际网络修改。
DEFAULT_ROBOT_IP = "192.168.58.2"

# FR3 URDF：来自 FAIRINO 官方 frcobot_ros 包（gitee.com/fair-innovation/frcobot_ros）
# 的 frcobot_description/urdf/fr3v6.urdf。已做两处适配：
#   1. 移除 visual/collision 网格（原仓库 mesh 文件不完整，placo 仅需运动学）；
#   2. 添加与 tool_Link 重合的 "flange" 别名帧（软约束模块固定读取 "flange"）。
# 实机验证：零位及 J1+15° 两姿态下 tool_Link FK 与实机法兰偏差 0.0mm，
# 确认本机为 FR3 V6 构型。
_FR3_URDF_PATH = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "..", "..", "assets", "fairino_fr3", "fr3.urdf")
)

# VR 手柄 Grip 触发阈值（与 UR 版 CONTROLLER_DEADZONE 一致）。
CONTROLLER_DEADZONE = 0.1

# FR3 遥操作默认 Home（度）。占位为零位，
# 实际 Home 请用 `python FR3/teleop_single_fr3_hand.py --save-home` 生成。
FR3_INITIAL_JOINT_DEG = np.zeros(6, dtype=float)

# FR3 关节软限位（度），来自实机 GetJointSoftLimitDeg() 实测：
# 布局为逐关节交替对 [low1, high1, low2, high2, ..., low6, high6]。
_FR3_JOINT_LIMITS_DEG = [
    (-175.0, 175.0),  # J1
    (-265.0, 85.0),   # J2
    (-150.0, 150.0),  # J3
    (-265.0, 85.0),   # J4
    (-175.0, 175.0),  # J5
    (-175.0, 175.0),  # J6
]

FR3_MANIPULATOR_CONFIG = {
    # FR3 官方 URDF（fr3v6，实机 FK 验证通过）。
    "urdf_path": _FR3_URDF_PATH,
    # 官方定义的法兰帧：tool_Link（wrist 输出端 j6_Link 前方 100mm，
    # 与实机 GetActualToolFlangePose 完全一致）。若加装夹爪，
    # 在 URDF 中新增 tool 帧后改此处。
    "tcp_frame": "tool_Link",
    # 与 URDF 关节名严格一致（已按 fairino3_v6.urdf 核实）。
    "joint_names": [
        "j1",
        "j2",
        "j3",
        "j4",
        "j5",
        "j6",
    ],
    # [(lower, upper)] x 6，单位弧度（实机软限位换算）。
    "joint_limits": [
        (float(np.deg2rad(low)), float(np.deg2rad(high)))
        for low, high in _FR3_JOINT_LIMITS_DEG
    ],
}


def fr3_joint_limits_arrays() -> tuple[np.ndarray, np.ndarray]:
    """返回 (lower, upper) 两个 (6,) 弧度数组，供 BCConfig 部署安全裁剪使用。"""
    lower = np.array([lim[0] for lim in FR3_MANIPULATOR_CONFIG["joint_limits"]], dtype=np.float64)
    upper = np.array([lim[1] for lim in FR3_MANIPULATOR_CONFIG["joint_limits"]], dtype=np.float64)
    return lower, upper
