"""单臂 FAIRINO FR3 遥操作控制器（UR5e 版 single_arm_ur_controller.py 的型号迁移）。

控制链路保持不变：
    VR手柄 -> XRoboToolkit -> SingleArmFR3Controller -> placo IK
             -> FR3 实时关节伺服 (SDK ServoMoveStart + ServoJ + ServoMoveEnd) -> FR3

与 UR 版的差异仅在底层：
  - RTDE servoJ         -> fairino SDK ServoJ（关节单位 °，cmdType=1 UDP 透传）
  - rtde_r.getActualQ() -> robot_state_pkg.jt_cur_pos（CNDE 状态流缓存，与 RTDE
                           同为"推流+本地缓存"结构，纯内存读、无 RPC、无多线程
                           竞争；CNDE 不可用时回退 RPC GetActualJointPosRadian）
  - rtde_c.moveJ()      -> robot.MoveJ()（仅用于 Home 复位，不用于遥操作实时控制）

底层 SDK 接口已在 FAIRINO 官方 Python SDK (fairino/Robot.py) 中核实存在：
  Robot.RPC / ServoMoveStart / ServoJ / ServoMoveEnd / MoveJ /
  GetActualJointPosRadian / GetMotionQueueLength / RobotEnable /
  GetRobotErrorCode / ResetAllError / StopMotion / CloseRPC
"""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser

import meshcat.transformations as tf
import numpy as np
import placo
from placo_utils.visualization import (
    frame_viz,
    robot_frame_viz,
    robot_viz,
)

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.fr3_config import (
    CONTROLLER_DEADZONE,
    DEFAULT_ROBOT_IP,
    FR3_INITIAL_JOINT_DEG,
    FR3_MANIPULATOR_CONFIG,
)
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    apply_delta_pose,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.bc.config import BCConfig
from xrobotoolkit_teleop.bc.gripper_norm import trigger_to_gripper_cmd
from xrobotoolkit_teleop.bc.policy import BCPolicy, apply_policy_action
from xrobotoolkit_teleop.hardware.soft_constraint import (
    DEFAULT_DZ_DAMP,
    DEFAULT_KR,
    DEFAULT_KZ,
    SoftConstraintTeleopHelper,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

DEFAULT_SCALE_FACTOR = 0.8

# ============================ FR3 实时控制参数 ============================
# ServoJ 指令下发周期（s）。UR 版 servo_time=0.017(~60Hz)；
# FR3 目标 >=50Hz，默认 100Hz。SDK 文档建议 cmdT 与实际下发周期一致。
FR3_SERVO_PERIOD = 0.01
# ServoJ/MoveJ 使用的传输方式：0=XML-RPC，1=UDP 透传（低延迟，推荐）。
# TODO: 若控制箱固件不支持 UDP 透传（端口 20007），改为 0。
FR3_SERVO_CMD_TYPE = 1
# MoveJ 回 Home 速度（°/s）。
FR3_MOVEJ_VEL_DEG = 30.0
# FR3 无外部轴时 ServoJ 的 axisPos 占位（4 轴，mm）。
FR3_EXAXIS_POS = [0.0, 0.0, 0.0, 0.0]

FR3_RIGHT_FLANGE_CONTROLLER_MANIPULATOR_CONFIG = {
    "arm": {
        "link_name": FR3_MANIPULATOR_CONFIG["tcp_frame"],
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_trigger": "right_trigger",
    },
}

FR3_LEFT_FLANGE_CONTROLLER_MANIPULATOR_CONFIG = {
    "arm": {
        "link_name": FR3_MANIPULATOR_CONFIG["tcp_frame"],
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_trigger": "left_trigger",
    },
}


def _import_fairino_robot():
    """导入 fairino SDK 的 Robot 模块。

    优先使用已 pip 安装的 fairino；否则回退到项目内
    FR3/fairino-python-sdk/linux（与 FR3/simple.py 相同的 SDK 副本）。
    """
    try:
        from fairino import Robot

        return Robot
    except ImportError:
        pass

    candidates = [
        # 项目内 SDK 副本：<repo>/FR3/fairino-python-sdk/linux
        os.path.normpath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                         "FR3", "fairino-python-sdk", "linux")
        ),
    ]
    for path in candidates:
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    try:
        from fairino import Robot

        return Robot
    except ImportError as exc:
        raise RuntimeError(
            "无法导入 fairino SDK（from fairino import Robot）。\n"
            "请执行以下任一方式安装：\n"
            "  1. cd FR3/fairino-python-sdk/linux/fairino && python setup.py install\n"
            "  2. 或将 FR3/fairino-python-sdk/linux 加入 PYTHONPATH。\n"
            "注意：SDK 的 libfairino/*.so 需与当前 Python 版本匹配，"
            "必要时按官方 README 重新编译。"
        ) from exc


def _resolve_placo_arm_q_slice(placo_robot: placo.RobotWrapper, joint_names: list[str]) -> slice:
    """按 joint_names 在 placo 状态向量 q 中解析 6 个臂关节的切片。

    UR 版硬编码 slice(7, 13)（URDF 含 7 维浮动基座）。FR3 URDF 结构未知，
    这里通过 pinocchio 模型的 idx_q 自动解析（已在 UR5e URDF 上验证得到 [7..12]）；
    解析失败时回退 UR 式切片并告警。
    """
    try:
        model = placo_robot.model
        idxs = []
        for name in joint_names:
            if not model.existJointName(name):
                raise ValueError(f"URDF 中不存在关节 {name!r}")
            idxs.append(int(model.joints[model.getJointId(name)].idx_q))
        if idxs != list(range(idxs[0], idxs[0] + 6)):
            raise ValueError(f"joint_names 在 URDF q 向量中不连续：{idxs}")
        return slice(idxs[0], idxs[0] + 6)
    except Exception as exc:
        print(
            f"WARNING: 无法从 URDF 解析 FR3 关节 q 切片（{exc}），"
            "回退使用 UR 版 slice(7, 13)。请检查 fr3_config.py 的 joint_names。"
        )
        return slice(7, 13)


class FR3Controller:
    """FAIRINO FR3 底层封装（对应 UR 版 URController）。

    单位约定：对外与 UR 版一致，全部使用弧度；SDK 交互时换算为 SDK 要求的度。
    """

    def __init__(
        self,
        robot_ip: str = DEFAULT_ROBOT_IP,
        initial_joint_positions: np.ndarray = np.zeros(6),
        servo_period: float = FR3_SERVO_PERIOD,
        servo_cmd_type: int = FR3_SERVO_CMD_TYPE,
        movej_vel_deg: float = FR3_MOVEJ_VEL_DEG,
    ):
        self.robot_ip = robot_ip
        self.initial_joint_positions = np.asarray(initial_joint_positions, dtype=np.float64).copy()
        self.servo_period = servo_period
        self.servo_cmd_type = servo_cmd_type
        self.movej_vel_deg = movej_vel_deg

        Robot = _import_fairino_robot()
        self.robot = Robot.RPC(robot_ip)
        # RPC 构造后 CNDE 状态流需要一点时间建立。
        time.sleep(0.5)

        if not getattr(self.robot, "is_connect", True):
            raise RuntimeError(
                f"无法连接 FR3 控制箱（{robot_ip}）。\n"
                "检查清单：\n"
                f"  1. Ping 控制箱：ping {robot_ip}\n"
                "  2. PC 网卡与控制箱同网段（默认 192.168.58.x）\n"
                "  3. 控制箱已上电、示教器/APP 无急停与报警"
            )
        print(f"Connected to FR3 robot at {robot_ip}")

        # 检查并尝试清除报警，再上使能（自动模式）。
        try:
            _, (main_code, sub_code) = self.robot.GetRobotErrorCode()
            if main_code != 0 or sub_code != 0:
                print(f"FR3 存在报警（主错误码 {main_code}，子错误码 {sub_code}），尝试 ResetAllError...")
                self.robot.ResetAllError()
                time.sleep(1.0)
        except Exception as exc:
            print(f"WARNING: 读取/清除 FR3 报警失败：{exc}")

        err = self.robot.Mode(0)  # 自动模式（SDK 外部控制所需）
        if err != 0:
            print(f"WARNING: FR3 Mode(0) 返回错误码 {err}（若示教器已在自动模式可忽略）。")
        err = self.robot.RobotEnable(1)
        if err != 0:
            raise RuntimeError(
                f"FR3 RobotEnable(1) 失败，错误码 {err}。"
                "请确认示教器/APP 无急停、无报警后再重试。"
            )

        self._servo_active = False
        self._last_q: np.ndarray | None = None
        # RPC 串行化锁：xmlrpc.client.ServerProxy 非线程安全，多线程共用同一 HTTP
        # 连接并发请求会抛 http.client.CannotSendRequest。高频状态读已改走 CNDE
        # 缓存，此锁仅保护残留的低频 RPC 调用（如回退读、MoveJ）。
        self._rpc_lock = threading.Lock()
        self._cnde_verified = False
        self._cnde_ok = self._wait_cnde_ready(timeout_s=2.0)

    def _wait_cnde_ready(self, timeout_s: float = 2.0) -> bool:
        """等待 CNDE 状态流开始刷新，超时则回退 RPC 读。

        探针用默认订阅内的 robot_time.millisecond（每帧随控制器时钟刷新）。
        不能用 frame_cnt：控制箱固件不支持订阅该状态（NOT_FOUND，会导致
        整个 CNDE 配置被拒绝，错误码 -18）。
        """
        pkg = getattr(self.robot, "robot_state_pkg", None)
        if pkg is None:
            return False
        try:
            first_ms = int(pkg.millisecond)
        except Exception:
            return False
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            try:
                if int(pkg.millisecond) != first_ms:
                    print("CNDE 状态流已就绪，关节角读取使用本地缓存（无 RPC）。")
                    return True
            except Exception:
                break
            time.sleep(0.01)
        print(
            "WARNING: CNDE 状态流未收到数据，关节角读取回退 RPC 轮询"
            "（慢且存在多线程竞争风险）。"
        )
        return False

    # ---------------- 状态读取 ----------------

    def get_current_joint_positions(self) -> np.ndarray:
        """当前关节角，numpy (6,)，弧度（与 UR 版 getActualQ 一致）。

        优先读 CNDE 缓存（robot_state_pkg.jt_cur_pos，控制箱 8ms 推流，SDK 接收
        线程持续刷新）：纯内存读，等价 UR 版 rtde_r.getActualQ()，任意线程可调。
        注意 jt_cur_pos 单位为度（RobotStatePkg 结构体注释），需换算为弧度。
        """
        if self._cnde_ok:
            pkg = getattr(self.robot, "robot_state_pkg", None)
            if pkg is not None:
                q_rad = np.deg2rad(np.array(pkg.jt_cur_pos, dtype=np.float64))
                self._verify_cnde_units_once(q_rad)
                self._last_q = q_rad
                return q_rad.copy()

        with self._rpc_lock:
            try:
                err, q = self.robot.GetActualJointPosRadian(1)
            except Exception as exc:
                if self._last_q is not None:
                    print(f"WARNING: GetActualJointPosRadian 异常（{exc}），使用上次缓存值。")
                    return self._last_q.copy()
                raise
        if err != 0 or q is None:
            if self._last_q is not None:
                print(f"WARNING: GetActualJointPosRadian 失败（错误码 {err}），使用上次缓存值。")
                return self._last_q.copy()
            raise RuntimeError(f"GetActualJointPosRadian 失败，错误码 {err}，且无缓存可用。")
        self._last_q = np.array(q, dtype=np.float64)
        return self._last_q.copy()

    def _verify_cnde_units_once(self, q_cnde_rad: np.ndarray) -> None:
        """首次 CNDE 读取时与 RPC 弧度值交叉比对，确认"度"单位假设成立。"""
        if self._cnde_verified:
            return
        self._cnde_verified = True
        try:
            with self._rpc_lock:
                err, q_rpc = self.robot.GetActualJointPosRadian(1)
            if err == 0 and q_rpc is not None:
                diff = float(np.max(np.abs(np.asarray(q_rpc, dtype=np.float64) - q_cnde_rad)))
                if diff < 0.01:
                    print(f"CNDE 关节角缓存校验通过：与 RPC 一致（最大差 {diff:.6f} rad）。")
                else:
                    print(
                        f"WARNING: CNDE 关节角缓存与 RPC 差异过大（最大 {diff:.3f} rad），"
                        "请检查 jt_cur_pos 单位换算！"
                        f"\n  CNDE(deg): {np.rad2deg(q_cnde_rad)}"
                        f"\n  RPC (deg): {np.rad2deg(np.asarray(q_rpc, dtype=np.float64))}"
                    )
        except Exception as exc:
            print(f"WARNING: CNDE 单位交叉校验失败（{exc}），沿用度单位假设。")

    # ---------------- 实时伺服（遥操作用，等价 RTDE servoJ） ----------------

    def servo_start(self) -> bool:
        """进入伺服模式（配合 ServoJ 使用，对应 UR servoJ 的隐式进入）。"""
        if self._servo_active:
            return True
        with self._rpc_lock:
            err = self.robot.ServoMoveStart(cmdType=self.servo_cmd_type)
        if err != 0:
            print(f"WARNING: ServoMoveStart 失败，错误码 {err}。")
            return False
        self._servo_active = True
        return True

    def servo_stop(self) -> None:
        """退出伺服模式（对应 UR servoStop）。"""
        if not self._servo_active:
            return
        try:
            with self._rpc_lock:
                self.robot.ServoMoveEnd(cmdType=self.servo_cmd_type)
        except Exception:
            pass
        self._servo_active = False

    def fr3_servo(self, joint_positions_rad: np.ndarray) -> bool:
        """FR3 实时关节伺服：单周期下发目标关节角（等价 UR RTDE servoJ）。

        TODO: 需要根据实际 FR3 SDK/固件行为确认与微调：
          - ServoJ 参数 acc/vel/filterT/gain 官方标注"暂不开放"，当前全部取 0；
          - cmdT 应等于实际下发周期（self.servo_period）；
          - cmdType=1(UDP) 若在你的固件上不稳定，改 FR3_SERVO_CMD_TYPE=0。
        """
        q_deg = np.rad2deg(np.asarray(joint_positions_rad, dtype=np.float64)).tolist()
        if self.servo_cmd_type == 0:
            # cmdType=0 走 XML-RPC：必须串行化；100Hz 下与其他 RPC 调用互斥。
            with self._rpc_lock:
                err = self.robot.ServoJ(
                    q_deg,
                    FR3_EXAXIS_POS,
                    acc=0.0,
                    vel=0.0,
                    cmdT=self.servo_period,
                    filterT=0.0,
                    gain=0.0,
                    id=0,
                    cmdType=self.servo_cmd_type,
                )
        else:
            # cmdType=1 走 UDP 透传（内部安全检查只读内存），无 RPC，不加锁。
            err = self.robot.ServoJ(
                q_deg,
                FR3_EXAXIS_POS,
                acc=0.0,
                vel=0.0,
                cmdT=self.servo_period,
                filterT=0.0,
                gain=0.0,
                id=0,
                cmdType=self.servo_cmd_type,
            )
        return err == 0

    def servo_joints(self, joint_positions_rad: np.ndarray) -> bool:
        """与 UR 版同名接口：下发一次目标关节角（弧度）。"""
        return self.fr3_servo(joint_positions_rad)

    # ---------------- 非实时运动（Home 复位用） ----------------

    def move_j(self, joint_positions_rad: np.ndarray, vel_deg: float | None = None) -> bool:
        """MoveJ 到目标关节角（仅用于 --reset / 录制结束回 Home）。"""
        if vel_deg is None:
            vel_deg = self.movej_vel_deg
        q_deg = np.rad2deg(np.asarray(joint_positions_rad, dtype=np.float64)).tolist()
        with self._rpc_lock:
            err = self.robot.MoveJ(q_deg, tool=0, user=0, vel=vel_deg)
        if err != 0:
            raise RuntimeError(
                f"FR3 MoveJ 失败，错误码 {err}。\n"
                "  - 检查示教器：自动模式、已使能、无报警/急停\n"
                "  - 目标关节角可能超限，可用 GetJointSoftLimitDeg() 核对"
            )
        return self._wait_motion_done(timeout_s=60.0)

    def _wait_motion_done(self, timeout_s: float = 60.0) -> bool:
        """轮询运动队列清空（与 FR3/simple.py 相同策略）。"""
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            try:
                # 只锁单次 RPC 调用，不锁整个轮询循环，避免长时间阻塞其他 RPC。
                with self._rpc_lock:
                    err, queue_len = self.robot.GetMotionQueueLength()
                if err == 0 and queue_len == 0:
                    return True
            except Exception:
                pass
            time.sleep(0.05)
        print("WARNING: 等待 FR3 运动完成超时。")
        return False

    def reset(self) -> None:
        """MoveJ 回 Home（对应 UR 版 reset）。"""
        print(f"Moving FR3 to initial joint positions (rad): {self.initial_joint_positions}")
        self.move_j(self.initial_joint_positions)
        print("Reached initial position.")

    def close(self) -> None:
        try:
            self.servo_stop()
        except Exception:
            pass
        try:
            with self._rpc_lock:
                self.robot.StopMotion()
        except Exception:
            pass
        try:
            with self._rpc_lock:
                self.robot.CloseRPC()
        except Exception:
            pass
        print("FR3 controller closed.")


class SingleArmFR3Controller:
    """Teleoperation controller for one FAIRINO FR3 arm driven by a PICO VR controller.

    接口与 SingleArmURController 保持一致（BC 录制器 / teleop 脚本按鸭子类型复用）。
    """

    def __init__(
        self,
        xr_client: XrClient,
        manipulator_config: dict = FR3_RIGHT_FLANGE_CONTROLLER_MANIPULATOR_CONFIG,
        robot_ip: str = DEFAULT_ROBOT_IP,
        initial_joint_deg: np.ndarray = FR3_INITIAL_JOINT_DEG,
        visualize_placo: bool = False,
        policy_mode: bool = False,
        bc_config: BCConfig | None = None,
        enable_soft_constraint: bool = True,
        R_headset_world: np.ndarray | None = None,
        robot_urdf_path: str | None = None,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        servo_period: float = FR3_SERVO_PERIOD,
        servo_cmd_type: int = FR3_SERVO_CMD_TYPE,
        use_dh_gripper: bool = False,
        dh_gripper_port: str = "/dev/ttyUSB0",
        dh_gripper_force: int = 50,
        dh_gripper_speed: int = 50,
        dh_gripper_slave_id: int = 1,
        dh_gripper_baudrate: int = 115200,
        dh_gripper_standard_position: bool = False,
        soft_constraint_kz: float = DEFAULT_KZ,
        soft_constraint_kr: float = DEFAULT_KR,
        soft_constraint_dz_damp: float = DEFAULT_DZ_DAMP,
    ):
        self.xr_client = xr_client
        self.robot_urdf_path = robot_urdf_path or FR3_MANIPULATOR_CONFIG["urdf_path"]
        if not self.robot_urdf_path or not os.path.isfile(self.robot_urdf_path):
            raise RuntimeError(
                f"FR3 URDF 路径无效：{self.robot_urdf_path!r}\n"
                "请在 xrobotoolkit_teleop/hardware/fr3_config.py 的 "
                "FR3_MANIPULATOR_CONFIG['urdf_path'] 填入 FR3 URDF 实际路径。"
            )
        self.R_headset_world = (
            R_HEADSET_TO_WORLD if R_headset_world is None else R_headset_world
        )
        self.scale_factor = scale_factor
        self.visualize_placo = visualize_placo
        self.manipulator_config = manipulator_config
        self.arm_name = next(iter(manipulator_config.keys()))
        self.use_dh_gripper = use_dh_gripper
        self.dh_gripper = None

        # placo q 切片按 FR3 URDF 关节名解析（UR 版为硬编码 slice(7, 13)）。
        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        self.arm_q_slice = _resolve_placo_arm_q_slice(
            self.placo_robot, FR3_MANIPULATOR_CONFIG["joint_names"]
        )

        self.robot_controller = FR3Controller(
            robot_ip=robot_ip,
            initial_joint_positions=np.deg2rad(initial_joint_deg),
            servo_period=servo_period,
            servo_cmd_type=servo_cmd_type,
        )

        self._dh_gripper_kwargs = None
        if use_dh_gripper:
            self._dh_gripper_kwargs = dict(
                port=dh_gripper_port,
                baudrate=dh_gripper_baudrate,
                slave_id=dh_gripper_slave_id,
                force=dh_gripper_force,
                speed=dh_gripper_speed,
                standard_position_mapping=dh_gripper_standard_position,
            )

        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = servo_period
        self.solver.mask_fbase(True)
        self.solver.add_kinetic_energy_regularization_task(1e-6)

        self.effector_task = {}
        self.init_ee_xyz = {}
        self.init_ee_quat = {}
        self.init_controller_xyz = {}
        self.init_controller_quat = {}
        for name, config in self.manipulator_config.items():
            initial_pose = np.eye(4)
            self.effector_task[name] = self.solver.add_frame_task(config["link_name"], initial_pose)
            self.effector_task[name].configure(f"{name}_frame", "soft", 1.0)
            manipulability = self.solver.add_manipulability_task(config["link_name"], "both", 1.0)
            manipulability.configure(f"{name}_manipulability", "soft", 5e-2)
            self.init_ee_xyz[name] = None
            self.init_ee_quat[name] = None
            self.init_controller_xyz[name] = None
            self.init_controller_quat[name] = None

        q_init = self.robot_controller.get_current_joint_positions()
        self.gripper_pos = 0
        self.placo_robot.state.q[self.arm_q_slice] = q_init
        self.placo_robot.update_kinematics()
        self.target_q = q_init.copy()
        self._control_lock = threading.Lock()
        self.control_active = False
        self._servo_engaged = False
        self._pending_gripper_trigger: float = 0.0

        # BC deployment hooks（与 UR 版一致：录制用 sample_bc_frame；policy 可选）
        self.policy_mode = policy_mode
        self.bc_config = bc_config or BCConfig()
        self.policy: BCPolicy | None = None
        self._policy_image: np.ndarray | None = None
        self._policy_image_lock = threading.Lock()
        self._reset_in_progress = False
        self.enable_soft_constraint = enable_soft_constraint
        self._soft_constraint = SoftConstraintTeleopHelper(
            kz=soft_constraint_kz,
            kr=soft_constraint_kr,
            dz_damp=soft_constraint_dz_damp,
        )
        for name, config in self.manipulator_config.items():
            self.effector_task[name].T_world_frame = self.placo_robot.get_T_world_frame(config["link_name"])

        if self.visualize_placo:
            self.placo_robot.update_kinematics()
            self.placo_vis = robot_viz(self.placo_robot)
            time.sleep(0.5)
            meshcat_url = self.placo_vis.viewer.url()
            print(f"Automatically opening meshcat at: {meshcat_url}")
            webbrowser.open(meshcat_url)
            self.placo_vis.display(self.placo_robot.state.q)
            for name, config in self.manipulator_config.items():
                robot_frame_viz(self.placo_robot, config["link_name"])
                frame_viz(
                    f"vis_target_{name}",
                    self.effector_task[name].T_world_frame,
                )

    # ---------------- XR 位姿处理（与 UR 版完全一致，未改动 XR 逻辑） ----------------

    def _process_xr_pose(self, xr_pose, arm_name: str):
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = np.array(
            [
                xr_pose[6],
                xr_pose[3],
                xr_pose[4],
                xr_pose[5],
            ]
        )

        controller_xyz = self.R_headset_world @ controller_xyz

        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
        )

        if self.init_controller_xyz[arm_name] is None:
            self.init_controller_xyz[arm_name] = controller_xyz.copy()
            self.init_controller_quat[arm_name] = controller_quat.copy()
            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])
        else:
            delta_xyz = (controller_xyz - self.init_controller_xyz[arm_name]) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(self.init_controller_quat[arm_name], controller_quat)
        return delta_xyz, delta_rot

    def _resync_cumulative_teleop_after_soft_off(self) -> None:
        for name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            if xr_grip_val > (1.0 - CONTROLLER_DEADZONE):
                T_world_ee = self.placo_robot.get_T_world_frame(config["link_name"])
                self.init_ee_xyz[name] = T_world_ee[:3, 3].copy()
                self.init_ee_quat[name] = tf.quaternion_from_matrix(T_world_ee)
                self.init_controller_xyz[name] = None
                self.init_controller_quat[name] = None

    def _update_soft_constraint_buttons(self) -> None:
        # A：开启约束（锁 z/rx/ry，允许 x/y/rz）；B：关闭约束。逻辑未改动。
        if not self.enable_soft_constraint:
            return
        self._soft_constraint.update_buttons(
            a_pressed=self.xr_client.get_button_state_by_name("A"),
            b_pressed=self.xr_client.get_button_state_by_name("B"),
            placo_robot=self.placo_robot,
            resync_active_grip=self._resync_cumulative_teleop_after_soft_off,
        )

    def _compute_soft_constraint_target(
        self,
        name: str,
        config: dict,
        xr_pose,
        init_xyz: np.ndarray,
        init_quat: np.ndarray,
    ) -> np.ndarray:
        delta_xyz, delta_rot = self._process_xr_pose(xr_pose, name)
        target_xyz, target_quat = apply_delta_pose(init_xyz, init_quat, delta_xyz, delta_rot)
        t_user = tf.quaternion_matrix(target_quat)
        t_user[:3, 3] = target_xyz
        return self._soft_constraint.project_target(t_user)

    # ---------------- BC / policy 接口（与 UR 版一致） ----------------

    def set_policy(self, policy: BCPolicy | None) -> None:
        """Attach a BC policy for deployment (policy_mode must be True)."""
        self.policy = policy

    def set_policy_image(self, image_bgr: np.ndarray) -> None:
        """Update the latest main-camera frame for policy inference."""
        with self._policy_image_lock:
            self._policy_image = image_bgr.copy()

    def get_gripper_cmd_normalized(self) -> float:
        """VR trigger as gripper_cmd. 0=open, 1=close."""
        config = self.manipulator_config[self.arm_name]
        trigger_val = self.xr_client.get_key_value_by_name(config["gripper_trigger"])
        return trigger_to_gripper_cmd(trigger_val)

    def get_gripper_state_normalized(self) -> float:
        """Estimated gripper opening. 0=open, 1=close.

        FR3 默认无物理夹爪：与 UR 版无夹爪分支相同，用扳机值作为代理状态。
        """
        if self.dh_gripper is not None:
            return self.dh_gripper.get_normalized_state()
        return self.get_gripper_cmd_normalized()

    def get_observation_state(self) -> np.ndarray:
        """BC observation.state = [qpos(6), gripper_state(1)]. 0=open, 1=close."""
        qpos = self.robot_controller.get_current_joint_positions()
        gripper_state = self.get_gripper_state_normalized()
        return np.concatenate([qpos, [gripper_state]]).astype(np.float64)

    def sample_bc_frame(self) -> dict:
        """Sample robot fields for BC recording (no image)."""
        with self._control_lock:
            target_q = self.target_q.copy()
            control_active = self.control_active
        return {
            "qpos": self.robot_controller.get_current_joint_positions().astype(np.float64),
            "target_q": target_q.astype(np.float64),
            "gripper_state": self.get_gripper_state_normalized(),
            "gripper_cmd": self.get_gripper_cmd_normalized(),
            "control_active": control_active,
        }

    def _run_policy_step(self, current_q: np.ndarray) -> bool:
        """Policy → target_q when policy_mode and human grip is not active."""
        if self.policy is None:
            return False
        with self._policy_image_lock:
            image = None if self._policy_image is None else self._policy_image.copy()
        if image is None:
            return False

        obs = {
            "observation.images.main": image,
            "observation.state": self.get_observation_state(),
        }
        action = self.policy.predict(obs)
        target_q, gripper_cmd = apply_policy_action(
            action,
            current_q,
            max_delta_q=self.bc_config.max_delta_q,
            joint_limits_lower=self.bc_config.joint_limits_lower,
            joint_limits_upper=self.bc_config.joint_limits_upper,
        )
        with self._control_lock:
            self.target_q = target_q
        self._pending_gripper_trigger = gripper_cmd
        return True

    # ---------------- IK ----------------

    def calc_target_joint_position(self):
        """Calculate target joint positions from PICO controller pose and grip input."""
        current_q = self.robot_controller.get_current_joint_positions()
        self.placo_robot.state.q[self.arm_q_slice] = current_q
        self.placo_robot.update_kinematics()
        self._update_soft_constraint_buttons()

        human_control_active = False
        for name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)
            human_control_active = human_control_active or active
            if self.use_dh_gripper and (active or not self.policy_mode):
                # In policy_mode, gripper_cmd comes from policy unless human has grip active.
                self._pending_gripper_trigger = self.xr_client.get_key_value_by_name(config["gripper_trigger"])

            if active:
                if self.init_ee_xyz[name] is None:
                    T_world_ee_current = self.placo_robot.get_T_world_frame(config["link_name"])
                    self.init_ee_xyz[name] = T_world_ee_current[:3, 3].copy()
                    self.init_ee_quat[name] = tf.quaternion_from_matrix(T_world_ee_current)
                    print(
                        f"{name} activated. Current EE xyz: {self.init_ee_xyz[name]}, "
                        f"quat: {self.init_ee_quat[name]}."
                    )
                    if self.enable_soft_constraint and self._soft_constraint.active:
                        self._soft_constraint.set_grip_start(
                            self.placo_robot.get_T_world_frame(config["link_name"])
                        )

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])

                if self.enable_soft_constraint and self._soft_constraint.active:
                    target_transform = self._compute_soft_constraint_target(
                        name,
                        config,
                        xr_pose,
                        self.init_ee_xyz[name],
                        self.init_ee_quat[name],
                    )
                else:
                    delta_xyz, delta_rot_angle_axis = self._process_xr_pose(xr_pose, name)
                    target_xyz, target_quat = apply_delta_pose(
                        self.init_ee_xyz[name],
                        self.init_ee_quat[name],
                        delta_xyz,
                        delta_rot_angle_axis,
                    )
                    target_transform = tf.quaternion_matrix(target_quat)
                    target_transform[:3, 3] = target_xyz

                self.effector_task[name].T_world_frame = target_transform

            else:
                if self.init_ee_xyz[name] is not None:
                    print(f"{name} deactivated.")
                    self.init_ee_xyz[name] = None
                    self.init_ee_quat[name] = None
                    self.init_controller_xyz[name] = None
                    self.init_controller_quat[name] = None
                self._soft_constraint.reset_motion_state()
                T_world_ee_current = self.placo_robot.get_T_world_frame(config["link_name"])
                self.effector_task[name].T_world_frame = T_world_ee_current

        with self._control_lock:
            self.control_active = human_control_active

        # Human grip always wins over policy (safety / takeover).
        if not human_control_active and self.policy_mode:
            if self._run_policy_step(current_q):
                return

        if not human_control_active:
            with self._control_lock:
                self.target_q = current_q.copy()
            return

        try:
            self.solver.solve(True)
            with self._control_lock:
                self.target_q = self.placo_robot.state.q[self.arm_q_slice].copy()

            if self.visualize_placo and hasattr(self, "placo_vis"):
                self.placo_vis.display(self.placo_robot.state.q)
                for name, config in self.manipulator_config.items():
                    robot_frame_viz(self.placo_robot, config["link_name"])
                    frame_viz(
                        f"vis_target_{name}",
                        self.effector_task[name].T_world_frame,
                    )

        except RuntimeError as e:
            print(f"IK solver failed: {e}. Returning last known good joint positions.")
        except Exception as e:
            print(f"An unexpected error occurred in IK: {e}. Returning last known good joint positions.")

    # ---------------- 夹爪 / 复位 / 生命周期 ----------------

    def init_dh_gripper(self) -> bool:
        """Connect and initialize DH gripper (call after FR3 RPC is up).

        Returns True if the gripper is ready, False if skipped or unavailable (arm-only).
        """
        if not self.use_dh_gripper or self._dh_gripper_kwargs is None:
            return False
        if self.dh_gripper is not None:
            return True
        port = self._dh_gripper_kwargs.get("port", "?")
        try:
            from xrobotoolkit_teleop.hardware.interface.dh_pgc_gripper import (
                DHPGCGripperTeleopInterface,
            )

            self.dh_gripper = DHPGCGripperTeleopInterface(**self._dh_gripper_kwargs)
            return True
        except Exception as e:
            print(
                f"WARNING: DH gripper on {port} unavailable ({e}). "
                "Continuing in arm-only mode."
            )
            self.use_dh_gripper = False
            self._dh_gripper_kwargs = None
            return False

    def reset(self):
        self.robot_controller.reset()

    def reset_to_home(self, open_gripper: bool = True) -> None:
        """退出 ServoJ，MoveJ 回 Home，同步遥操作状态。遥操作期间调用安全。"""
        with self._control_lock:
            self._reset_in_progress = True

        try:
            # robot 线程检测到 _reset_in_progress 后会 ServoMoveEnd；此处兜底。
            if self._servo_engaged:
                self.robot_controller.servo_stop()
                self._servo_engaged = False

            self.robot_controller.reset()
            q_home = self.robot_controller.initial_joint_positions.copy()

            if open_gripper and self.dh_gripper is not None:
                self.dh_gripper.open_gripper()
                self._pending_gripper_trigger = 0.0

            with self._control_lock:
                self.target_q = q_home.copy()
                self.control_active = False
            self.placo_robot.state.q[self.arm_q_slice] = q_home
            self.placo_robot.update_kinematics()

            for name, config in self.manipulator_config.items():
                self.init_ee_xyz[name] = None
                self.init_ee_quat[name] = None
                self.init_controller_xyz[name] = None
                self.init_controller_quat[name] = None
                self.effector_task[name].T_world_frame = self.placo_robot.get_T_world_frame(config["link_name"])
            self._soft_constraint.clear_all()
        finally:
            with self._control_lock:
                self._reset_in_progress = False

    def close(self):
        if self.dh_gripper is not None:
            self.dh_gripper.disconnect()
            self.dh_gripper = None
        if self.robot_controller is not None:
            self.robot_controller.close()
            self.robot_controller = None

    # ---------------- 线程 ----------------

    def run_robot_thread(self, stop_event):
        """FR3 高频关节伺服线程（等价 UR RTDE servoJ 线程）。

        默认 100Hz（FR3_SERVO_PERIOD=0.01）。伺服流生命周期：
        ServoMoveStart -> 循环 ServoJ -> （reset/退出时）ServoMoveEnd。
        """
        print(f"Starting FR3 arm control thread at {1.0 / self.robot_controller.servo_period:.0f}Hz...")
        servo_fail_reported = False
        while not stop_event.is_set():
            t_start = time.perf_counter()

            with self._control_lock:
                reset_in_progress = self._reset_in_progress
                q_cmd = self.target_q.copy()

            if reset_in_progress:
                # MoveJ 回 Home 期间必须退出伺服模式。
                if self.robot_controller._servo_active:
                    self.robot_controller.servo_stop()
                    self._servo_engaged = False
                time.sleep(0.02)
                continue

            if not self.robot_controller._servo_active:
                if not self.robot_controller.servo_start():
                    if not servo_fail_reported:
                        print(
                            "WARNING: ServoMoveStart rejected. 检查示教器：自动模式、"
                            "已使能、无报警，且无其他程序占用控制。"
                        )
                        servo_fail_reported = True
                    time.sleep(0.1)
                    continue

            ok = self.robot_controller.servo_joints(q_cmd)
            self._servo_engaged = True
            if not ok and not servo_fail_reported:
                print(
                    "WARNING: FR3 ServoJ rejected. 检查示教器：自动模式、已使能、"
                    "无报警，且无其他程序占用 ServoJ。"
                )
                servo_fail_reported = True

            # 周期对齐（等价 UR 的 initPeriod/waitPeriod）。
            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.0, self.robot_controller.servo_period - elapsed))

    def run_dh_gripper_thread(self, stop_event):
        """Modbus gripper I/O off the IK/servo path to avoid arm stutter."""
        print("Starting DH gripper control thread...")
        while not stop_event.is_set():
            if self.dh_gripper is not None:
                self.dh_gripper.update_from_trigger(self._pending_gripper_trigger)
            time.sleep(0.02)

    def run_ik_thread(self, stop_event):
        print("Starting IK calculation thread...")
        while not stop_event.is_set():
            try:
                start = time.time()
                self.calc_target_joint_position()
                elapsed = time.time() - start
                time.sleep(max(0, 0.01 - elapsed))
            except Exception as e:
                print(f"Error in IK calculation: {e}")
