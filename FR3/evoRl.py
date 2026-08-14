#!/usr/bin/env python3
"""
FR3 EvoRL 推理控制脚本

通过 ZMQ 连接远程 evo1 推理服务器（GPU 端），获取目标关节角并驱动 FR3 机械臂。

工作流：
  1. ZMQ REQ 连接推理服务器 (192.168.0.104:5555)
  2. 每轮采集：相机图像 + 机器人当前关节角 → 打包发给服务器
  3. 收到 action（目标关节角）→ ServoJ 实时驱动机械臂
  4. 循环执行，直到收到 stop 或用户中断

服务端协议（ZMQ REQ/REP, JSON）：
  发送: {"image_main": <base64 JPEG>, "image_pip": <base64 JPEG>,
         "state": [q1..q6, gripper], "task": "task description"}
  接收: {"action": [q1..q6] 或 [q1..q6, gripper], "status": "ok"/"error",
         "done": true/false}

使用方式：
  # 启动（需 RealSense 相机已连接）
  python FR3/evoRl.py --task "pick up the red block"

  # 指定推理服务器地址
  python FR3/evoRl.py --server-ip 192.168.0.104 --server-port 5555 \
      --task "pick up the red block"

  # 无相机模式（用零图像测试通信 + 机器人控制链路）
  python FR3/evoRl.py --no-camera --task "test"

  # 跳过安全确认
  python FR3/evoRl.py --task "..." --yes
"""

import argparse
import base64
import io
import math
import os
import sys
import time

import cv2
import numpy as np
import zmq

# 图像保存目录
SAVE_IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imges")

# ── SDK 导入（与 simple.py 相同策略） ──────────────────────────────
SDK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fairino-python-sdk", "linux")
if os.path.isdir(SDK_PATH):
    sys.path.insert(0, SDK_PATH)

from fairino import Robot  # noqa: E402

# ── 默认常量 ──────────────────────────────────────────────────────
ROBOT_IP = "192.168.58.2"
SERVER_IP = "192.168.0.104"
SERVER_PORT = 5555
MOVEJ_VEL_DEG = 15.0
ZMQ_TIMEOUT_MS = 10000    # ZMQ 等待推理结果超时 (ms)
JPEG_QUALITY = 70         # JPEG 压缩质量（0-100）

# FR3 关节软限位（度）
JOINT_LIMITS_DEG = [
    (-175.0, 175.0), (-265.0, 85.0), (-150.0, 150.0),
    (-265.0, 85.0),  (-175.0, 175.0), (-175.0, 175.0),
]
JOINT_LIMITS_LOW_RAD  = np.array([math.radians(lo) for lo, _ in JOINT_LIMITS_DEG])
JOINT_LIMITS_HIGH_RAD = np.array([math.radians(hi) for _, hi in JOINT_LIMITS_DEG])


# ── 工具函数 ──────────────────────────────────────────────────────

def connect_robot(robot_ip: str):
    """连接 + 报警检查 + 使能，失败退出。"""
    robot = Robot.RPC(robot_ip)
    time.sleep(0.5)
    if not getattr(robot, "is_connect", True):
        print(f"[!] 连接机器人失败（{robot_ip}），请检查 IP 与网线连接")
        sys.exit(1)
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
        print(f"[!] RobotEnable(1) 失败，错误码 {err}，请确认无急停/报警")
        sys.exit(1)
    print("[✓] 机器人已使能，自动模式就绪")
    return robot


def get_current_joints_rad(robot) -> np.ndarray:
    """读取当前关节角（弧度，6 维）。"""
    ret = robot.GetActualJointPosRadian(1)
    if isinstance(ret, tuple) and ret[0] == 0 and ret[1] is not None:
        return np.array(ret[1], dtype=np.float64)
    raise RuntimeError(f"读取关节角失败: {ret}")


def clip_joints(q_rad: np.ndarray) -> np.ndarray:
    """关节角裁剪到软限位。"""
    clipped = np.clip(q_rad, JOINT_LIMITS_LOW_RAD, JOINT_LIMITS_HIGH_RAD)
    diff = float(np.max(np.abs(clipped - q_rad)))
    if diff > 1e-6:
        print(f"[!] 关节角被裁剪（最大偏差 {math.degrees(diff):.3f}°）")
    return clipped


def frame_to_base64_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> str:
    """OpenCV BGR 图像 → base64 JPEG 字符串。"""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ── 相机管理 ──────────────────────────────────────────────────────

class CameraManager:
    """相机管理：支持 RealSense（主+辅）或 OpenCV USB 相机。"""

    def __init__(self, no_camera: bool = False):
        self.no_camera = no_camera
        self.cam_main = None   # 主相机 (image_main)
        self.cam_pip = None    # 辅相机 (image_pip)
        self._rs_pipelines = []
        self._cap = None       # OpenCV 备选

    def start(self):
        if self.no_camera:
            print("[*] 无相机模式：图像帧将使用全零占位")
            return True

        # 尝试 RealSense
        if self._try_realsense():
            return True

        # 回退 OpenCV USB 相机
        print("[*] RealSense 不可用，尝试 OpenCV USB 相机...")
        return self._try_opencv()

    def _try_realsense(self) -> bool:
        try:
            import pyrealsense2 as rs
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) == 0:
                return False

            print(f"[✓] 发现 {len(devices)} 个 RealSense 设备")
            for i, dev in enumerate(devices):
                name = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                print(f"  [{i}] {name} (SN: {serial})")

            # 主相机 pipeline
            cfg_main = rs.config()
            cfg_main.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            pipe_main = rs.pipeline()
            pipe_main.start(cfg_main)
            self._rs_pipelines.append(pipe_main)
            self.cam_main = ("rs", pipe_main)
            print("[✓] RealSense 主相机已启动 (640x480@30)")

            # 辅相机（如有第二个设备）
            if len(devices) >= 2:
                serial2 = devices[1].get_info(rs.camera_info.serial_number)
                cfg_pip = rs.config()
                cfg_pip.enable_device(serial2)
                cfg_pip.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                pipe_pip = rs.pipeline()
                pipe_pip.start(cfg_pip)
                self._rs_pipelines.append(pipe_pip)
                self.cam_pip = ("rs", pipe_pip)
                print("[✓] RealSense 辅相机已启动")
            else:
                # 单相机：主辅用同一画面
                self.cam_pip = self.cam_main
                print("[*] 单相机模式：image_pip 复用主相机画面")

            # 预热
            time.sleep(1.0)
            return True

        except Exception as exc:
            print(f"[*] RealSense 初始化失败: {exc}")
            return False

    def _try_opencv(self) -> bool:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[!] 无可用相机（RealSense 和 OpenCV 均失败）")
            return False
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cap = cap
        self.cam_main = ("cv", cap)
        self.cam_pip = self.cam_main  # 单相机复用
        print("[✓] OpenCV USB 相机已启动 (640x480)")
        return True

    def capture(self):
        """采集一帧，返回 (frame_main, frame_pip)，均为 BGR numpy 数组。
        无相机模式返回 None, None。"""
        if self.no_camera:
            return None, None

        frame_main = self._grab(self.cam_main)
        frame_pip = self._grab(self.cam_pip) if self.cam_pip is not self.cam_main else frame_main
        return frame_main, frame_pip

    def _grab(self, cam):
        kind, obj = cam
        if kind == "rs":
            import pyrealsense2 as rs
            frames = obj.wait_for_frames(timeout_ms=500)
            color = frames.get_color_frame()
            if color:
                return np.asanyarray(color.get_data())
        elif kind == "cv":
            ret, frame = obj.read()
            if ret:
                return frame
        # 失败：返回黑色占位帧
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def stop(self):
        for pipe in self._rs_pipelines:
            try:
                pipe.stop()
            except Exception:
                pass
        if self._cap is not None:
            self._cap.release()


# ── ZMQ 推理客户端 ─────────────────────────────────────────────────

class InferenceClient:
    """ZMQ REQ 客户端，与 evo1 推理服务器通信。"""

    def __init__(self, server_ip: str, server_port: int, timeout_ms: int = ZMQ_TIMEOUT_MS):
        self._endpoint = f"tcp://{server_ip}:{server_port}"
        self._timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.socket.setsockopt(zmq.LINGER, 0)

        self.socket.connect(self._endpoint)
        print(f"[✓] ZMQ 客户端已连接: {self._endpoint}")

    def request_action(self, image_main_b64: str, image_pip_b64: str,
                       state: list, task: str) -> dict:
        """发送观测，接收推理结果。

        Returns:
            {"action": [...], "status": "ok"} 或 {"action": [], "status": "error", ...}
            超时/异常时返回 {"action": [], "status": "timeout"}
        """
        msg = {
            "image_main": image_main_b64,
            "image_pip": image_pip_b64,
            "state": state,
            "task": task,
        }
        try:
            self.socket.send_json(msg)
            reply = self.socket.recv_json()
            return reply
        except zmq.Again:
            # 超时：REQ socket 需要重置才能继续
            print("[!] ZMQ 接收超时，重置 socket...")
            self._reset_socket()
            return {"action": [], "status": "timeout"}
        except Exception as exc:
            print(f"[!] ZMQ 通信异常: {exc}")
            self._reset_socket()
            return {"action": [], "status": "error", "message": str(exc)}

    def _reset_socket(self):
        """REQ socket 超时后需关闭重建，否则状态机卡死。"""
        try:
            self.socket.close()
        except Exception:
            pass
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, self._timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, 5000)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(self._endpoint)

    def close(self):
        try:
            self.socket.close()
        except Exception:
            pass
        try:
            self.context.term()
        except Exception:
            pass


# ── MoveJ 运动 ───────────────────────────────────────────────────

def move_joints(robot, q_rad: np.ndarray, vel_deg: float = MOVEJ_VEL_DEG) -> bool:
    """MoveJ 到目标关节角（弧度），等待运动完成后返回。

    与 ServoJ 不同，MoveJ 是一次性指令：控制器自己规划轨迹，
    不需要持续喂 100Hz 指令，等价 simple.py 的 test_joints。
    """
    q_deg = np.rad2deg(q_rad).tolist()
    err = robot.MoveJ(q_deg, tool=0, user=0, vel=vel_deg)
    if err != 0:
        print(f"[!] MoveJ 失败，错误码 {err}")
        return False
    # 等待运动完成
    t0 = time.time()
    while time.time() - t0 < 60.0:
        try:
            err, queue_len = robot.GetMotionQueueLength()
            if err == 0 and queue_len == 0:
                return True
        except Exception:
            pass
        time.sleep(0.05)
    print("[!] MoveJ 等待超时")
    return False


# ── 主控制循环 ─────────────────────────────────────────────────────

def run_inference_loop(robot, zmq_client: InferenceClient,
                       camera: CameraManager,
                       task: str, max_steps: int = 0):
    """同步主循环：采集 → 请求推理 → MoveJ 移动机械臂 → 读新状态 → 循环。

    每一步严格串行：发数据给服务端 → 收 action → MoveJ 到位 → 下一步。

    Args:
        robot: FR3 机器人实例，None 表示不连接机械臂（仅测试通信）
        max_steps: 最大步数，0 表示无限循环
    """
    no_robot = robot is None
    # 占位图像（无相机模式）
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    placeholder_b64 = frame_to_base64_jpeg(placeholder)

    # 读取当前关节角
    if no_robot:
        current_q = np.zeros(6, dtype=np.float64)
        print("[*] 无机械臂：关节角使用全零占位")
    else:
        try:
            current_q = get_current_joints_rad(robot)
        except Exception as exc:
            print(f"[!] 读取初始关节角失败: {exc}")
            return

    gripper_cmd = 0.0
    raw_action: list = []  # 服务端原始返回值（打印用）
    step = 0
    t_start = time.perf_counter()
    success_count = 0
    error_count = 0

    motion_label = "MoveJ" if robot is not None else "不下发运动"
    print(f"\n[*] 开始同步推理循环（{motion_label}）")
    print(f"    任务: {task}")
    print(f"    初始关节角(deg): {[round(v, 1) for v in np.rad2deg(current_q)]}\n")

    try:
        while True:
            if max_steps > 0 and step >= max_steps:
                print(f"\n[*] 达到最大步数 {max_steps}，停止")
                break

            # ── 1. 采集图像 ──
            frame_main, frame_pip = camera.capture()
            if frame_main is not None:
                img_main_b64 = frame_to_base64_jpeg(frame_main)
                img_pip_b64 = frame_to_base64_jpeg(frame_pip if frame_pip is not None else frame_main)
            else:
                img_main_b64 = placeholder_b64
                img_pip_b64 = placeholder_b64

            # 前 3 步保存图像到 imges/ 供调试
            if step < 3:
                os.makedirs(SAVE_IMG_DIR, exist_ok=True)
                for name, frame in [("main", frame_main), ("pip", frame_pip)]:
                    if frame is not None:
                        path = os.path.join(SAVE_IMG_DIR, f"step{step}_{name}.jpg")
                        cv2.imwrite(path, frame)
                if step == 0:
                    print(f"[*] 图像已保存到 {SAVE_IMG_DIR}/")

            # state = [q1..q6, gripper] (7 维)
            state = list(current_q) + [gripper_cmd]

            # ── 2. 请求推理 ──
            reply = zmq_client.request_action(img_main_b64, img_pip_b64, state, task)
            status = reply.get("status", "unknown")

            target_q = current_q.copy()  # 默认保持当前位置
            done = bool(reply.get("done", False))  # 无论 status 如何都读取 done
            if status == "ok":
                action = reply.get("action", [])
                raw_action = list(action)  # 保存服务端原始返回值
                if len(action) >= 6:
                    target_q = clip_joints(np.array(action[:6], dtype=np.float64))
                    if len(action) >= 7:
                        gripper_cmd = float(action[6])
                    success_count += 1
                else:
                    print(f"[!] action 维度不足（{len(action)}），保持当前位置")
                    error_count += 1
            elif status == "timeout":
                print(f"[!] 推理超时（step {step}），保持当前位置")
                error_count += 1
            else:
                msg = reply.get("message", "unknown")
                print(f"[!] 推理错误（step {step}）: {msg}")
                error_count += 1
                # 即使报错，如果服务端明确返回 done=true，也停止
                if done:
                    print("[*] 服务端返回 done=true，忽略错误并停止")

            # ── 3. MoveJ 到目标关节角（无机械臂时跳过；done-only 包无 action 时保持当前位置） ──
            if robot is not None and status == "ok":
                target_deg = np.rad2deg(target_q)
                print(f"    MoveJ 目标(deg): {[round(v, 1) for v in target_deg[:3]]}")
                move_joints(robot, target_q)

            # ── 4. 读新状态（下一步发给服务端） ──
            if not no_robot:
                try:
                    current_q = get_current_joints_rad(robot)
                except Exception:
                    pass

            step += 1

            # 打印每一步
            dt = time.perf_counter() - t_start
            fps = step / max(dt, 1e-6)
            raw_str = ", ".join(f"{v:.4f}" for v in raw_action)
            q_deg = np.rad2deg(current_q)
            q_str = ", ".join(f"{v:.1f}" for v in q_deg[:3])
            done_str = " [DONE]" if done else ""
            print(f"  [{step} 步] {fps:.1f} step/s | "
                  f"成功 {success_count} / 失败 {error_count} | "
                  f"服务端 action = [{raw_str}] | "
                  f"当前 q_deg[:3] = [{q_str}°]{done_str}")

            # 服务端返回 done=true 时优雅结束
            if done:
                print("\n[*] 服务端返回 done=true，任务完成，停止循环")
                break

    except KeyboardInterrupt:
        print(f"\n[!] 用户中断（已执行 {step} 步）")
    finally:
        dt = time.perf_counter() - t_start
        print(f"\n[✓] 控制循环结束：{step} 步, {dt:.1f}s, "
              f"成功 {success_count}, 失败 {error_count}")


# ── 入口 ──────────────────────────────────────────────────────────

def confirm_motion(yes=False):
    if yes:
        return True
    print("\n[安全确认] 脚本将控制机械臂运动（MoveJ）。"
          "\n  请确认机械臂周围无障碍物、急停按钮可随时触达。")
    answer = input("输入 yes 继续，其他任意键取消: ").strip().lower()
    return answer == "yes"


def main():
    parser = argparse.ArgumentParser(
        description="FR3 EvoRL 推理控制（ZMQ 连接 evo1 服务器，驱动 FR3 机械臂）"
    )
    # 服务器
    parser.add_argument("--server-ip", default=SERVER_IP,
                        help=f"推理服务器 IP（默认 {SERVER_IP}）")
    parser.add_argument("--server-port", type=int, default=SERVER_PORT,
                        help=f"推理服务器 ZMQ 端口（默认 {SERVER_PORT}）")
    # 机器人
    parser.add_argument("--robot-ip", default=ROBOT_IP,
                        help=f"FR3 控制箱 IP（默认 {ROBOT_IP}）")
    parser.add_argument("--movej-vel", type=float, default=MOVEJ_VEL_DEG,
                        help=f"MoveJ 速度 °/s（默认 {MOVEJ_VEL_DEG}）")
    # 任务
    parser.add_argument("--task", type=str, default="pick up the object",
                        help="任务描述文本（发给推理服务器）")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="最大推理步数，0 表示无限循环（默认 0）")
    # 相机
    parser.add_argument("--no-camera", action="store_true",
                        help="无相机模式（用零图像占位，测试通信+控制链路）")
    parser.add_argument("--no-robot", action="store_true",
                        help="不连接机械臂（仅测试 ZMQ 推理通信）")
    # 其他
    parser.add_argument("--zmq-timeout", type=int, default=ZMQ_TIMEOUT_MS,
                        help=f"ZMQ 接收超时 ms（默认 {ZMQ_TIMEOUT_MS}）")
    parser.add_argument("--yes", action="store_true",
                        help="跳过安全确认")
    args = parser.parse_args()

    if not args.no_robot and not confirm_motion(args.yes):
        print("已取消。")
        sys.exit(0)

    # ── 1. 连接机器人（--no-robot 时跳过） ──
    robot = None
    if args.no_robot:
        print("[*] --no-robot：不连接机械臂，仅测试推理通信")
    else:
        robot = connect_robot(args.robot_ip)

    # ── 2. 初始化相机 ──
    camera = CameraManager(no_camera=args.no_camera)
    if not camera.start():
        if not args.no_camera:
            print("[!] 相机初始化失败，退出")
            if robot is not None:
                robot.CloseRPC()
            sys.exit(1)

    # ── 3. 初始化 ZMQ 客户端 ──
    zmq_client = InferenceClient(args.server_ip, args.server_port,
                                  timeout_ms=args.zmq_timeout)

    # ── 4. 主循环 ──
    try:
        run_inference_loop(robot, zmq_client, camera,
                           task=args.task, max_steps=args.max_steps)
    except KeyboardInterrupt:
        print("\n[!] 用户中断，停止运动")
        if robot is not None:
            try:
                robot.StopMotion()
            except Exception:
                pass
    finally:
        zmq_client.close()
        camera.stop()
        if robot is not None:
            try:
                robot.CloseRPC()
            except Exception:
                pass
        print("[✓] 全部资源已释放")


if __name__ == "__main__":
    main()
