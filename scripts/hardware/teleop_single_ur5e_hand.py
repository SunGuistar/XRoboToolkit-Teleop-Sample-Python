"""
单 UR5e 灵巧手遥操作 + RealSense 相机 BC 数据采集（无 VR 视频推流）。

与 BC 录制按键共用一条 XRoboToolkit SDK 连接。

常用命令（实验室默认）：
  python scripts/hardware/teleop_single_ur5e_hand.py

  # 更高分辨率（D435 + D435I 双机时会自动限制 fps）：
  python scripts/hardware/teleop_single_ur5e_hand.py --quality balanced

  # 仅主相机（录制不含辅相机画面）：
  python scripts/hardware/teleop_single_ur5e_hand.py --no-pip

手柄操作（默认 controller_side=right）：
  - 按住 Grip：移动机械臂（人工操作始终优先于策略）
  - X：开始/停止 BC 录制（停止后自动保存并回 Home）
  - Y：丢弃当前录制（并回 Home）

Home 姿态：
  - 固定保存在 scripts/hardware/teleop_single_ur5e_hand.home.json（非每次连接时重读）
  - 更新 Home：机械臂摆好后执行  python scripts/hardware/teleop_single_ur5e_hand.py --save-home
  - BC 录制结束 / --reset 时回到该 Home

BC 数据采集（V1）：
  - 独立 10Hz 录制线程
  - 训练图像：主相机 + 辅相机（可用 --no-pip / --no-record-save-pip 关闭辅相机）
  - 原始 episode：logs/ur5e_bc/raw/episode_XXXXX.pkl

  observation.state = [qpos(6), gripper_state(1)]   — 0=张开, 1=闭合（扳机代理，无物理夹爪）
  action            = [target_q(6), gripper_cmd(1)] — 0=张开, 1=闭合

  检查 episode：python scripts/check_ur5e_bc_episode.py logs/ur5e_bc/raw/episode_00000.pkl
  转 LeRobot：  python scripts/convert_ur5e_raw_to_lerobot.py --raw-dir logs/ur5e_bc/raw

启动顺序：相机预热与健康检查 → 机械臂线程 → BC 录制器 → 遥操作（Ctrl+C 退出）。
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

from xrobotoolkit_teleop.bc.config import BCConfig
from xrobotoolkit_teleop.bc.recorder import BCEpisodeRecorder
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.universal_robots import (
    DEFAULT_ROBOT_IP,
    LEFT_ROBOT_IP,
    RIGHT_ROBOT_IP,
)
from xrobotoolkit_teleop.hardware.single_arm_ur_controller import (
    LEFT_CONTROLLER_MANIPULATOR_CONFIG,
    RIGHT_CONTROLLER_MANIPULATOR_CONFIG,
    SingleArmURController,
)

_RS_UTILS_PATH = Path(__file__).resolve().parents[1] / "visualization" / "realsense_vr_stream.py"
_spec = importlib.util.spec_from_file_location("realsense_vr_stream", _RS_UTILS_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"无法加载 RealSense 工具模块：{_RS_UTILS_PATH}")
rs_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rs_utils)

DEFAULT_MAIN_SERIAL = "406122070449"
DEFAULT_PIP_SERIAL = "152122074297"
_HOME_CACHE_PATH = Path(__file__).with_name("teleop_single_ur5e_hand.home.json")


def _format_joint_deg(q_deg: np.ndarray) -> str:
    return np.array2string(q_deg, precision=2, separator=", ", suppress_small=True)


def _save_home_cache(q_deg: np.ndarray) -> None:
    payload = {
        "joint_deg": [float(v) for v in q_deg[:6]],
        "note": "固定 Home；仅 --save-home 时更新，遥操作启动不会覆盖。",
    }
    _HOME_CACHE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_home_cache() -> np.ndarray | None:
    if not _HOME_CACHE_PATH.exists():
        return None
    data = json.loads(_HOME_CACHE_PATH.read_text(encoding="utf-8"))
    q_deg = np.array(data["joint_deg"], dtype=float)
    if q_deg.shape != (6,):
        raise RuntimeError(f"Home 文件格式无效：{_HOME_CACHE_PATH}")
    return q_deg


def _resolve_home_joint_deg(
    cli_joint_deg: tuple[float, float, float, float, float, float] | None,
) -> np.ndarray:
    if cli_joint_deg is not None:
        return np.array(cli_joint_deg, dtype=float)
    cached = _load_home_cache()
    if cached is not None:
        return cached
    raise RuntimeError(
        f"未找到 Home 文件：{_HOME_CACHE_PATH}\n"
        "请先将机械臂摆到目标 Home 位，再执行：\n"
        "  python scripts/hardware/teleop_single_ur5e_hand.py --save-home"
    )


def _save_home_from_robot(robot_ip: str, manipulator_config: dict) -> None:
    """读取机械臂当前关节角并写入 Home 文件（仅此命令会更改 Home）。"""
    xr_client = XrClient()
    arm_controller: SingleArmURController | None = None
    try:
        arm_controller = SingleArmURController(
            xr_client,
            manipulator_config=manipulator_config,
            robot_ip=robot_ip,
            initial_joint_deg=np.zeros(6, dtype=float),
        )
        q_rad = arm_controller.robot_controller.get_current_joint_positions()[:6].copy()
        q_deg = np.rad2deg(q_rad)
        _save_home_cache(q_deg)
        print(f"Home 已保存（度）：{_format_joint_deg(q_deg)}")
        print(f"文件：{_HOME_CACHE_PATH}")
    finally:
        if arm_controller is not None:
            arm_controller.close()
        xr_client.close()


class _CameraReader:
    """后台线程读取 RealSense 彩色帧。"""

    def __init__(self, pipeline, name: str):
        self._pipeline = pipeline
        self._name = name
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._stop = threading.Event()
        self._errors = 0
        self._thread = threading.Thread(target=self._loop, name=f"rs-{name}", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                color = frames.get_color_frame()
                if not color:
                    continue
                data = np.asanyarray(color.get_data()).copy()
                with self._lock:
                    self._frame = data
            except RuntimeError as exc:
                self._errors += 1
                if self._errors == 1 or self._errors % 30 == 0:
                    print(f"警告：{self._name} 取帧异常：{exc}", file=sys.stderr)
                time.sleep(0.01)

    def get_latest(self, timeout_s: float = 2.0) -> np.ndarray:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                if self._frame is not None:
                    return self._frame.copy()
            time.sleep(0.005)
        raise RuntimeError(f"{self._name} 在 {timeout_s:.1f}s 内未收到画面")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)


def _list_realsense_devices() -> list[tuple[str, str]]:
    devices = rs_utils.list_devices()
    if not devices:
        raise RuntimeError("未检测到 RealSense 相机。")
    print("已连接 RealSense 相机：")
    for name, sn in devices:
        print(f"  {name}  (序列号: {sn})")
    return devices


def _validate_serial(serial: str, devices: list[tuple[str, str]], label: str) -> None:
    if serial not in {s for _, s in devices}:
        raise RuntimeError(f"未找到{label}相机，序列号 {serial}。")


def _resolve_dual_color_stream(
    serials: list[str],
    width: int,
    height: int,
    fps: int,
) -> tuple[int, int, int]:
    """协商双相机共用的彩色流参数（提示信息为中文）。"""
    color_profiles = getattr(rs_utils, "_color_bgr8_profiles")
    device_profiles = [color_profiles(serial) for serial in serials]
    for serial, profiles in zip(serials, device_profiles, strict=True):
        if not profiles:
            raise RuntimeError(f"未找到 RealSense 相机，序列号 {serial}。")

    requested = (width, height, fps)
    if all(requested in profiles for profiles in device_profiles):
        return width, height, fps

    common_fps_at_res = sorted(
        {
            profile_fps
            for profile_fps in {p[2] for p in device_profiles[0] if (p[0], p[1]) == (width, height)}
            if all((width, height, profile_fps) in profiles for profiles in device_profiles)
        },
        reverse=True,
    )
    usable_fps = [profile_fps for profile_fps in common_fps_at_res if profile_fps <= fps]
    if usable_fps:
        resolved_fps = usable_fps[0]
        if (width, height, resolved_fps) != requested:
            labels = "、".join(serials)
            print(
                f"双相机 ({labels}) 不支持 {width}x{height}@{fps}fps，"
                f"改用 {width}x{height}@{resolved_fps}fps。"
            )
        return width, height, resolved_fps

    for fallback in ((640, 480, 30), (640, 480, 15), (424, 240, 30)):
        if all(fallback in profiles for profiles in device_profiles):
            if fallback != requested:
                labels = "、".join(serials)
                print(
                    f"双相机 ({labels}) 不支持 {width}x{height}@{fps}fps，"
                    f"回退至 {fallback[0]}x{fallback[1]}@{fallback[2]}fps。"
                )
            return fallback

    raise RuntimeError(
        f"相机 {serials} 无共同支持的彩色流（请求 {width}x{height}@{fps}fps）。"
    )


def _warmup_camera(serial: str, capture_width: int, capture_height: int, fps: int):
    try:
        return rs_utils.warmup_camera(serial, capture_width, capture_height, fps)
    except RuntimeError as exc:
        raise RuntimeError(
            f"RealSense {serial} 启动失败（{capture_width}x{capture_height} @ {fps}fps）：{exc}"
        ) from exc


def _run_reset(robot_ip: str, home_joint_deg: np.ndarray, manipulator_config: dict) -> None:
    print(f"复位：正在将 UR 移至 Home 位（{robot_ip}）...")
    print(f"  目标关节角（度）：{_format_joint_deg(home_joint_deg)}")
    xr_client = XrClient()
    arm_controller: SingleArmURController | None = None
    try:
        arm_controller = SingleArmURController(
            xr_client,
            manipulator_config=manipulator_config,
            robot_ip=robot_ip,
            initial_joint_deg=home_joint_deg,
        )
        arm_controller.reset_to_home(open_gripper=False)
        print("机械臂复位完成。")
    except Exception as e:
        print(f"复位失败：{e}")
        raise SystemExit(1) from e
    finally:
        if arm_controller is not None:
            arm_controller.close()
        xr_client.close()


def _start_arm_threads(arm_controller: SingleArmURController, stop_signal: threading.Event) -> list[threading.Thread]:
    threads = [
        threading.Thread(target=arm_controller.run_robot_thread, args=(stop_signal,), name="ur-servo"),
        threading.Thread(target=arm_controller.run_ik_thread, args=(stop_signal,), name="ur-ik"),
    ]
    for t in threads:
        t.start()
    return threads


def _verify_camera(reader: _CameraReader, label: str, timeout_s: float = 5.0) -> tuple[int, int]:
    frame = reader.get_latest(timeout_s=timeout_s)
    h, w = frame.shape[:2]
    return h, w


def _print_camera_status(
    main_label: str,
    main_sn: str,
    main_size: tuple[int, int],
    fps: int,
    *,
    aux_label: str | None = None,
    aux_sn: str | None = None,
    aux_size: tuple[int, int] | None = None,
) -> None:
    mh, mw = main_size
    print(f"  主相机 ({main_label}，{main_sn})：{mw}x{mh} @ {fps}fps  正常")
    if aux_label is not None and aux_sn is not None and aux_size is not None:
        ph, pw = aux_size
        print(f"  辅相机 ({aux_label}，{aux_sn})：{pw}x{ph} @ {fps}fps  正常")


def _run_camera_health_monitor(
    readers: list[_CameraReader],
    labels: list[str],
    stop_signal: threading.Event,
    interval_s: float = 10.0,
) -> threading.Thread:
    def _loop() -> None:
        while not stop_signal.wait(interval_s):
            for reader, label in zip(readers, labels, strict=True):
                try:
                    reader.get_latest(timeout_s=1.0)
                except RuntimeError:
                    print(f"相机健康检查：{label} 无画面输出", file=sys.stderr)

    thread = threading.Thread(target=_loop, name="camera-health", daemon=True)
    thread.start()
    return thread


def _init_cameras(
    main_serial: str,
    aux_serial: str | None,
    capture_width: int,
    capture_height: int,
    fps: int,
) -> tuple[list, list[_CameraReader], int, int, int]:
    devices = _list_realsense_devices()
    _validate_serial(main_serial, devices, "主")

    if aux_serial:
        _validate_serial(aux_serial, devices, "辅")
        if main_serial == aux_serial:
            raise RuntimeError("主相机与辅相机序列号不能相同。")
        capture_width, capture_height, fps = _resolve_dual_color_stream(
            [main_serial, aux_serial], capture_width, capture_height, fps
        )
        print(f"正在启动 RealSense 双相机（{capture_width}x{capture_height} @ {fps}fps）...")
        main_pipeline, _ = _warmup_camera(main_serial, capture_width, capture_height, fps)
        aux_pipeline, _ = _warmup_camera(aux_serial, capture_width, capture_height, fps)
        main_reader = _CameraReader(main_pipeline, "主相机")
        aux_reader = _CameraReader(aux_pipeline, "辅相机")
        pipelines = [main_pipeline, aux_pipeline]
        readers = [main_reader, aux_reader]
        aux_sn = aux_serial
    else:
        print(f"正在启动 RealSense 相机（{capture_width}x{capture_height} @ {fps}fps）...")
        main_pipeline, _ = _warmup_camera(main_serial, capture_width, capture_height, fps)
        main_reader = _CameraReader(main_pipeline, "主相机")
        pipelines = [main_pipeline]
        readers = [main_reader]
        aux_sn = None

    main_label = rs_utils._device_label(main_serial)
    aux_label = rs_utils._device_label(aux_sn) if aux_sn else None

    print("正在验证相机画面...")
    main_size = _verify_camera(readers[0], "主相机")
    aux_size = None
    if len(readers) > 1:
        aux_size = _verify_camera(readers[1], "辅相机")

    print("相机就绪：")
    _print_camera_status(
        main_label,
        main_serial,
        main_size,
        fps,
        aux_label=aux_label,
        aux_sn=aux_sn,
        aux_size=aux_size,
    )

    return pipelines, readers, capture_width, capture_height, fps


def _run_teleop_loop(stop_signal: threading.Event) -> None:
    print("遥操作已启动 — 按住 Grip 移动机械臂，X/Y 控制 BC 录制（Ctrl+C 退出）。")
    try:
        while not stop_signal.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n用户中断，正在退出。")


def main(
    reset: bool = False,
    save_home: bool = False,
    visual: bool = False,
    robot_ip: str = DEFAULT_ROBOT_IP,
    controller_side: Literal["left", "right"] = "right",
    initial_joint_deg: tuple[float, float, float, float, float, float] | None = None,
    main_serial: str = DEFAULT_MAIN_SERIAL,
    pip_serial: str = DEFAULT_PIP_SERIAL,
    no_pip: bool = False,
    quality: Literal["low", "balanced", "high"] = "low",
    fps: int = 30,
    camera_health_interval: float = 10.0,
    enable_recording: bool = True,
    record_fps: float = 10.0,
    record_save_dir: str = "logs/ur5e_bc/raw",
    record_image_format: Literal["jpeg", "raw"] = "jpeg",
    record_save_pip: bool = True,
    record_auto_reset: bool = True,
    policy_mode: bool = False,
) -> None:
    if controller_side == "left":
        manipulator_config = LEFT_CONTROLLER_MANIPULATOR_CONFIG
        if robot_ip == DEFAULT_ROBOT_IP:
            robot_ip = LEFT_ROBOT_IP
    else:
        manipulator_config = RIGHT_CONTROLLER_MANIPULATOR_CONFIG

    if save_home:
        _save_home_from_robot(robot_ip, manipulator_config)
        return

    home_joint_deg = _resolve_home_joint_deg(initial_joint_deg)

    if reset:
        _run_reset(robot_ip, home_joint_deg, manipulator_config)
        return

    use_aux_serial = None if no_pip else pip_serial
    if no_pip:
        record_save_pip = False

    preset = rs_utils.QUALITY_PRESETS[quality]
    capture_width = preset["width"]
    capture_height = preset["height"]

    xr_client = XrClient()
    arm_controller: SingleArmURController | None = None
    bc_recorder: BCEpisodeRecorder | None = None
    stop_signal = threading.Event()
    arm_threads: list[threading.Thread] = []
    pipelines: list = []
    readers: list[_CameraReader] = []
    cameras_ready = False
    bc_config = BCConfig(
        record_fps=record_fps,
        save_dir=Path(record_save_dir),
        image_format=record_image_format,
        save_optional_pip_image=record_save_pip,
        auto_reset_after_episode=record_auto_reset,
    )

    def cleanup_cameras() -> None:
        for reader in readers:
            reader.stop()
        for pipeline in pipelines:
            try:
                pipeline.stop()
            except RuntimeError:
                pass

    try:
        arm_controller = SingleArmURController(
            xr_client,
            manipulator_config=manipulator_config,
            robot_ip=robot_ip,
            initial_joint_deg=home_joint_deg,
            visualize_placo=visual,
            policy_mode=policy_mode,
            bc_config=bc_config,
        )
        print(f"Home 关节角（度）：{_format_joint_deg(home_joint_deg)}")

        if policy_mode:
            print("policy_mode=True：请在遥操作前通过 arm_controller.set_policy() 挂载策略。")
        side_label = "左手柄" if controller_side == "left" else "右手柄"
        print(
            f"单臂遥操作：robot_ip={robot_ip}，{side_label}。"
            "按住 Grip 移动机械臂。"
        )
        arm_controller.calc_target_joint_position()

        pipelines, readers, capture_width, capture_height, fps = _init_cameras(
            main_serial, use_aux_serial, capture_width, capture_height, fps
        )
        cameras_ready = True
        quality_labels = {"low": "低", "balanced": "均衡", "high": "高"}
        print(f"采集画质：{quality_labels.get(quality, quality)}  {capture_width}x{capture_height} @ {fps}fps")

        health_labels = ["主相机"]
        if len(readers) > 1:
            health_labels.append("辅相机")
        _run_camera_health_monitor(readers, health_labels, stop_signal, interval_s=camera_health_interval)

        arm_threads = _start_arm_threads(arm_controller, stop_signal)

        if enable_recording:
            aux_reader = readers[1] if len(readers) > 1 else None
            bc_recorder = BCEpisodeRecorder(
                xr_client=xr_client,
                arm_controller=arm_controller,
                main_reader=readers[0],
                pip_reader=aux_reader,
                config=bc_config,
            )
            bc_recorder.start(stop_signal)
            print(f"BC 录制已开启，{record_fps}Hz → {bc_config.save_dir}")

        _run_teleop_loop(stop_signal)

    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        if cameras_ready:
            cleanup_cameras()
        raise SystemExit(1) from exc
    finally:
        stop_signal.set()
        if bc_recorder is not None:
            bc_recorder.stop()
        for t in arm_threads:
            t.join(timeout=3.0)
        if cameras_ready:
            cleanup_cameras()
        if arm_controller is not None:
            arm_controller.close()
        xr_client.close()
        print("遥操作已停止。")


if __name__ == "__main__":
    tyro.cli(main)
