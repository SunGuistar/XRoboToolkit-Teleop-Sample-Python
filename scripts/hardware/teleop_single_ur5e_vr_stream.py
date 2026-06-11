"""
Single UR5e teleop + RealSense PiP VR stream in one process (one XRoboToolkit SDK connection).

PC Service only allows one SDK feedback client; running teleop and realsense_vr_stream
in separate terminals breaks A-button swap or arm tracking. This script shares one XrClient.

Typical usage (lab defaults):
  python scripts/hardware/teleop_single_ur5e_vr_stream.py

  python scripts/hardware/teleop_single_ur5e_vr_stream.py \\
    --main-serial 406122070449 --pip-serial 148522072134 \\
    --quality balanced --pip-scale 0.38

  # VR 右下角 PiP 显示对准调试叠加（与 debug_alignment_vision.py 相同）
  python scripts/hardware/teleop_single_ur5e_vr_stream.py --debug

Controls (default controller_side=right):
  - Hold grip: move arm (human takeover always wins over policy)
  - Trigger: DH gripper
  - A: swap main / PiP cameras
  - X: 开始/停止 BC 录制（停止后自动保存并回 Home）
  - Y: 丢弃当前录制（并回 Home）
  - B (in ZEDMINI): 2D / stereo view toggle — not used for recording

BC data collection (V1):
  - Independent 10Hz logging thread (not tied to VR stream loop)
  - Training image: main camera only (pip optional, off by default)
  - Raw episodes: logs/ur5e_bc/raw/episode_XXXXX.pkl

  observation.state = [qpos(6), gripper_state(1)]   — 0=open, 1=close
  action            = [target_q(6), gripper_cmd(1)] — 0=open, 1=close

  Check episode:  python scripts/check_ur5e_bc_episode.py logs/ur5e_bc/raw/episode_00000.pkl
  Convert LeRobot: python scripts/convert_ur5e_raw_to_lerobot.py --raw-dir logs/ur5e_bc/raw

Order: script starts → camera warmup → arm threads → BC recorder → wait for PICO Listen → stream.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
from pathlib import Path
from collections.abc import Callable
from typing import Literal

import numpy as np
import tyro

from xrobotoolkit_teleop.bc.config import BCConfig
from xrobotoolkit_teleop.bc.recorder import BCEpisodeRecorder
from xrobotoolkit_teleop.shared_autonomy.alignment import AlignmentVisionRenderer
from xrobotoolkit_teleop.shared_autonomy.config import COLOR_HSV_PRESETS, TargetDetectionConfig
from xrobotoolkit_teleop.shared_autonomy.gripper_color_config import GripperColorConfig

_VR_STREAM_PATH = Path(__file__).resolve().parents[1] / "visualization" / "realsense_vr_stream.py"
_spec = importlib.util.spec_from_file_location("realsense_vr_stream", _VR_STREAM_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Cannot load VR stream module from {_VR_STREAM_PATH}")
vr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vr)
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.universal_robots import (
    DEFAULT_ROBOT_IP,
    LEFT_INITIAL_JOINT_DEG,
    LEFT_ROBOT_IP,
    RIGHT_INITIAL_JOINT_DEG,
    RIGHT_ROBOT_IP,
    URController,
)
from xrobotoolkit_teleop.hardware.single_arm_ur_controller import (
    LEFT_CONTROLLER_MANIPULATOR_CONFIG,
    RIGHT_CONTROLLER_MANIPULATOR_CONFIG,
    SingleArmURController,
)

DH_GRIPPER_PORT = "/dev/ttyUSB0"
DH_GRIPPER_FORCE = 50
DH_GRIPPER_SPEED = 50
DH_GRIPPER_SLAVE_ID = 1
DH_GRIPPER_BAUDRATE = 115200

DEFAULT_MAIN_SERIAL = "406122070449"
DEFAULT_PIP_SERIAL = "148522072134"


def _run_reset(robot_ip: str, initial_joint: np.ndarray) -> None:
    print(f"Reset: moving UR to home @ {robot_ip} (no VR / no DH gripper)...")
    ur = URController(
        robot_ip=robot_ip,
        initial_joint_positions=np.deg2rad(initial_joint),
        use_gripper=False,
    )
    try:
        ur.reset()
        print("Arm reset completed.")
    except Exception as e:
        print(f"Error during reset: {e}")
        ur.close()
        raise SystemExit(1) from e
    ur.close()


def _make_alignment_renderer() -> AlignmentVisionRenderer:
    """与 debug_alignment_vision.py 默认参数一致的对准视觉渲染器。"""
    preset_lower, preset_upper = COLOR_HSV_PRESETS["target"]
    target_config = TargetDetectionConfig(hsv_lower=preset_lower, hsv_upper=preset_upper)
    gripper_config = GripperColorConfig()
    return AlignmentVisionRenderer(target_config, gripper_config)


def _wrap_get_frame_with_alignment_debug(
    pip_source: vr.PipFrameSource,
    renderer: AlignmentVisionRenderer,
) -> Callable[[], np.ndarray]:
    """物理 pip 相机帧叠加对准调试画面，再按 swap 状态合成 VR 流。"""

    def get_frame() -> np.ndarray:
        main_color = pip_source._main_reader.get_latest()
        pip_vis = renderer.render(pip_source._pip_reader.get_latest())
        with pip_source._lock:
            swapped = pip_source._swapped
        if swapped:
            return vr.compose_pip_frame(
                pip_vis, main_color, pip_source._pip_scale, pip_source._pip_margin
            )
        return vr.compose_pip_frame(
            main_color, pip_vis, pip_source._pip_scale, pip_source._pip_margin
        )

    return get_frame


def _start_arm_threads(arm_controller: SingleArmURController, stop_signal: threading.Event) -> list[threading.Thread]:
    threads = [
        threading.Thread(target=arm_controller.run_robot_thread, args=(stop_signal,), name="ur-servo"),
        threading.Thread(target=arm_controller.run_ik_thread, args=(stop_signal,), name="ur-ik"),
    ]
    if arm_controller.use_dh_gripper:
        threads.append(
            threading.Thread(
                target=arm_controller.run_dh_gripper_thread,
                args=(stop_signal,),
                name="dh-gripper",
            )
        )
    for t in threads:
        t.start()
    return threads


def main(
    reset: bool = False,
    visual: bool = False,
    robot_ip: str = DEFAULT_ROBOT_IP,
    controller_side: Literal["left", "right"] = "right",
    initial_joint_deg: tuple[float, float, float, float, float, float] | None = None,
    main_serial: str = DEFAULT_MAIN_SERIAL,
    pip_serial: str = DEFAULT_PIP_SERIAL,
    quality: Literal["low", "balanced", "high"] = "balanced",
    pip_scale: float = 0.28,
    pip_margin: int = 16,
    swap_button: Literal["A", "B", "X", "Y"] = "A",
    no_swap_on_a: bool = False,
    headset_ip: str = "192.168.3.198",
    pc_service_ip: str = "192.168.181.64",
    stream_port: int = 12345,
    connect_timeout: float = 120.0,
    preview: bool = False,
    layout: Literal["sbs", "screen"] = "screen",
    fps: int = 30,
    debug: bool = False,
    enable_recording: bool = True,
    record_fps: float = 10.0,
    record_save_dir: str = "logs/ur5e_bc/raw",
    record_image_format: Literal["jpeg", "raw"] = "jpeg",
    record_save_pip: bool = False,
    record_auto_reset: bool = True,
    policy_mode: bool = False,
) -> None:
    if controller_side == "left":
        manipulator_config = LEFT_CONTROLLER_MANIPULATOR_CONFIG
        default_initial_joint_deg = LEFT_INITIAL_JOINT_DEG
        if robot_ip == DEFAULT_ROBOT_IP:
            robot_ip = LEFT_ROBOT_IP
    else:
        manipulator_config = RIGHT_CONTROLLER_MANIPULATOR_CONFIG
        default_initial_joint_deg = RIGHT_INITIAL_JOINT_DEG

    initial_joint = (
        np.array(initial_joint_deg, dtype=float)
        if initial_joint_deg is not None
        else default_initial_joint_deg
    )

    if reset:
        _run_reset(robot_ip, initial_joint)
        return

    if not 0.1 <= pip_scale <= 0.5:
        print("--pip-scale must be between 0.1 and 0.5.", file=sys.stderr)
        raise SystemExit(1)

    preset = vr.QUALITY_PRESETS[quality]
    capture_width = preset["width"]
    capture_height = preset["height"]
    bitrate = preset["bitrate"]
    x264_preset = preset["preset"]
    stereo_sbs = True
    encode_width = capture_width * 2
    encode_height = capture_height

    vr.print_usb_pico_hint(pc_service_ip, headset_ip, stream_port)

    xr_client = XrClient()
    arm_controller: SingleArmURController | None = None
    bc_recorder: BCEpisodeRecorder | None = None
    stop_signal = threading.Event()
    arm_threads: list[threading.Thread] = []
    pipelines: list = []
    readers: list = []
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
            initial_joint_deg=initial_joint,
            visualize_placo=visual,
            use_dh_gripper=True,
            dh_gripper_port=DH_GRIPPER_PORT,
            dh_gripper_force=DH_GRIPPER_FORCE,
            dh_gripper_speed=DH_GRIPPER_SPEED,
            dh_gripper_slave_id=DH_GRIPPER_SLAVE_ID,
            dh_gripper_baudrate=DH_GRIPPER_BAUDRATE,
            policy_mode=policy_mode,
            bc_config=bc_config,
        )
        if policy_mode:
            print("policy_mode=True: attach a policy via arm_controller.set_policy() before teleop.")
        dh_gripper_connected = arm_controller.init_dh_gripper()
        gripper_hint = (
            f"Trigger ({controller_side}) controls DH gripper on {DH_GRIPPER_PORT} (proportional)."
            if dh_gripper_connected
            else "DH gripper not connected — arm-only teleop."
        )
        print(
            f"Single-arm teleop: robot_ip={robot_ip}, controller_side={controller_side}. "
            f"Hold grip to move the arm. {gripper_hint}"
        )
        arm_controller.calc_target_joint_position()

        main_pipeline, pip_pipeline, main_sn, pip_sn = vr.init_pip_camera_pipelines(
            capture_width, capture_height, fps, main_serial, pip_serial
        )
        pipelines = [main_pipeline, pip_pipeline]
        get_frame, readers, pip_source = vr.make_frame_source(
            main_pipeline,
            pip_pipeline,
            pip_scale,
            pip_margin,
            main_label=vr._device_label(main_sn),
            pip_label=vr._device_label(pip_sn),
        )
        print("Waiting for first frames from both cameras...")
        get_frame()
        print("Both cameras producing frames.")
        cameras_ready = True

        before_frame = None
        if pip_source is not None and not no_swap_on_a:
            before_frame = vr.init_swap_on_xr_client(pip_source, xr_client, button=swap_button)

        if debug:
            if pip_source is None:
                raise RuntimeError("--debug 需要双相机 PiP（请指定 --pip-serial）。")
            if pip_serial != DEFAULT_PIP_SERIAL:
                print(f"VR PiP debug overlay: aux camera {pip_serial}")
            else:
                print(f"VR PiP debug overlay: aux camera {pip_serial} (alignment vision)")
            get_frame = _wrap_get_frame_with_alignment_debug(pip_source, _make_alignment_renderer())

        layout_desc = (
            f"PiP main={main_sn} pip={pip_sn} scale={pip_scale} + "
            f"SBS + letterbox (TV-style panel per eye)"
        )
        print(
            f"Quality={quality}  capture {capture_width}x{capture_height} @ {fps}fps → "
            f"encode {encode_width}x{encode_height}  bitrate={bitrate // 1_000_000}Mbps  ({layout_desc})"
        )

        arm_threads = _start_arm_threads(arm_controller, stop_signal)

        if enable_recording:
            pip_reader = readers[1] if len(readers) > 1 else None
            bc_recorder = BCEpisodeRecorder(
                xr_client=xr_client,
                arm_controller=arm_controller,
                main_reader=readers[0],
                pip_reader=pip_reader,
                config=bc_config,
            )
            bc_recorder.start(stop_signal)

        codec = vr.create_h264_encoder(encode_width, encode_height, fps, bitrate, x264_preset)

        try:
            sock = vr.wait_for_headset_connection(
                headset_ip, stream_port, pc_service_ip, timeout_s=connect_timeout
            )
        except ConnectionError as exc:
            print(exc, file=sys.stderr)
            cleanup_cameras()
            cameras_ready = False
            raise SystemExit(1) from exc

        exit_code = vr.stream_to_vr(
            sock=sock,
            get_frame=get_frame,
            readers=readers,
            pipelines=pipelines,
            codec=codec,
            fps=fps,
            stereo_sbs=stereo_sbs,
            layout=layout,
            preview=preview,
            before_frame=before_frame,
            on_stop=None,
        )
        cameras_ready = False
        raise SystemExit(exit_code)

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
        print("Teleop and VR stream stopped.")


if __name__ == "__main__":
    tyro.cli(main)
