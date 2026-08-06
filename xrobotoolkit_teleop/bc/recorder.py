"""Raw episode recorder for UR5e BC (sample + encode/save threads)."""

from __future__ import annotations

import pickle
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from xrobotoolkit_teleop.bc.config import BCConfig
from xrobotoolkit_teleop.utils.image_utils import compress_image_to_jpg

if TYPE_CHECKING:
    from xrobotoolkit_teleop.common.xr_client import XrClient
    from xrobotoolkit_teleop.hardware.single_arm_ur_controller import SingleArmURController

_SENTINEL_FLUSH = object()
_SENTINEL_DISCARD = object()


@dataclass
class _RawFrame:
    """采样线程产出；图像为 BGR numpy，尚未 JPEG 编码。"""

    timestamp: float
    frame_index: int
    episode_index: int
    image_main_bgr: np.ndarray
    image_pip_bgr: np.ndarray | None
    qpos: np.ndarray
    target_q: np.ndarray
    gripper_state: float
    gripper_cmd: float
    control_active: bool


class BCEpisodeRecorder:
    """
    双线程录制：采样线程 10Hz 取数入队；编码线程 JPEG + 累积帧，停止时写盘。

    手柄按键（VR）：
      X — 开始 / 停止当前局录制
      Y — 丢弃本局（不保存，并回 Home）

    按 X 停止后自动保存（无交互确认）。Episode 原子写入
    ``{save_dir}/episode_{index:05d}.pkl``（绝对路径会打印到终端）。

    BC conventions (see bc/config.py):
      observation.state = [qpos(6), gripper_state(1)]   — 0=open, 1=close
      action            = [target_q(6), gripper_cmd(1)] — 0=open, 1=close
    """

    def __init__(
        self,
        xr_client: XrClient,
        arm_controller: SingleArmURController,
        main_reader: Any,
        config: BCConfig | None = None,
        pip_reader: Any | None = None,
    ):
        self.xr_client = xr_client
        self.arm_controller = arm_controller
        self.main_reader = main_reader
        self.pip_reader = pip_reader
        self.config = config or BCConfig()

        self._stop_event = threading.Event()
        self._sample_thread: threading.Thread | None = None
        self._encode_thread: threading.Thread | None = None
        self._lock = threading.Lock()

        qmax = self.config.encode_queue_maxsize
        self._frame_queue: queue.Queue[_RawFrame | object] = queue.Queue(maxsize=0 if qmax <= 0 else qmax)
        self._encode_done = threading.Event()
        self._encode_done.set()

        self._is_recording = False
        self._episode_index = self._next_episode_index()
        self._frames: list[dict] = []
        self._frame_index = 0
        self._episode_start_time: float | None = None
        self._session_start_time = time.time()

        self._prev_x = False
        self._prev_y = False

    def _next_episode_index(self) -> int:
        existing = sorted(self.config.save_dir.glob("episode_*.pkl"))
        if not existing:
            return 0
        last = existing[-1].stem
        try:
            return int(last.split("_")[1]) + 1
        except (IndexError, ValueError):
            return len(existing)

    def start(self, stop_event: threading.Event) -> None:
        """启动采样线程与编码线程。"""
        self._stop_event = stop_event
        self._encode_thread = threading.Thread(target=self._encode_loop, name="bc-encode", daemon=True)
        self._sample_thread = threading.Thread(target=self._sample_loop, name="bc-sample", daemon=True)
        self._encode_thread.start()
        self._sample_thread.start()
        print(
            f"BC 录制就绪 @ {self.config.record_fps}Hz（采样+编码双线程）\n"
            f"  保存目录（绝对路径）：{self.config.save_dir}\n"
            "  X：开始/停止并自动保存  |  Y：丢弃本局  |  A：主/辅画面切换（不变）"
        )

    def stop(self) -> None:
        if self._is_recording:
            print("正在退出：丢弃未完成的录制。")
            self._discard_episode(wait_encode=True)
        for t in (self._sample_thread, self._encode_thread):
            if t is not None:
                t.join(timeout=2.0)

    def _sample_loop(self) -> None:
        interval = 1.0 / self.config.record_fps
        next_tick = time.perf_counter()
        while not self._stop_event.is_set():
            self._poll_buttons()
            if self._is_recording:
                now = time.perf_counter()
                if now >= next_tick:
                    # 单帧异常不允许杀死采样线程：否则录制会静默中断（按键也失效）。
                    try:
                        self._enqueue_raw_sample()
                    except Exception as exc:
                        print(f"BC 采样异常（跳过本帧，录制继续）：{exc}", file=sys.stderr)
                    next_tick += interval
                    # 严重落后时重新对齐，避免 CPU 饥饿后连续 burst 两帧。
                    if time.perf_counter() - next_tick > interval:
                        next_tick = time.perf_counter() + interval
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        print("BC 采样线程已停止。")

    def _encode_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                item = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is _SENTINEL_DISCARD:
                with self._lock:
                    self._frames = []
                    self._frame_index = 0
                self._encode_done.set()
                continue

            if item is _SENTINEL_FLUSH:
                self._encode_done.set()
                continue

            assert isinstance(item, _RawFrame)
            try:
                encoded = self._encode_raw_frame(item)
            except RuntimeError as exc:
                print(f"BC 编码：帧 {item.frame_index} 失败：{exc}", file=sys.stderr)
                continue

            with self._lock:
                self._frames.append(encoded)
        print("BC 编码线程已停止。")

    def _poll_buttons(self) -> None:
        x_pressed = self.xr_client.get_button_state_by_name("X")
        y_pressed = self.xr_client.get_button_state_by_name("Y")

        if x_pressed and not self._prev_x:
            if self._is_recording:
                self._stop_and_finalize_episode()
            else:
                self._start_episode()

        if y_pressed and not self._prev_y and self._is_recording:
            print("--- 已丢弃本局录制（Y）---")
            self._discard_episode(wait_encode=True)
            if self.config.auto_reset_after_episode:
                self._schedule_reset_to_home(after_discard=True)

        self._prev_x = x_pressed
        self._prev_y = y_pressed

    def _start_episode(self) -> None:
        self._wait_encode_idle()
        with self._lock:
            self._frames = []
            self._frame_index = 0
            self._episode_start_time = time.time()
            self._is_recording = True
        print(f"--- 开始录制第 {self._episode_index} 局（再按 X 停止）---")

    def _discard_episode(self, wait_encode: bool = False) -> None:
        self._is_recording = False
        self._encode_done.clear()
        self._frame_queue.put(_SENTINEL_DISCARD)
        if wait_encode:
            self._wait_encode_idle(timeout=10.0)
        with self._lock:
            self._episode_start_time = None

    def _wait_encode_idle(self, timeout: float = 30.0) -> None:
        if not self._encode_done.wait(timeout=timeout):
            print("警告：等待编码队列排空超时。", file=sys.stderr)

    def _stop_and_finalize_episode(self) -> None:
        self._is_recording = False
        self._encode_done.clear()
        self._frame_queue.put(_SENTINEL_FLUSH)
        self._wait_encode_idle()

        with self._lock:
            frames_copy = list(self._frames)
            ep_index = self._episode_index
            start_t = self._episode_start_time
        end_t = time.time()

        if not frames_copy:
            print("本局无数据，未保存。")
            with self._lock:
                self._frames = []
                self._frame_index = 0
                self._episode_start_time = None
            return

        self._save_episode(
            frames_copy,
            ep_index,
            start_t,
            end_t,
            success=self.config.default_episode_success,
        )
        self._episode_index = ep_index + 1
        if self.config.auto_reset_after_episode:
            self._schedule_reset_to_home()
        with self._lock:
            self._frames = []
            self._frame_index = 0
            self._episode_start_time = None

    def _schedule_reset_to_home(self, *, after_discard: bool = False) -> None:
        """后台回 Home，不阻塞采样线程按键检测。"""

        def _run() -> None:
            try:
                if after_discard:
                    print("本局已丢弃，正在回 Home...")
                else:
                    print("录制已保存，正在回 Home...")
                self.arm_controller.reset_to_home()
                print("已回到 Home，可摆放积木并按 X 开始下一局。")
            except Exception as exc:
                print(f"回 Home 失败：{exc}", file=sys.stderr)

        threading.Thread(target=_run, name="bc-reset", daemon=True).start()

    def _prepare_record_image(self, image_bgr: np.ndarray) -> np.ndarray:
        """get_latest 已返回 copy；可选缩小以减轻编码/CPU 压力。"""
        scale = self.config.record_image_scale
        if scale >= 0.999:
            return image_bgr
        if scale <= 0.1:
            raise ValueError("record_image_scale must be in (0.1, 1.0].")
        h, w = image_bgr.shape[:2]
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _enqueue_raw_sample(self) -> None:
        try:
            image_main = self.main_reader.get_latest(timeout_s=0.2)
        except RuntimeError as exc:
            print(f"BC 采样：主相机丢帧：{exc}", file=sys.stderr)
            return

        image_pip = None
        if self.config.save_optional_pip_image and self.pip_reader is not None:
            try:
                image_pip = self._prepare_record_image(self.pip_reader.get_latest(timeout_s=0.1))
            except RuntimeError:
                pass

        sample = self.arm_controller.sample_bc_frame()
        with self._lock:
            frame_index = self._frame_index
            episode_index = self._episode_index
            self._frame_index += 1

        raw = _RawFrame(
            timestamp=time.time() - self._session_start_time,
            frame_index=frame_index,
            episode_index=episode_index,
            image_main_bgr=self._prepare_record_image(image_main),
            image_pip_bgr=image_pip,
            qpos=sample["qpos"],
            target_q=sample["target_q"],
            gripper_state=sample["gripper_state"],
            gripper_cmd=sample["gripper_cmd"],
            control_active=sample["control_active"],
        )

        try:
            self._frame_queue.put_nowait(raw)
        except queue.Full:
            print(
                f"BC 采样：编码队列已满（>{self.config.encode_queue_maxsize}），丢弃帧 {frame_index}",
                file=sys.stderr,
            )

    def _encode_image(self, image_bgr: np.ndarray) -> np.ndarray | bytes:
        if self.config.image_format == "raw":
            return image_bgr.copy()
        jpeg = compress_image_to_jpg(image_bgr, quality=self.config.jpeg_quality)
        if jpeg is None:
            raise RuntimeError("JPEG 编码失败。")
        return jpeg

    def _encode_raw_frame(self, raw: _RawFrame) -> dict:
        frame: dict = {
            "timestamp": raw.timestamp,
            "frame_index": raw.frame_index,
            "episode_index": raw.episode_index,
            "image_main": self._encode_image(raw.image_main_bgr),
            "qpos": raw.qpos,
            "target_q": raw.target_q,
            "gripper_state": raw.gripper_state,
            "gripper_cmd": raw.gripper_cmd,
            "control_active": raw.control_active,
        }
        if raw.image_pip_bgr is not None:
            frame["image_pip"] = self._encode_image(raw.image_pip_bgr)
        return frame

    def _save_episode(
        self,
        frames: list[dict],
        episode_index: int,
        start_time: float | None,
        end_time: float,
        success: bool,
    ) -> None:
        metadata = {
            "episode_index": episode_index,
            "success": success,
            "task": self.config.task,
            "fps": self.config.record_fps,
            "num_frames": len(frames),
            "start_time": start_time,
            "end_time": end_time,
            "duration": (end_time - start_time) if start_time else 0.0,
            "image_format": self.config.image_format,
            "schema_version": self.config.schema_version,
            "observation_state_layout": "qpos(6) + gripper_state(1), 0=open 1=close",
            "action_layout": "target_q(6) + gripper_cmd(1), 0=open 1=close",
        }
        payload = {"metadata": metadata, "frames": frames}
        filename = f"episode_{episode_index:05d}.pkl"
        final_path = self.config.save_dir / filename
        tmp_path = final_path.with_suffix(".pkl.tmp")

        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_path.replace(final_path)
            abs_path = final_path.resolve()
            if not abs_path.is_file():
                raise OSError(f"写入后文件不存在：{abs_path}")
            size_mb = abs_path.stat().st_size / (1024 * 1024)
            success_label = "成功" if success else "失败"
            print(
                f"已保存第 {episode_index} 局（{len(frames)} 帧，{success_label}，{size_mb:.1f} MB）\n"
                f"  → {abs_path}"
            )
        except OSError as exc:
            print(f"保存失败：{exc}", file=sys.stderr)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
