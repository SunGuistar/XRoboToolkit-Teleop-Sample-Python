"""Teleoperation wrapper for DH-Robotics PGC-300-60 (proportional VR trigger control)."""

from __future__ import annotations

import threading
import time

from xrobotoolkit_teleop.hardware.interface.pgc300_60_driver import (
    DEFAULT_BAUDRATE,
    DEFAULT_SLAVE_ID,
    PGC30060Gripper,
)
from xrobotoolkit_teleop.utils.parallel_gripper_utils import calc_parallel_gripper_position


class DHPGCGripperTeleopInterface:
    """
    Rate-limited proportional gripper control for VR teleop.

    VR trigger: released (0) -> open, pressed (1) -> closed (interpolated in register space).
    """

    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        slave_id: int = DEFAULT_SLAVE_ID,
        force: int = 50,
        speed: int = 50,
        min_command_interval: float = 0.08,
        position_deadband: int = 25,
        standard_position_mapping: bool = False,
    ):
        self._lock = threading.Lock()
        self._min_interval = min_command_interval
        self._deadband = position_deadband
        self._last_target: int | None = None
        self._last_send_time = 0.0
        self._write_failures = 0

        self._gripper = PGC30060Gripper.from_serial(
            port=port,
            baudrate=baudrate,
            slave_id=slave_id,
            standard_position_mapping=standard_position_mapping,
        )
        print(f"DH PGC-300-60: connecting on {port} (slave={slave_id}, {baudrate} baud)...")
        self._gripper.initialize()
        self._gripper.apply_motion_config(force, speed)
        self._register_open = self._gripper.register_value_for_open()
        self._register_closed = self._gripper.register_value_for_closed()
        # Start teleop with fingers open (init may home to closed depending on flash config).
        self._gripper.set_position(self._register_open, wait=False, verify=False)
        self._last_target = self._register_open
        print(
            f"DH PGC-300-60 ready (open register={self._register_open}, "
            f"closed register={self._register_closed})."
        )

    def target_position_from_trigger(self, trigger_val: float) -> int:
        trigger_val = max(0.0, min(1.0, float(trigger_val)))
        # Match Robotiq teleop: trigger released = open, trigger pressed = closed.
        return int(
            round(
                calc_parallel_gripper_position(
                    self._register_open,
                    self._register_closed,
                    trigger_val,
                )
            )
        )

    def get_normalized_state(self) -> float:
        """Last commanded register position mapped to [0, 1]. 0=open, 1=close."""
        from xrobotoolkit_teleop.bc.gripper_norm import register_to_normalized

        with self._lock:
            if self._last_target is None:
                return 0.0
            return register_to_normalized(self._last_target, self._register_open, self._register_closed)

    def open_gripper(self) -> None:
        """Open fingers (0=open in BC convention)."""
        with self._lock:
            self._gripper.set_position(self._register_open, wait=False, verify=False)
            self._last_target = self._register_open
            self._last_send_time = time.monotonic()

    def update_from_trigger(self, trigger_val: float) -> None:
        """Send position command when trigger or rate limit warrants an update."""
        target = self.target_position_from_trigger(trigger_val)
        now = time.monotonic()

        with self._lock:
            if self._last_target is not None:
                if abs(target - self._last_target) < self._deadband:
                    if now - self._last_send_time < self._min_interval:
                        return
            if now - self._last_send_time < self._min_interval:
                return

            try:
                self._gripper.set_position(target, wait=False, verify=False)
                self._last_target = target
                self._last_send_time = now
                self._write_failures = 0
            except Exception as exc:
                self._write_failures += 1
                if self._write_failures <= 3 or self._write_failures % 50 == 0:
                    print(f"WARNING: DH gripper write failed ({self._write_failures}x): {exc}")

    def disconnect(self) -> None:
        with self._lock:
            self._gripper.disconnect()
