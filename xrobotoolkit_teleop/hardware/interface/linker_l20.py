"""Linker Hand L20 dexterous hand interface (CAN via LinkerHand Python SDK)."""

from __future__ import annotations

import os
import sys
import time
from typing import Literal

L20_NUM_JOINTS = 20

# Joint order matches LinkerHand SDK / GUI constants for L20.
L20_JOINT_NAMES = (
    "thumb_mcp",
    "index_mcp",
    "middle_mcp",
    "ring_mcp",
    "pinky_mcp",
    "thumb_abduction",
    "index_abduction",
    "middle_abduction",
    "ring_abduction",
    "pinky_abduction",
    "thumb_yaw",
    "reserved_11",
    "reserved_12",
    "reserved_13",
    "reserved_14",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
)

# When curling index (mcp/tip -> 0), abduction must leave the "open" value (~10) or mcp sticks ~17.
CURL_INDEX_ABDUCTION = 125
CURL_GESTURE_NAMES = frozenset({"fist", "thumbs_up"})
# Per-finger motor current [thumb, index, middle, ring, pinky]; boost index slightly when curling.
CURL_FINGER_CURRENT = [180, 230, 200, 200, 200]

# Preset poses for L20 (values 0-255).
L20_PRESET_POSES: dict[str, list[int]] = {
    "open": [255, 255, 255, 255, 255, 255, 10, 100, 180, 240, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
    "fist": [40, 0, 0, 0, 0, 131, CURL_INDEX_ABDUCTION, 100, 180, 240, 19, 255, 255, 255, 255, 135, 0, 0, 0, 0],
    "ok": [191, 95, 255, 255, 255, 136, 107, 100, 180, 240, 72, 255, 255, 255, 255, 116, 99, 255, 255, 255],
    "thumbs_up": [255, 0, 0, 0, 0, 127, CURL_INDEX_ABDUCTION, 100, 180, 240, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0],
}

DEFAULT_SPEED = [120, 200, 200, 200, 200]
DEFAULT_TORQUE = [180, 180, 180, 180, 180]

_SDK_INSTALL_HINT = (
    "LinkerHand Python SDK not found. Install it first:\n"
    "  git clone https://github.com/linker-bot/linkerhand-python-sdk.git\n"
    "  cd linkerhand-python-sdk && pip install -r requirements.txt\n"
    "Then run with --sdk-path /path/to/linkerhand-python-sdk\n"
    "or set LINKERHAND_SDK_PATH to the SDK repo root."
)


def _default_sdk_candidates() -> list[str]:
    home = os.path.expanduser("~")
    return [
        os.path.join(home, "linkerhand-python-sdk"),
        os.path.join(home, "linker_hand_python_sdk"),
        "/home/arctic/linkerhand-python-sdk",
    ]


def ensure_linkerhand_sdk(sdk_path: str | None = None) -> None:
    """Add LinkerHand SDK repo root to ``sys.path`` if needed."""
    if "LinkerHand" in sys.modules:
        return

    candidates: list[str] = []
    if sdk_path:
        candidates.append(os.path.abspath(sdk_path))
    env_path = os.environ.get("LINKERHAND_SDK_PATH")
    if env_path:
        candidates.append(os.path.abspath(env_path))
    candidates.extend(_default_sdk_candidates())

    seen: set[str] = set()
    for path in candidates:
        if path in seen or not os.path.isdir(path):
            continue
        seen.add(path)
        if path not in sys.path:
            sys.path.insert(0, path)

    try:
        import LinkerHand  # noqa: F401
    except ImportError as exc:
        raise ImportError(_SDK_INSTALL_HINT) from exc


class LinkerL20Interface:
    """Thin wrapper around ``LinkerHandApi`` for the L20 dexterous hand."""

    def __init__(
        self,
        hand_type: Literal["left", "right"] = "right",
        can: str = "can0",
        sdk_path: str | None = None,
        speed: list[int] | None = None,
        torque: list[int] | None = None,
    ):
        ensure_linkerhand_sdk(sdk_path)
        from LinkerHand.linker_hand_api import LinkerHandApi

        self.hand_type = hand_type
        self.can = can
        self._api = LinkerHandApi(hand_type=hand_type, hand_joint="L20", can=can)
        self.set_speed(speed or DEFAULT_SPEED)
        self.set_torque(torque or DEFAULT_TORQUE)

    def set_speed(self, speed: list[int]) -> None:
        self._api.set_speed(speed=speed)

    def set_torque(self, torque: list[int]) -> None:
        # L20 firmware may not support set_torque; ignore SDK warning.
        self._api.set_torque(torque=torque)

    def move(self, pose: list[int]) -> None:
        if len(pose) != L20_NUM_JOINTS:
            raise ValueError(f"L20 pose must have {L20_NUM_JOINTS} values, got {len(pose)}")
        if any(not 0 <= int(v) <= 255 for v in pose):
            raise ValueError("All pose values must be in [0, 255]")
        self._api.finger_move(pose=pose)

    def move_preset(self, name: str, *, two_phase_curl: bool = True) -> None:
        key = name.lower()
        if key not in L20_PRESET_POSES:
            known = ", ".join(sorted(L20_PRESET_POSES))
            raise ValueError(f"Unknown preset '{name}'. Available: {known}")
        pose = list(L20_PRESET_POSES[key])
        if key in CURL_GESTURE_NAMES and two_phase_curl:
            self._apply_curl_gesture(pose)
        else:
            self.move(pose)

    def _apply_curl_gesture(self, pose: list[int]) -> None:
        """Pre-set index abduction, then curl (fixes loose index when abduction was still at 'open' value)."""
        pose[6] = CURL_INDEX_ABDUCTION
        try:
            self._api.set_current(CURL_FINGER_CURRENT)
        except Exception:
            pass

        prep = list(pose)
        prep[1] = 255
        prep[16] = 255
        self.move(prep)
        time.sleep(0.5)
        self.move(pose)
        time.sleep(0.4)

    def get_state(self) -> list[float]:
        return self._api.get_state()

    def get_version(self):
        return self._api.get_embedded_version()

    def get_serial_number(self) -> str:
        return self._api.get_serial_number()

    def get_temperature(self) -> list[float]:
        return self._api.get_temperature()

    def get_fault(self) -> list[float]:
        return self._api.get_fault()

    def get_torque(self) -> list[float]:
        return self._api.get_torque()

    def get_force(self):
        return self._api.get_force()

    def clear_faults(self) -> None:
        self._api.clear_faults()

    def close(self) -> None:
        # SDK close_can() has a known bug (undefined `modbus`); keep can0 up for reuse.
        pass
