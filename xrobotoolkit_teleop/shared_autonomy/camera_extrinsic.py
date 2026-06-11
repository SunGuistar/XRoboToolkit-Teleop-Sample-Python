"""固定辅机外参：相机坐标系 -> 机械臂基座坐标系。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def extrinsic_path_for_serial(serial: str) -> Path:
    return Path(ASSET_PATH) / "calibration" / f"pip_{serial}_extrinsic.json"


def _rpy_deg_to_matrix(rpy_deg: tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(rpy_deg)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


@dataclass
class CameraExtrinsic:
    camera_serial: str = ""
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    calibration_note: str = ""

    def rotation_matrix(self) -> np.ndarray:
        return _rpy_deg_to_matrix(self.rpy_deg)

    def camera_to_base(self, p_cam: np.ndarray) -> np.ndarray:
        r = self.rotation_matrix()
        t = np.array(self.translation_m, dtype=np.float64)
        p = np.asarray(p_cam, dtype=np.float64).reshape(3)
        return r @ p + t

    def base_to_camera(self, p_base: np.ndarray) -> np.ndarray:
        r = self.rotation_matrix()
        t = np.array(self.translation_m, dtype=np.float64)
        p = np.asarray(p_base, dtype=np.float64).reshape(3)
        return r.T @ (p - t)

    @classmethod
    def from_one_point(
        cls,
        *,
        camera_serial: str,
        tcp_xyz_m: np.ndarray,
        target_cam_xyz_m: np.ndarray,
        rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
        note: str = "",
    ) -> CameraExtrinsic:
        """单点标定：假设 TCP 位于目标点基座坐标，求平移 t = P_tcp - R @ P_cam。"""
        r = _rpy_deg_to_matrix(rpy_deg)
        p_tcp = np.asarray(tcp_xyz_m, dtype=np.float64).reshape(3)
        p_cam = np.asarray(target_cam_xyz_m, dtype=np.float64).reshape(3)
        t = p_tcp - r @ p_cam
        return cls(
            camera_serial=camera_serial,
            translation_m=(float(t[0]), float(t[1]), float(t[2])),
            rpy_deg=rpy_deg,
            calibration_note=note,
        )

    def save(self, path: Path | None = None) -> Path:
        out = path or extrinsic_path_for_serial(self.camera_serial)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: Path) -> CameraExtrinsic:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            camera_serial=str(data.get("camera_serial", "")),
            translation_m=tuple(data.get("translation_m", (0.0, 0.0, 0.0))),
            rpy_deg=tuple(data.get("rpy_deg", (0.0, 0.0, 0.0))),
            calibration_note=str(data.get("calibration_note", "")),
        )
