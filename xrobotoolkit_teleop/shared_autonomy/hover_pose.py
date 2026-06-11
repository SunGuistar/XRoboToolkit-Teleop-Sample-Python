"""由目标基座坐标构造 MoveL hover 位姿。"""

from __future__ import annotations

import numpy as np

from xrobotoolkit_teleop.shared_autonomy.camera_extrinsic import CameraExtrinsic
from xrobotoolkit_teleop.shared_autonomy.depth_sample import DepthPatchSample


def target_cam_xyz_from_sample(sample: DepthPatchSample) -> np.ndarray | None:
    if not sample.valid:
        return None
    return np.array([sample.camera_x, sample.camera_y, sample.camera_z], dtype=np.float64)


def median_cam_xyz(samples: list[DepthPatchSample]) -> np.ndarray | None:
    valid = [s for s in samples if s.valid]
    if not valid:
        return None
    xyz = np.array([[s.camera_x, s.camera_y, s.camera_z] for s in valid], dtype=np.float64)
    return np.median(xyz, axis=0)


def target_base_xyz_from_sample(sample: DepthPatchSample, extrinsic: CameraExtrinsic) -> np.ndarray | None:
    p_cam = target_cam_xyz_from_sample(sample)
    if p_cam is None:
        return None
    return extrinsic.camera_to_base(p_cam)


def target_base_xyz_from_samples(samples: list[DepthPatchSample], extrinsic: CameraExtrinsic) -> np.ndarray | None:
    p_cam = median_cam_xyz(samples)
    if p_cam is None:
        return None
    return extrinsic.camera_to_base(p_cam)


def build_hover_tcp_pose(
    target_base_xyz: np.ndarray,
    current_tcp_pose: np.ndarray,
    approach_height_m: float,
) -> np.ndarray:
    """在基座系目标点正上方 approach_height_m 处 hover，保持当前 TCP 姿态。"""
    pose = np.array(current_tcp_pose, dtype=np.float64).copy()
    pose[0] = float(target_base_xyz[0])
    pose[1] = float(target_base_xyz[1])
    pose[2] = float(target_base_xyz[2]) + approach_height_m
    return pose
