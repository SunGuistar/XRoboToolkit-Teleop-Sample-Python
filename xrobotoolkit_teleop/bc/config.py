"""Central configuration for UR5e BC data collection and deployment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Gripper convention (used everywhere in BC pipeline):
#   0.0 = fully open
#   1.0 = fully closed

DEFAULT_TASK = "pick block and place to fixed area"

# UR5e joint limits in radians (from UR datasheet, degrees converted).
JOINT_LIMITS_LOWER = np.deg2rad(np.array([-363.0, -130.0, -50.0, -190.0, -363.0, -363.0], dtype=np.float64))
JOINT_LIMITS_UPPER = np.deg2rad(np.array([363.0, 50.0, 180.0, 0.0, 363.0, 363.0], dtype=np.float64))


@dataclass
class BCConfig:
    """Parameters for raw episode recording and downstream BC deployment."""

    # Recording
    record_fps: float = 10.0
    save_dir: Path = field(default_factory=lambda: Path("logs/ur5e_bc/raw"))
    image_format: str = "jpeg"  # "jpeg" or "raw"
    jpeg_quality: int = 85
    # 录制前缩小主相机图像（1.0=原图）。缩小可减轻 copy/JPEG CPU，缓解采样线程饥饿。
    record_image_scale: float = 1.0
    encode_queue_maxsize: int = 128  # 0=无限；满时采样线程丢帧并告警
    auto_reset_after_episode: bool = True  # 保存 episode 后自动 moveJ 回 Home
    default_episode_success: bool = True  # 自动保存时 metadata.success 默认值
    save_optional_pip_image: bool = False
    task: str = DEFAULT_TASK
    schema_version: int = 1

    # Deployment safety (policy → servoJ)
    max_delta_q: float = 0.05  # rad per control step
    joint_limits_lower: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_LOWER.copy())
    joint_limits_upper: np.ndarray = field(default_factory=lambda: JOINT_LIMITS_UPPER.copy())

    # LeRobot conversion defaults
    lerobot_repo_id: str = "local/ur5e_block_pick"
    lerobot_robot_type: str = "ur5e"
    camera_key: str = "observation.images.main"
    state_dim: int = 7  # qpos(6) + gripper_state(1)
    action_dim: int = 7  # target_q(6) + gripper_cmd(1)

    def __post_init__(self) -> None:
        self.save_dir = Path(self.save_dir).resolve()
        self.save_dir.mkdir(parents=True, exist_ok=True)
