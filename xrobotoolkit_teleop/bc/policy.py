"""Policy deployment interface (training not implemented in V1)."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class BCPolicy(Protocol):
    """BC policy: obs dict → action vector [target_q(6), gripper_cmd(1)]."""

    def predict(self, obs: dict) -> np.ndarray:
        ...


class NullPolicy:
    """Placeholder policy that holds the current joint configuration."""

    def predict(self, obs: dict) -> np.ndarray:
        state = np.asarray(obs["observation.state"], dtype=np.float64).reshape(7)
        qpos = state[:6]
        gripper = state[6]
        return np.concatenate([qpos, [gripper]]).astype(np.float64)


def apply_policy_action(
    action: np.ndarray,
    current_q: np.ndarray,
    max_delta_q: float,
    joint_limits_lower: np.ndarray,
    joint_limits_upper: np.ndarray,
) -> tuple[np.ndarray, float]:
    """
    Parse policy output into filtered target_q and gripper_cmd.

    action layout: [target_q(6), gripper_cmd(1)]
    gripper_cmd: 0=open, 1=close
    """
    from xrobotoolkit_teleop.bc.safety import safety_filter_target_q

    action = np.asarray(action, dtype=np.float64).reshape(7)
    raw_target_q = action[:6]
    gripper_cmd = float(np.clip(action[6], 0.0, 1.0))
    target_q = safety_filter_target_q(
        raw_target_q,
        current_q,
        max_delta_q=max_delta_q,
        joint_limits_lower=joint_limits_lower,
        joint_limits_upper=joint_limits_upper,
    )
    return target_q, gripper_cmd
