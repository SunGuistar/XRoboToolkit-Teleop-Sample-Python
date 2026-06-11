"""Safety filter for policy outputs before servoJ."""

from __future__ import annotations

import numpy as np

from xrobotoolkit_teleop.bc.config import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER


def safety_filter_target_q(
    target_q: np.ndarray,
    current_q: np.ndarray,
    max_delta_q: float = 0.05,
    joint_limits_lower: np.ndarray | None = None,
    joint_limits_upper: np.ndarray | None = None,
) -> np.ndarray:
    """
    Clamp policy joint targets before sending to servoJ.

    Steps:
      1. Reject NaN/Inf (hold current_q)
      2. Per-joint delta limit vs current_q
      3. Clip to joint limits
    """
    target_q = np.asarray(target_q, dtype=np.float64).reshape(6)
    current_q = np.asarray(current_q, dtype=np.float64).reshape(6)
    lower = joint_limits_lower if joint_limits_lower is not None else JOINT_LIMITS_LOWER
    upper = joint_limits_upper if joint_limits_upper is not None else JOINT_LIMITS_UPPER

    if not np.all(np.isfinite(target_q)):
        return current_q.copy()

    delta = np.clip(target_q - current_q, -max_delta_q, max_delta_q)
    filtered = current_q + delta
    return np.clip(filtered, lower, upper).astype(np.float64)
