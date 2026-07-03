import time
from typing import TYPE_CHECKING

import numpy as np
import meshcat.transformations as tf

if TYPE_CHECKING:
    from xrobotoolkit_teleop.common.xr_client import XrClient

R_HEADSET_TO_WORLD = np.array(
    [
        [0, 0, -1],
        [-1, 0, 0],
        [0, 1, 0],
    ]
)


def calibrate_R_headset_world(xr_client: "XrClient", timeout_s: float = 10.0) -> np.ndarray:
    """Wait for live XR tracking and return SDK tracking frame → robot world rotation.

    Headset/controller poses share the tracking origin created when the VR app starts.
    Teleop maps that frame to the UR/placo world (Z up) via a fixed axis remap.
    MuJoCo and hardware must both call this after ``XrClient()`` so they use the live
    session frame instead of a synthetic origin.
    """
    deadline = time.monotonic() + timeout_s
    headset_pose = None
    while time.monotonic() < deadline:
        head = xr_client.get_pose_by_name("headset")
        pos = np.asarray(head[:3], dtype=np.float64)
        quat = np.array([head[6], head[3], head[4], head[5]], dtype=np.float64)
        if np.all(np.isfinite(pos)) and is_valid_quaternion(quat):
            headset_pose = head
            break
        time.sleep(0.05)

    if headset_pose is None:
        raise RuntimeError(
            "XR tracking unavailable. Start the PICO VR app and PC-Service, then retry teleop."
        )

    pos = headset_pose[:3]
    print(
        "XR 追踪原点已绑定（VR 应用启动时建立）— 头显位姿: "
        f"xyz=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
    )
    return R_HEADSET_TO_WORLD.copy()


def is_valid_quaternion(quat, tol=1e-6):
    if not isinstance(quat, (list, tuple, np.ndarray)) or len(quat) != 4:
        return False

    if not np.all(np.isfinite(quat)):
        return False

    magnitude = np.sqrt(np.sum(np.square(quat)))
    return abs(magnitude - 1.0) <= tol


def quaternion_to_angle_axis(quat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Converts a quaternion to an angle-axis representation.

    Args:
        quat: Quaternion in (w, x, y, z) format.
        eps: Tolerance for checking if the angle is close to zero.

    Returns:
        Angle-axis vector [ax*angle, ay*angle, az*angle].
    """
    q = np.array(quat, dtype=np.float64, copy=True)
    if q[0] < 0.0:
        q = -q

    w = q[0]
    vec_part = q[1:]

    angle = 2.0 * np.arccos(np.clip(w, -1.0, 1.0))
    if angle < eps:
        return np.zeros(3, dtype=np.float64)

    sin_half_angle = np.sin(angle / 2.0)
    if sin_half_angle < eps:
        return np.zeros(3, dtype=np.float64)

    axis = vec_part / sin_half_angle
    return axis * angle


def quat_diff_as_angle_axis(q1: np.ndarray, q2: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Calculates the rotation from q1 to q2 as an angle-axis vector.

    This computes DeltaQ such that q2 = DeltaQ * q1.
    The result is the angle-axis representation of DeltaQ.

    Args:
        q1: Source quaternion (w, x, y, z).
        q2: Target quaternion (w, x, y, z).
        eps: Tolerance for small angle calculations.

    Returns:
        Angle-axis vector [ax*angle, ay*angle, az*angle] representing DeltaQ.
    """
    if not (is_valid_quaternion(q1, tol=eps) and is_valid_quaternion(q2, tol=eps)):
        print("Warning: Invalid quaternion input to calculate_rotation_error_as_angle_axis.")

    q1_inv = tf.quaternion_inverse(q1)
    delta_q = tf.quaternion_multiply(q2, q1_inv)

    return quaternion_to_angle_axis(delta_q, eps)


def apply_delta_pose(
    source_pos: np.ndarray,
    source_rot: np.ndarray,
    delta_pos: np.ndarray,
    delta_rot: np.ndarray,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies delta pose transformation on source pose using NumPy.

    Args:
        source_pos: Position of source frame. Shape is (3,).
        source_rot: Quaternion orientation of source frame in (w, x, y, z). Shape is (4,).
        delta_pos: Cartesian position displacement. Shape is (3,).
        delta_rot: Orientation displacement in angle-axis format. Shape is (3,).
        eps: The tolerance to consider orientation displacement as zero. Defaults to 1.0e-6.

    Returns:
        A tuple containing the displaced position and orientation frames.
        target_pos: Shape is (3,).
        target_rot: Shape is (4,).
    """
    if not (
        isinstance(source_pos, np.ndarray)
        and source_pos.shape == (3,)
        and isinstance(source_rot, np.ndarray)
        and source_rot.shape == (4,)
        and isinstance(delta_pos, np.ndarray)
        and delta_pos.shape == (3,)
        and isinstance(delta_rot, np.ndarray)
        and delta_rot.shape == (3,)
    ):
        raise ValueError(
            "Inputs must be 1D NumPy arrays with shapes: "
            "source_pos (3,), source_rot (4,), delta_pos (3,), delta_rot (3,)."
        )

    # Calculate target position
    target_pos = source_pos + delta_pos

    # Calculate target rotation
    angle = np.linalg.norm(delta_rot)
    rot_delta_quat: np.ndarray
    if angle > eps:
        axis = delta_rot / angle
        rot_delta_quat = tf.quaternion_about_axis(angle, axis)
    else:
        rot_delta_quat = np.array([1.0, 0.0, 0.0, 0.0])

    target_rot = tf.quaternion_multiply(rot_delta_quat, source_rot)

    return target_pos, target_rot
