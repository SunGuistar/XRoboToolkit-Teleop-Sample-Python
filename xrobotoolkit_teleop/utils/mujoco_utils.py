import mujoco
import numpy as np
import placo


def calc_mujoco_qpos_from_placo_q(
    mujoco_model: mujoco.MjModel,
    placo_robot: placo.RobotWrapper,
    placo_q: np.ndarray,
    floating_base: bool = False,
) -> np.ndarray:
    """
    Convert Placo joint configuration to MuJoCo qpos.

    Args:
        mujoco_model: The MuJoCo model.
        placo_robot: The Placo robot wrapper.
        placo_q: The joint configuration in Placo format.

    Returns:
        np.ndarray: The corresponding MuJoCo qpos.
    """
    mujoco_qpos = np.zeros(mujoco_model.nq)
    if floating_base:
        mujoco_qpos[:7] = placo_q[:7]  # Assuming the first 7 are for floating base

    placo_joint_names = [
        name for name in placo_robot.model.names if name != "root_joint" and name != "universe"
    ]

    # Start index for actuated joints in placo_q, depends on floating_base
    placo_q_offset = 7

    for i, placo_joint_name in enumerate(placo_joint_names):
        # Placo q for actuated joints starts after the floating base if it exists
        placo_joint_value = placo_q[placo_q_offset + i]

        success = set_mujoco_joint_pos_by_name(
            mujoco_model,
            mujoco_qpos,
            placo_joint_name,
            placo_joint_value,
        )
        if not success:
            raise ValueError(f"Joint '{placo_joint_name}' not found in MuJoCo model.")

    return mujoco_qpos


def calc_placo_q_from_mujoco_qpos(
    mujoco_model: mujoco.MjModel,
    mujoco_qpos: np.ndarray,
    floating_base: bool = False,
) -> np.ndarray:
    """
    Convert MuJoCo qpos to Placo joint configuration.

    Args:
        mujoco_model: The MuJoCo model.
        mujoco_qpos: The joint configuration in MuJoCo format.

    Returns:
        np.ndarray: The corresponding Placo joint configuration.
    """
    placo_q = np.zeros(len(mujoco_model.jnt_names) + 7)

    if floating_base:
        placo_q[:7] = mujoco_qpos[:7]  # Assuming the first 7 are for floating base
    else:
        placo_q[:7] = np.array([0, 0, 0, 1, 0, 0, 0])

    for i, joint_name in enumerate(mujoco_model.jnt_names):
        if joint_name == "root_joint" or joint_name == "universe":
            continue
        mujoco_joint_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if mujoco_joint_id != -1:
            qpos_addr = mujoco_model.jnt_qposadr[mujoco_joint_id]
            if qpos_addr < len(mujoco_qpos):
                placo_q[i + (7 if floating_base else 0)] = mujoco_qpos[qpos_addr]

    return placo_q


def set_mujoco_joint_pos_by_name(
    mujoco_model: mujoco.MjModel,
    qpos: np.ndarray,
    joint_name: str,
    joint_pos: float,
) -> None:
    """
    Set the position of a MuJoCo joint by its name.

    Args:
        mujoco_model: The MuJoCo model.
        mujoco_data: The MuJoCo data.
        joint_name: The name of the joint to set.
        joint_pos: The desired position for the joint.
    """
    joint_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id == -1:
        return False  # Joint not found

    qpos_addr = mujoco_model.jnt_qposadr[joint_id]
    qpos[qpos_addr] = joint_pos
    return True  # Joint position set successfully


def calc_mujoco_ctrl_from_qpos(mujoco_model: mujoco.MjModel, mujoco_qpos: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo qpos to control signals.

    Args:
        mujoco_model: The MuJoCo model.
        mujoco_qpos: The joint configuration in MuJoCo format.

    Returns:
        np.ndarray: The corresponding control signals.
    """
    mujoco_ctrl = np.zeros(mujoco_model.nu)
    for i in range(mujoco_model.nu):
        # Get the ID of the transmission target (e.g., joint ID) for actuator i
        target_id = mujoco_model.actuator_trnid[i, 0]

        # Assuming the actuator targets a joint (common case for robot arms)
        # and that the joint is 1-DOF.
        qpos_addr = mujoco_model.jnt_qposadr[target_id]

        # Assign the corresponding qpos value to the control signal
        mujoco_ctrl[i] = mujoco_qpos[qpos_addr]

    return mujoco_ctrl
