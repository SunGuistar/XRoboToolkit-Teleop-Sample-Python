"""
Single UR5e hardware teleoperation using a PICO VR controller.

Does not require a second UR arm or Dynamixel head motors (those are optional lab hardware
in teleop_dual_ur5e_hardware.py, not part of the PICO headset).

DH PGC-300-60 gripper: RS485 on /dev/ttyUSB0, VR trigger proportional open/close (no extra CLI flags).
"""

import threading
import time
from typing import Literal

import numpy as np
import tyro

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.universal_robots import (
    DEFAULT_ROBOT_IP,
    LEFT_INITIAL_JOINT_DEG,
    LEFT_ROBOT_IP,
    RIGHT_INITIAL_JOINT_DEG,
    RIGHT_ROBOT_IP,
)
from xrobotoolkit_teleop.hardware.single_arm_ur_controller import (
    LEFT_CONTROLLER_MANIPULATOR_CONFIG,
    RIGHT_CONTROLLER_MANIPULATOR_CONFIG,
    SingleArmURController,
)

# DH PGC-300-60 defaults (this lab unit: register 1000=open, 0=closed)
DH_GRIPPER_PORT = "/dev/ttyUSB0"
DH_GRIPPER_FORCE = 50
DH_GRIPPER_SPEED = 50
DH_GRIPPER_SLAVE_ID = 1
DH_GRIPPER_BAUDRATE = 115200


def main(
    reset: bool = False,
    visual: bool = False,
    robot_ip: str = DEFAULT_ROBOT_IP,
    controller_side: Literal["left", "right"] = "right",
    initial_joint_deg: tuple[float, float, float, float, float, float] | None = None,
):
    """
    Run single-arm UR5e teleoperation with a PICO VR controller.

    Args:
        reset: Move the arm to the configured home joints, then exit.
        visual: Open a Meshcat browser window for IK debugging.
        robot_ip: IP address of the UR controller (set in universal_robots.py or override here).
        controller_side: Which PICO hand controller drives the arm ("left" or "right").
        initial_joint_deg: Optional home joint angles in degrees (6 values). Defaults depend on
            controller_side when not set.
    """
    if controller_side == "left":
        manipulator_config = LEFT_CONTROLLER_MANIPULATOR_CONFIG
        default_initial_joint_deg = LEFT_INITIAL_JOINT_DEG
        if robot_ip == DEFAULT_ROBOT_IP:
            robot_ip = LEFT_ROBOT_IP
    else:
        manipulator_config = RIGHT_CONTROLLER_MANIPULATOR_CONFIG
        default_initial_joint_deg = RIGHT_INITIAL_JOINT_DEG

    initial_joint = (
        np.array(initial_joint_deg, dtype=float)
        if initial_joint_deg is not None
        else default_initial_joint_deg
    )

    xr_client = XrClient()
    arm_controller = None
    try:
        arm_controller = SingleArmURController(
            xr_client,
            manipulator_config=manipulator_config,
            robot_ip=robot_ip,
            initial_joint_deg=initial_joint,
            visualize_placo=visual,
            use_dh_gripper=True,
            dh_gripper_port=DH_GRIPPER_PORT,
            dh_gripper_force=DH_GRIPPER_FORCE,
            dh_gripper_speed=DH_GRIPPER_SPEED,
            dh_gripper_slave_id=DH_GRIPPER_SLAVE_ID,
            dh_gripper_baudrate=DH_GRIPPER_BAUDRATE,
        )
    except RuntimeError:
        xr_client.close()
        raise

    if reset:
        print("Reset flag detected. Running arm reset procedure...")
        try:
            arm_controller.robot_controller.reset()
            print("Arm reset procedure completed.")
        except Exception as e:
            print(f"Error during reset: {e}")
            arm_controller.close()
            xr_client.close()
            raise SystemExit(1) from e
    else:
        print(
            f"Single-arm teleop: robot_ip={robot_ip}, controller_side={controller_side}. "
            f"Hold the VR grip button to move the arm. "
            f"Trigger ({controller_side}) controls DH gripper on {DH_GRIPPER_PORT} (proportional)."
        )
        arm_controller.calc_target_joint_position()

        stop_signal = threading.Event()
        arm_thread = threading.Thread(
            target=arm_controller.run_robot_thread,
            args=(stop_signal,),
        )
        ik_thread = threading.Thread(
            target=arm_controller.run_ik_thread,
            args=(stop_signal,),
        )
        threads = [arm_thread, ik_thread]
        if arm_controller.use_dh_gripper:
            gripper_thread = threading.Thread(
                target=arm_controller.run_dh_gripper_thread,
                args=(stop_signal,),
            )
            threads.append(gripper_thread)

        for t in threads:
            t.start()

        while not stop_signal.is_set():
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected. Exiting...")
                stop_signal.set()

        for t in threads:
            t.join()

    if arm_controller is not None:
        arm_controller.close()
    xr_client.close()
    print("Controller stopped and disconnected.")


if __name__ == "__main__":
    tyro.cli(main)
