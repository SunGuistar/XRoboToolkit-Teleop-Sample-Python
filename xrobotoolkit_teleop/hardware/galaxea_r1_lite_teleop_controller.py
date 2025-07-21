import os
import threading
import time
from typing import Dict

import meshcat.transformations as tf
import numpy as np
import rospy

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.hardware.interface.galaxea import A1XController, R1LiteChassisController, R1LiteTorsoController
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

# Default paths and configurations for R1 Lite dual arm
DEFAULT_DUAL_A1X_URDF_PATH = os.path.join(ASSET_PATH, "galaxea/A1X/dual_a1x.urdf")
DEFAULT_SCALE_FACTOR = 1.0
CONTROLLER_DEADZONE = 0.1

# R1 Lite always has both arms - no single arm configuration needed
DEFAULT_END_EFFECTOR_CONFIG = {
    "right_arm": {
        "link_name": "right_gripper_link",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": [
                "right_gripper_finger_joint1",
            ],
            "open_pos": [
                -2.9,
            ],
            "close_pos": [
                0.0,
            ],
        },
    },
    "left_arm": {
        "link_name": "left_gripper_link",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "left_trigger",
            "joint_names": [
                "left_gripper_finger_joint1",
            ],
            "open_pos": [
                -2.9,
            ],
            "close_pos": [
                0.0,
            ],
        },
    },
}


class GalaxeaR1LiteTeleopController(BaseTeleopController):
    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_DUAL_A1X_URDF_PATH,
        end_effector_config: dict = DEFAULT_END_EFFECTOR_CONFIG,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        chassis_velocity_scale: list[float] = [0.75, 0.75, 1.0],
        visualize_placo: bool = False,
        ros_rate_hz: int = 100,
        enable_log_data: bool = True,
        log_dir: str = "logs/galaxea_r1_lite",
        log_freq: float = 50,
    ):
        super().__init__(
            robot_urdf_path=robot_urdf_path,
            end_effector_config=end_effector_config,
            floating_base=False,
            R_headset_world=R_headset_world,
            scale_factor=scale_factor,
            q_init=None,
            dt=1.0 / ros_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
        )

        self._start_time = 0
        self.ros_rate_hz = ros_rate_hz
        self.visualize_placo = visualize_placo
        self.chassis_velocity_scale = chassis_velocity_scale

        if self.visualize_placo:
            self._init_placo_viz()

        self._prev_b_button_state = False
        self._is_logging = False

    def _placo_setup(self):
        super()._placo_setup()
        # R1 Lite always has both left and right arms
        self.placo_arm_joint_slice = {}
        for arm_name in ["left_arm", "right_arm"]:
            config = self.end_effector_config[arm_name]
            ee_link_name = config["link_name"]
            arm_prefix = ee_link_name.replace("gripper_link", "")
            arm_joint_names = [f"{arm_prefix}arm_joint{i}" for i in range(1, 7)]
            self.placo_arm_joint_slice[arm_name] = slice(
                self.placo_robot.get_joint_offset(arm_joint_names[0]),
                self.placo_robot.get_joint_offset(arm_joint_names[-1]) + 1,
            )

    def _robot_setup(self):
        rospy.init_node("galaxea_r1_lite_teleop_controller", anonymous=True)

        # setup arm controllers
        self.arm_controllers: Dict[str, A1XController] = {}
        for arm_name in ["left_arm", "right_arm"]:
            arm_prefix = arm_name.replace("_arm", "")
            controller = A1XController(
                arm_control_topic=f"/motion_control/control_arm_{arm_prefix}",
                gripper_control_topic=f"/motion_control/control_gripper_{arm_prefix}",
                arm_state_topic=f"/hdas/feedback_arm_{arm_prefix}",
                rate_hz=1.0 / self.dt,
                gripper_position_control=False,
            )
            self.arm_controllers[arm_name] = controller

        print("Waiting for initial joint states from both R1 Lite arms...")
        all_controllers_ready = False
        while not rospy.is_shutdown() and not all_controllers_ready:
            all_controllers_ready = all(controller.timestamp > 0 for controller in self.arm_controllers.values())
            rospy.sleep(0.1)
        print("Both arm controllers received initial state.")

        # Setup chassis controller
        self.chassis_controller = R1LiteChassisController(
            chassis_state_topic="/hdas/feedback_chassis",
            chassis_control_topic="/motion_target/target_speed_chassis",
            rate_hz=1.0 / self.dt,
        )

        print("Waiting for initial chassis state...")
        while not rospy.is_shutdown() and self.chassis_controller.timestamp == -1:
            rospy.sleep(0.1)
        print("Chassis controller received initial state.")

        self.torso_controller = R1LiteTorsoController(
            torso_state_topic="/hdas/feedback_torso",
            torso_control_topic="/motion_target/target_speed_torso",
            rate_hz=1.0 / self.dt,
        )

        print("Waiting for initial torso state...")
        while not rospy.is_shutdown() and self.torso_controller.timestamp == -1:
            rospy.sleep(0.1)
        print("Torso controller received initial state.")

    def _update_robot_state(self):
        """Reads current joint states from both arm controllers and updates Placo."""
        for arm_name, controller in self.arm_controllers.items():
            self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]] = controller.qpos

    def _send_command(self):
        """Sends the solved joint targets to both arm controllers."""
        for arm_name, controller in self.arm_controllers.items():
            if self.active.get(arm_name, False):
                controller.q_des = self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]].copy()

            # R1 Lite always has gripper config for both arms
            controller.q_des_gripper = [
                self.gripper_pos_target[arm_name][gripper_joint]
                for gripper_joint in self.gripper_pos_target[arm_name].keys()
            ]

            controller.publish_arm_control()
            controller.publish_gripper_control()

        self.chassis_controller.publish_chassis_control()
        self.torso_controller.publish_torso_control()

    def _update_joystick_velocity_command(self):
        """Updates the chassis velocity commands based on joystick input."""
        left_axis = self.xr_client.get_joystick_state("left")
        right_axis = self.xr_client.get_joystick_state("right")

        vx = left_axis[1] * self.chassis_velocity_scale[0]
        vy = left_axis[0] * self.chassis_velocity_scale[1]
        omega = -right_axis[0] * self.chassis_velocity_scale[2]

        self.chassis_controller.set_velocity_command(vx, vy, omega)

    def _update_torso_velocity_command(self):
        buttonY = self.xr_client.get_button_state_by_name("Y")
        buttonX = self.xr_client.get_button_state_by_name("X")

        vz = 2.5 if buttonY else -2.5 if buttonX else 0.0
        self.torso_controller.set_velocity_command(vz)

    def _get_link_pose(self, link_name: str):
        """Gets the current world pose for a given link name from Placo."""
        T_world_link = self.placo_robot.get_T_world_frame(link_name)
        pos = T_world_link[:3, 3]
        quat = tf.quaternion_from_matrix(T_world_link)
        return pos, quat

    def _log_data(self):
        if self.enable_log_data:
            timestamp = time.time() - self._start_time
            data_entry = {
                "timestamp": timestamp,
                "qpos": {arm: controller.qpos for arm, controller in self.arm_controllers.items()},
                "qvel": {arm: controller.qvel for arm, controller in self.arm_controllers.items()},
                "qpos_des": {arm: controller.q_des for arm, controller in self.arm_controllers.items()},
                "gripper_qpos": {arm: controller.qpos_gripper for arm, controller in self.arm_controllers.items()},
                "gripper_qpos_des": {arm: controller.q_des_gripper for arm, controller in self.arm_controllers.items()},
            }
            self.data_logger.add_entry(data_entry)

    def _ik_thread(self, stop_event: threading.Event):
        """Dedicated thread for running the IK solver."""
        rate = rospy.Rate(1.0 / self.dt)
        while not stop_event.is_set():
            self._update_robot_state()
            self._update_gripper_target()
            self._update_joystick_velocity_command()
            self._update_torso_velocity_command()
            self._update_ik()
            if self.visualize_placo:
                self._update_placo_viz()
            rate.sleep()
        print("IK loop has stopped.")

    def _control_thread(self, stop_event: threading.Event):
        """Dedicated thread for sending commands to hardware."""
        rate = rospy.Rate(self.ros_rate_hz)
        while not stop_event.is_set():
            self._send_command()
            rate.sleep()
        self.chassis_controller.stop_chassis()
        print("Control loop has stopped.")

    def _data_logging_thread(self, stop_event: threading.Event):
        rate = rospy.Rate(self.log_freq)
        while not stop_event.is_set():
            if self.enable_log_data:
                self._check_logging_button()
                if self._is_logging:
                    self._log_data()
            rate.sleep()
        print("Data logging thread has stopped.")

    def _check_logging_button(self):
        """Checks for the 'B' button press to toggle data logging."""
        b_button_state = self.xr_client.get_button_state_by_name("B")
        right_axis_click = self.xr_client.get_button_state_by_name("right_axis_click")

        # Check for a rising edge (button pressed)
        if b_button_state and not self._prev_b_button_state:
            self._is_logging = not self._is_logging
            if self._is_logging:
                print("--- Started data logging ---")
            else:
                print("--- Stopped data logging. Saving data... ---")
                self.data_logger.save()
                self.data_logger.reset()

        if right_axis_click and self._is_logging:
            print("--- Stopped data logging. Discarding data... ---")
            self.data_logger.reset()
            self._is_logging = False

        self._prev_b_button_state = b_button_state

    def run(self):
        """
        Main entry point that starts the multi-threaded IK and control loops.
        """
        self._start_time = time.time()
        ik_thread = threading.Thread(target=self._ik_thread, args=(self._stop_event,))
        control_thread = threading.Thread(target=self._control_thread, args=(self._stop_event,))

        ik_thread.daemon = True
        control_thread.daemon = True

        ik_thread.start()
        control_thread.start()

        if self.enable_log_data:
            log_thread = threading.Thread(target=self._data_logging_thread, args=(self._stop_event,))
            log_thread.daemon = True
            log_thread.start()

        print("R1 Lite dual-arm teleoperation running. Press Ctrl+C to exit.")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
        finally:
            print("Shutting down...")
            self._stop_event.set()
            ik_thread.join(timeout=2.0)
            control_thread.join(timeout=2.0)
            if self.enable_log_data:
                log_thread.join(timeout=2.0)
            print("All threads have been shut down.")
