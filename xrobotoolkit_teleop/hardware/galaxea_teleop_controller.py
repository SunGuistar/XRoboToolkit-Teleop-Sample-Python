import os
import threading
import time
import webbrowser

import meshcat.transformations as tf
import numpy as np
import placo
import rospy
from placo_utils.visualization import frame_viz, robot_frame_viz, robot_viz

from xrobotoolkit_teleop.hardware.galaxea import A1XController
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    apply_delta_pose,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.parallel_gripper_utils import calc_parallel_gripper_position
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH
from xrobotoolkit_teleop.utils.xr_client import XrClient

# Default paths and configurations for a single right arm
DEFAULT_SINGLE_A1X_URDF_PATH = os.path.join(ASSET_PATH, "galaxea/A1X/a1x.urdf")
DEFAULT_DUAL_A1X_URDF_PATH = os.path.join(ASSET_PATH, "galaxea/A1X/dual_a1x.urdf")
DEFAULT_SCALE_FACTOR = 1.0
CONTROLLER_DEADZONE = 0.1

# Default end-effector configuration for a single right arm without a gripper
DEFAULT_END_EFFECTOR_CONFIG = {
    "right_arm": {
        "link_name": "gripper_link",  # URDF link name for the single arm
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "joint_name": "right_gripper_finger_joint1",
            "gripper_trigger": "right_trigger",
            "open_pos": -2.0,
            "close_pos": 0.0,
        },
    },
}

DEFAULT_DUAL_END_EFFECTOR_CONFIG = {
    "right_arm": {
        "link_name": "right_gripper_link",  # URDF link name for the single arm
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "joint_name": "right_gripper_finger_joint1",
            "gripper_trigger": "right_trigger",
            "open_pos": -2.0,
            "close_pos": 0.0,
        },
    },
    "left_arm": {
        "link_name": "left_gripper_link",  # URDF link name for the single arm
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_config": {
            "joint_name": "left_gripper_finger_joint1",
            "gripper_trigger": "left_trigger",
            "open_pos": -2.0,
            "close_pos": 0.0,
        },
    },
}


class GalaxeaA1XTeleopController:
    def __init__(
        self,
        xr_client: XrClient,
        robot_urdf_path: str = DEFAULT_SINGLE_A1X_URDF_PATH,
        end_effector_config: dict = DEFAULT_END_EFFECTOR_CONFIG,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = True,
        ros_rate_hz: int = 100,
    ):

        self.xr_client = xr_client
        self.robot_urdf_path = robot_urdf_path
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.visualize_placo = visualize_placo
        self.end_effector_config = end_effector_config

        rospy.init_node("galaxea_teleop_controller", anonymous=True)
        self.right_controller = A1XController(
            arm_control_topic="/motion_control/control_arm_right",
            gripper_control_topic="/motion_control/control_gripper_right",
            arm_state_topic="/hdas/feedback_arm_right",
            rate_hz=ros_rate_hz,
        )

        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = 1.0 / ros_rate_hz
        self.solver.mask_fbase(True)
        self.solver.add_kinetic_energy_regularization_task(1e-6)

        self.effector_task = {}
        self.init_ee_xyz = {}
        self.init_ee_quat = {}
        self.init_controller_xyz = {}
        self.init_controller_quat = {}
        for name, config in self.end_effector_config.items():
            self.effector_task[name] = self.solver.add_frame_task(config["link_name"], np.eye(4))
            self.effector_task[name].configure(f"{name}_frame", "soft", 1.0)
            self.solver.add_manipulability_task(config["link_name"], "both", 1.0).configure(
                f"{name}_manipulability", "soft", 5e-2
            )
            self.init_ee_xyz[name] = None
            self.init_ee_quat[name] = None
            self.init_controller_xyz[name] = None
            self.init_controller_quat[name] = None

        self.gripper_pos_target = {}
        for name, config in end_effector_config.items():
            if "gripper_config" in config:
                gripper_config = config["gripper_config"]
                self.gripper_pos_target[gripper_config["joint_name"]] = gripper_config["open_pos"]

        print("Waiting for initial joint state from the Galaxea arm...")
        while not rospy.is_shutdown() and self.right_controller.timestamp == 0:
            rospy.sleep(0.1)
        print("Initial joint state received.")

        self.placo_robot.state.q[7:13] = self.right_controller.qpos

        if self.visualize_placo:
            self.placo_robot.update_kinematics()
            self.placo_vis = robot_viz(self.placo_robot)
            time.sleep(0.5)
            webbrowser.open(self.placo_vis.viewer.url())
            self.placo_vis.display(self.placo_robot.state.q)
            for name, config in self.end_effector_config.items():
                robot_frame_viz(self.placo_robot, config["link_name"])
                frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

    def _process_xr_pose(self, xr_pose, arm_name: str):
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = np.array([xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]])

        controller_xyz = self.R_headset_world @ controller_xyz
        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat), tf.quaternion_conjugate(R_quat)
        )

        if self.init_controller_xyz[arm_name] is None:
            self.init_controller_xyz[arm_name] = controller_xyz.copy()
            self.init_controller_quat[arm_name] = controller_quat.copy()
            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])
        else:
            delta_xyz = (controller_xyz - self.init_controller_xyz[arm_name]) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(self.init_controller_quat[arm_name], controller_quat)
        return delta_xyz, delta_rot

    def update_ik(self):
        self.placo_robot.state.q[7:13] = self.right_controller.qpos
        self.placo_robot.update_kinematics()

        for arm_name, config in self.end_effector_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)

            if active:
                if self.init_ee_xyz[arm_name] is None:
                    self.init_ee_xyz[arm_name] = self.placo_robot.get_T_world_frame(config["link_name"])[:3, 3]
                    self.init_ee_quat[arm_name] = tf.quaternion_from_matrix(
                        self.placo_robot.get_T_world_frame(config["link_name"])
                    )
                    print(f"{arm_name} activated.")

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot = self._process_xr_pose(xr_pose, arm_name)
                target_xyz, target_quat = apply_delta_pose(
                    self.init_ee_xyz[arm_name], self.init_ee_quat[arm_name], delta_xyz, delta_rot
                )
                target_pose = tf.quaternion_matrix(target_quat)
                target_pose[:3, 3] = target_xyz
                self.effector_task[arm_name].T_world_frame = target_pose

                try:
                    self.solver.solve(True)
                    self.right_controller.q_des = self.placo_robot.state.q[7:13].copy()

                    if self.visualize_placo:
                        self.placo_vis.display(self.placo_robot.state.q)
                        for name in self.end_effector_config:
                            frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)
                except RuntimeError as e:
                    print(f"IK solver failed: {e}")
                except Exception as e:
                    print(f"An unexpected error occurred in IK: {e}")

            else:
                if self.init_ee_xyz[arm_name] is not None:
                    self.init_ee_xyz[arm_name] = None
                    self.init_controller_xyz[arm_name] = None
                    print(f"{arm_name} deactivated.")
                self.effector_task[arm_name].T_world_frame = self.placo_robot.get_T_world_frame(config["link_name"])

            # Update gripper position based on XR input
            if "gripper_config" in config:
                gripper_config = config["gripper_config"]
                trigger_value = self.xr_client.get_key_value_by_name(gripper_config["gripper_trigger"])
                gripper_pos = calc_parallel_gripper_position(
                    gripper_config["open_pos"],
                    gripper_config["close_pos"],
                    trigger_value,
                )
                joint_name = gripper_config["joint_name"]
                self.gripper_pos_target[joint_name] = [gripper_pos]

        self.right_controller.q_des_gripper = self.gripper_pos_target[
            self.end_effector_config["right_arm"]["gripper_config"]["joint_name"]
        ]

    def run_ik_thread(self, stop_event: threading.Event):
        print("Starting Galaxea A1X single arm teleop controller IK loop...")
        while not (rospy.is_shutdown() or stop_event.is_set()):
            start = time.time()
            self.update_ik()
            elapsed = time.time() - start
            time.sleep(max(0, (self.solver.dt - elapsed)))
        print("Galaxea teleop controller IK loop shutting down.")

    def run_control_thread(self, stop_event: threading.Event):
        print("Starting Galaxea A1X single arm teleop controller loop...")
        while not (rospy.is_shutdown() or stop_event.is_set()):
            self.right_controller.publish_arm_control()
            self.right_controller.publish_gripper_control()
            self.right_controller.rate.sleep()
        print("Galaxea teleop controller shutting down.")


class GalaxeaDualA1XTeleopController:
    def __init__(
        self,
        xr_client: XrClient,
        robot_urdf_path: str = DEFAULT_DUAL_A1X_URDF_PATH,
        end_effector_config: dict = DEFAULT_DUAL_END_EFFECTOR_CONFIG,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = True,
        ros_rate_hz: int = 100,
    ):

        self.xr_client = xr_client
        self.robot_urdf_path = robot_urdf_path
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.visualize_placo = visualize_placo
        self.end_effector_config = end_effector_config

        rospy.init_node("galaxea_teleop_controller", anonymous=True)

        self.left_controller = A1XController(
            arm_control_topic="/motion_control/control_arm_left",
            gripper_control_topic="/motion_control/control_gripper_left",
            arm_state_topic="/hdas/feedback_arm_left",
            rate_hz=ros_rate_hz,
        )

        self.right_controller = A1XController(
            arm_control_topic="/motion_control/control_arm_right",
            gripper_control_topic="/motion_control/control_gripper_right",
            arm_state_topic="/hdas/feedback_arm_right",
            rate_hz=ros_rate_hz,
        )

        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = 1.0 / ros_rate_hz
        self.solver.mask_fbase(True)
        self.solver.add_kinetic_energy_regularization_task(1e-6)

        self.effector_task = {}
        self.init_ee_xyz = {}
        self.init_ee_quat = {}
        self.init_controller_xyz = {}
        self.init_controller_quat = {}
        self.arm_active = {}
        for name, config in self.end_effector_config.items():
            self.effector_task[name] = self.solver.add_frame_task(config["link_name"], np.eye(4))
            self.effector_task[name].configure(f"{name}_frame", "soft", 1.0)
            self.solver.add_manipulability_task(config["link_name"], "both", 1.0).configure(
                f"{name}_manipulability", "soft", 5e-2
            )
            self.init_ee_xyz[name] = None
            self.init_ee_quat[name] = None
            self.init_controller_xyz[name] = None
            self.init_controller_quat[name] = None
            self.arm_active[name] = False

        self.gripper_pos_target = {}
        for name, config in end_effector_config.items():
            if "gripper_config" in config:
                gripper_config = config["gripper_config"]
                self.gripper_pos_target[gripper_config["joint_name"]] = gripper_config["open_pos"]

        print("Waiting for initial joint state from the Galaxea arm...")
        while not rospy.is_shutdown() and self.right_controller.timestamp == 0:
            rospy.sleep(0.1)
        print("Initial joint state received.")

        self.placo_robot.state.q[7:13] = self.left_controller.qpos
        self.placo_robot.state.q[15:21] = self.right_controller.qpos

        if self.visualize_placo:
            self.placo_robot.update_kinematics()
            self.placo_vis = robot_viz(self.placo_robot)
            time.sleep(0.5)
            webbrowser.open(self.placo_vis.viewer.url())
            self.placo_vis.display(self.placo_robot.state.q)
            for name, config in self.end_effector_config.items():
                robot_frame_viz(self.placo_robot, config["link_name"])
                frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

    def _process_xr_pose(self, xr_pose, arm_name: str):
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = np.array([xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]])

        controller_xyz = self.R_headset_world @ controller_xyz
        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat), tf.quaternion_conjugate(R_quat)
        )

        if self.init_controller_xyz[arm_name] is None:
            self.init_controller_xyz[arm_name] = controller_xyz.copy()
            self.init_controller_quat[arm_name] = controller_quat.copy()
            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])
        else:
            delta_xyz = (controller_xyz - self.init_controller_xyz[arm_name]) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(self.init_controller_quat[arm_name], controller_quat)
        return delta_xyz, delta_rot

    def update_ik(self):
        self.placo_robot.state.q[7:13] = self.left_controller.qpos
        self.placo_robot.state.q[15:21] = self.right_controller.qpos
        self.placo_robot.update_kinematics()

        for arm_name, config in self.end_effector_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            self.arm_active[arm_name] = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)

            if self.arm_active[arm_name]:
                if self.init_ee_xyz[arm_name] is None:
                    self.init_ee_xyz[arm_name] = self.placo_robot.get_T_world_frame(config["link_name"])[:3, 3]
                    self.init_ee_quat[arm_name] = tf.quaternion_from_matrix(
                        self.placo_robot.get_T_world_frame(config["link_name"])
                    )
                    print(f"{arm_name} activated.")

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot = self._process_xr_pose(xr_pose, arm_name)
                target_xyz, target_quat = apply_delta_pose(
                    self.init_ee_xyz[arm_name], self.init_ee_quat[arm_name], delta_xyz, delta_rot
                )
                target_pose = tf.quaternion_matrix(target_quat)
                target_pose[:3, 3] = target_xyz
                self.effector_task[arm_name].T_world_frame = target_pose

            else:
                if self.init_ee_xyz[arm_name] is not None:
                    self.init_ee_xyz[arm_name] = None
                    self.init_controller_xyz[arm_name] = None
                    print(f"{arm_name} deactivated.")
                self.effector_task[arm_name].T_world_frame = self.placo_robot.get_T_world_frame(config["link_name"])

        try:
            self.solver.solve(True)

            if self.visualize_placo:
                self.placo_vis.display(self.placo_robot.state.q)
                for name in self.end_effector_config:
                    frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)
        except RuntimeError as e:
            print(f"IK solver failed: {e}")
        except Exception as e:
            print(f"An unexpected error occurred in IK: {e}")

        for arm_name, config in self.end_effector_config.items():
            if arm_name == "left_arm" and self.arm_active[arm_name]:
                self.left_controller.q_des = self.placo_robot.state.q[7:13].copy()
            if arm_name == "right_arm" and self.arm_active[arm_name]:
                self.right_controller.q_des = self.placo_robot.state.q[15:21].copy()
            # Update gripper position based on XR input
            if "gripper_config" in config:
                gripper_config = config["gripper_config"]
                trigger_value = self.xr_client.get_key_value_by_name(gripper_config["gripper_trigger"])
                gripper_pos = calc_parallel_gripper_position(
                    gripper_config["open_pos"],
                    gripper_config["close_pos"],
                    trigger_value,
                )
                joint_name = gripper_config["joint_name"]
                self.gripper_pos_target[joint_name] = [gripper_pos]

        self.left_controller.q_des_gripper = self.gripper_pos_target[
            self.end_effector_config["left_arm"]["gripper_config"]["joint_name"]
        ]
        self.right_controller.q_des_gripper = self.gripper_pos_target[
            self.end_effector_config["right_arm"]["gripper_config"]["joint_name"]
        ]
        print(self.right_controller.q_des_gripper)

    def run_ik_thread(self, stop_event: threading.Event):
        print("Starting Galaxea A1X single arm teleop controller IK loop...")
        while not (rospy.is_shutdown() or stop_event.is_set()):
            start = time.time()
            self.update_ik()
            elapsed = time.time() - start
            time.sleep(max(0, (self.solver.dt - elapsed)))
        print("Galaxea teleop controller IK loop shutting down.")

    def run_control_thread(self, stop_event: threading.Event):
        print("Starting Galaxea A1X single arm teleop controller loop...")
        while not (rospy.is_shutdown() or stop_event.is_set()):
            self.left_controller.publish_arm_control()
            self.left_controller.publish_gripper_control()
            self.right_controller.publish_arm_control()
            self.right_controller.publish_gripper_control()
            self.right_controller.rate.sleep()
        print("Galaxea teleop controller shutting down.")
