import os
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
from xrobotoolkit_teleop.utils.gripper_utils import calc_parallel_gripper_position
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH
from xrobotoolkit_teleop.utils.xr_client import XrClient

# Default paths and configurations for a single left arm
DEFAULT_SINGLE_A1X_URDF_PATH = os.path.join(ASSET_PATH, "galaxea/A1X/a1x.urdf")
DEFAULT_SCALE_FACTOR = 1.0
CONTROLLER_DEADZONE = 0.1

# Default end-effector configuration for a single left arm without a gripper
DEFAULT_END_EFFECTOR_CONFIG = {
    "left_arm": {
        "link_name": "gripper_link",  # URDF link name for the single arm
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


class GalaxeaTeleopControllerSingleArm:
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
        # rospy.init_node("galaxea_teleop_controller_single", anonymous=True)

        self.xr_client = xr_client
        self.robot_urdf_path = robot_urdf_path
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.visualize_placo = visualize_placo
        self.end_effector_config = end_effector_config
        # self.rate = rospy.Rate(ros_rate_hz)

        # Define joint names for the single left arm (no gripper)
        # left_joint_names = [f"arm_joint_{i}" for i in range(1, 7)]

        # Instantiate hardware controller for the left arm
        # Assuming the single arm uses the "left" prefix for its topics
        self.left_controller = A1XController()

        # Placo Setup
        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = 1.0 / ros_rate_hz
        self.solver.mask_fbase(True)
        self.solver.add_kinetic_energy_regularization_task(1e-6)

        # Setup Placo tasks for the single arm
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

        # Wait for the first joint state message to initialize Placo state
        print("Waiting for initial joint state from the Galaxea arm...")
        while not rospy.is_shutdown() and self.left_controller.timestamp == 0:
            rospy.sleep(0.1)
        print("Initial joint state received.")

        # Set initial Placo state from hardware. Assumes single arm URDF has 7 DoF base + 6 DoF arm
        self.placo_robot.state.q[7:13] = self.left_controller.qpos

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
        controller_quat = np.array([xr_pose[6], xr_pose[3], xr_pose[4], xr_pose[5]])  # w, x, y, z

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

    def update_ik_and_publish(self):
        # Update placo state with the latest from hardware
        self.placo_robot.state.q[7:13] = self.left_controller.qpos
        self.placo_robot.update_kinematics()

        # Process XR inputs and update IK task (only one loop for the single arm)
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

                # Solve IK
                try:
                    self.solver.solve(True)
                    # Update target joint positions from IK solution
                    self.left_controller.q_des = self.placo_robot.state.q[7:13].copy()

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
                # Keep task target at current robot pose when not active
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

        # if abs(self.gripper_pos_target["left_gripper_finger_joint1"][0] - self.left_controller.qpos_gripper[0]) > 0.01:
        self.left_controller.q_des_gripper = self.gripper_pos_target["left_gripper_finger_joint1"]
        self.left_controller.publish_gripper_control()
        # Publish commands to hardware
        self.left_controller.publish_arm_control()

    def run(self):
        print("Starting Galaxea A1X single arm teleop controller loop...")
        while not rospy.is_shutdown():
            self.update_ik_and_publish()
            self.left_controller.rate.sleep()
        print("Galaxea teleop controller shutting down.")


if __name__ == "__main__":
    # This is an example of how to run the controller
    # You would need to have your XR client and ROS environment set up
    try:
        xr_client = XrClient()
        controller = GalaxeaTeleopControllerSingleArm(xr_client=xr_client)
        controller.run()
    except rospy.ROSInterruptException:
        print("ROS node interrupted.")
    except Exception as e:
        print(f"An error occurred: {e}")
