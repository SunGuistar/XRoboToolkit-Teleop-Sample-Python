import time
import webbrowser
from typing import Any, Dict

import meshcat.transformations as tf
import numpy as np
import placo
from placo_utils.visualization import (
    frame_viz,
    robot_frame_viz,
    robot_viz,
)

from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    apply_delta_pose,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.xr_client import XrClient


class PlacoTeleopController:
    def __init__(
        self,
        robot_urdf_path: str,
        end_effector_config: Dict[str, Dict[str, Any]],
        floating_base=False,
        R_headset_world=R_HEADSET_TO_WORLD,
        scale_factor=1.0,
        q_init=None,
        dt=0.01,
    ):
        """
        Initialize the teleoperation controller.

        Args:
            robot_urdf_path: Path to the robot URDF file for IK solving
            end_effector_config: Dictionary containing end effector configurations
            floating_base: Whether the robot has a floating base
            R_headset_world: Rotation matrix to transform headset coordinates to world coordinates
            visualize_placo: Whether to visualize the placo IK solver
            scale_factor: Scale factor for teleoperation
            q_init: Initial joint configuration
            dt: Simulation timestep
        """
        self.robot_urdf_path = robot_urdf_path
        self.end_effector_config = end_effector_config
        self.floating_base = floating_base
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.q_init = q_init
        self.dt = dt

        # To be initialized later
        self.robot = None
        self.solver = None
        self.t = 0
        self.effector_task = {name: None for name in end_effector_config.keys()}

        # Initial poses
        self.init_ee_xyz = {name: None for name in end_effector_config.keys()}
        self.init_ee_quat = {name: None for name in end_effector_config.keys()}
        self.init_controller_xyz = {name: None for name in end_effector_config.keys()}
        self.init_controller_quat = {name: None for name in end_effector_config.keys()}

        self.xr_client = XrClient()
        self._setup_placo()

    def _get_end_effector_info(self, ee_name):
        """Get the end effector position and orientation from placo robot."""
        self.robot.update_kinematics()
        T_world_ee = self.robot.get_T_world_frame(ee_name)

        ee_xyz = T_world_ee[:3, 3]
        ee_quat = tf.quaternion_from_matrix(T_world_ee)

        return ee_xyz, ee_quat

    def _setup_placo(self):
        """Set up the placo inverse kinematics solver."""
        self.robot = placo.RobotWrapper(self.robot_urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.dt = self.dt
        self.solver.add_kinetic_energy_regularization_task(1e-6)

        # Set initial configuration
        if self.q_init is not None:
            if self.floating_base:
                self.robot.state.q = self.q_init.copy()
            else:
                self.solver.mask_fbase(True)
                self.robot.state.q[7:] = self.q_init.copy()
        else:
            # Use default configuration
            if not self.floating_base:
                self.solver.mask_fbase(True)
                self.robot.state.q[:7] = np.array(
                    [0, 0, 0, 0, 0, 0, 1]
                )  # Identity quaternion for base

        # Print all joint names from placo model
        print("Joint names in the Placo model:")
        for joint_name in self.robot.model.names:
            if joint_name != "universe":  # Exclude the universe joint
                print(f"  {joint_name}")

        # Update kinematics with initial configuration
        self.robot.update_kinematics()

        # Set up end effector tasks
        for name, config in self.end_effector_config.items():
            ee_xyz, ee_quat = self._get_end_effector_info(config["link_name"])
            ee_target = tf.quaternion_matrix(ee_quat)
            ee_target[:3, 3] = ee_xyz
            self.effector_task[name] = self.solver.add_frame_task(config["link_name"], ee_target)
            self.effector_task[name].configure(name, "soft", 1.0)
            manipulability = self.solver.add_manipulability_task(config["link_name"], "both", 1.0)
            manipulability.configure("manipulability", "soft", 5e-2)

        self.robot.update_kinematics()
        self.placo_vis = robot_viz(self.robot)
        webbrowser.open(self.placo_vis.viewer.url())
        self.placo_vis.display(self.robot.state.q)
        for name, config in self.end_effector_config.items():
            robot_frame_viz(self.robot, config["link_name"])
            frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

    def run(self):
        """Run the main teleoperation loop."""

        try:
            while True:
                start_time = time.time()

                self.t += self.dt

                for name, config in self.end_effector_config.items():
                    xr_grip = self.xr_client.get_key_value_by_name(config["control_trigger"])
                    active = xr_grip > 0.5

                    # Process current pose
                    if active:
                        if self.init_ee_xyz[name] is None:
                            self.init_ee_xyz[name], self.init_ee_quat[name] = (
                                self._get_end_effector_info(config["link_name"])
                            )
                            print(f"{name} is activated.")

                        xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                        delta_xyz, delta_rot = self._process_xr_pose(xr_pose, name)

                        target_xyz, target_quat = apply_delta_pose(
                            self.init_ee_xyz[name],
                            self.init_ee_quat[name],
                            delta_xyz,
                            delta_rot,
                        )

                        target = tf.quaternion_matrix(target_quat)
                        target[:3, 3] = target_xyz

                        self.effector_task[name].T_world_frame = target
                    else:
                        if self.init_ee_xyz[name] is not None:
                            # Reset the initial pose when the controller is released
                            self.init_ee_xyz[name] = None
                            self.init_ee_quat[name] = None
                            self.init_controller_xyz[name] = None
                            self.init_controller_quat[name] = None
                            print(f"{name} is deactivated.")

                self._update_kinematics()

                # Sleep to maintain desired timestep
                elapsed = time.time() - start_time
                sleep_time = max(0, self.dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nTeleoperation stopped.")

    def _process_xr_pose(self, xr_pose, src_name):
        """Process the current XR controller pose."""
        # Get position and orientation
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = [
            xr_pose[6],  # w
            xr_pose[3],  # x
            xr_pose[4],  # y
            xr_pose[5],  # z
        ]

        controller_xyz = self.R_headset_world @ controller_xyz

        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
        )

        if self.init_controller_xyz[src_name] is None:
            # First time processing the pose
            self.init_controller_xyz[src_name] = controller_xyz
            self.init_controller_quat[src_name] = controller_quat

            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])
        else:
            # Calculate relative transformation from init pose
            delta_xyz = (controller_xyz - self.init_controller_xyz[src_name]) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(
                self.init_controller_quat[src_name], controller_quat
            )

        return delta_xyz, delta_rot

    def _update_kinematics(self):
        """Update the placo robot kinematics by solving IK."""
        try:
            self.solver.solve(True)
            self.robot.update_kinematics()
        except RuntimeError as e:
            if "QPError" in str(e) and "NaN in the QP solution" in str(e):
                print(f"IK solver failed with QPError (NaN): {e}")
                # Keep current configuration on failure
                self.robot.update_kinematics()
            else:
                print(f"An unexpected RuntimeError occurred in IK solver: {e}")
                return
        except Exception as e:
            print(f"An unexpected error occurred in IK solver: {e}")
            return

        self.placo_vis.display(self.robot.state.q)
        for name, config in self.end_effector_config.items():
            robot_frame_viz(self.robot, config["link_name"])
            frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

    def get_current_joint_positions(self):
        """Get the current joint positions."""
        if self.floating_base:
            return self.robot.state.q.copy()
        else:
            return self.robot.state.q[7:].copy()

    def set_joint_positions(self, q):
        """Set the joint positions."""
        if self.floating_base:
            self.robot.state.q = q.copy()
        else:
            self.robot.state.q[7:] = q.copy()
        self.robot.update_kinematics()
