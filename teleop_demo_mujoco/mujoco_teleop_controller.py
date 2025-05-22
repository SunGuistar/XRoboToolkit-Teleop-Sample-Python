from typing import Dict, Any
import os
import mujoco
from mujoco import viewer as mj_viewer
import numpy as np
import pinocchio
import placo
import meshcat.transformations as tf
import pyroboticsservice as xr

from placo_utils.visualization import (
    robot_viz,
    robot_frame_viz,
    frame_viz,
)

from teleop_demo_mujoco.geometry import (
    R_HEADSET_TO_WORLD,
    apply_delta_pose,
    quat_diff_as_angle_axis,
)


class MujocoTeleopController:
    def __init__(
        self,
        xml_path: str,
        robot_urdf_path: str,
        end_effector_config: Dict[str, Dict[str, Any]],
        floating_base=False,
        R_headset_world=R_HEADSET_TO_WORLD,
        visualize_placo=False,
        scale_factor=1.0,
        q_init=None,
    ):
        """
        Initialize the teleoperation controller.

        Args:
            xml_path: Path to the MuJoCo XML scene file
            robot_urdf_path: Path to the robot URDF file for IK solving
            end_effector_config: Dictionary containing end effector configurations
            floating_base: Whether the robot has a floating base
            R_headset_world: Rotation matrix to transform headset coordinates to world coordinates
            visualize_placo: Whether to visualize the placo IK solver
            scale_factor: Scale factor for teleoperation
        """
        self.xml_path = xml_path
        self.robot_urdf_path = robot_urdf_path
        self.end_effector_config = end_effector_config
        self.floating_base = floating_base
        self.R_headset_world = R_headset_world
        self.visualize_placo = visualize_placo
        self.scale_factor = scale_factor
        self.q_init = q_init

        # To be initialized later
        self.mj_model = None
        self.mj_data = None
        self.placo_robot = None
        self.solver = None
        self.dt = None
        self.t = 0
        self.effector_task = {name: None for name in end_effector_config.keys()}
        self.target_mocap_idx = {
            name: -1 for name in end_effector_config.keys()
        }

        # Initial poses
        self.init_ee_xyz = {name: None for name in end_effector_config.keys()}
        self.init_ee_quat = {name: None for name in end_effector_config.keys()}
        self.init_controller_xyz = {
            name: None for name in end_effector_config.keys()
        }
        self.init_controller_quat = {
            name: None for name in end_effector_config.keys()
        }

    def initialize(self):
        """Set up the MuJoCo simulation and the IK solver."""
        self._setup_mujoco()
        self._setup_placo()

    def _get_end_effector_info(self, ee_name):
        """Get the end effector position and orientation."""
        ee_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, ee_name
        )
        if ee_id == -1:
            raise ValueError(
                f"End effector body '{ee_name}' not found in the model."
            )

        ee_xyz = self.mj_data.xpos[ee_id].copy()
        ee_quat = self.mj_data.xquat[ee_id].copy()

        return ee_xyz, ee_quat

    def _setup_mujoco(self):
        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Print all joint names
        print("Joint names in the Mujoco model:")
        for i in range(self.mj_model.njnt):
            joint_name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i
            )
            print(f"  {joint_name}")

        # Configure scene lighting
        self.mj_model.vis.headlight.ambient = [0.4, 0.4, 0.4]
        self.mj_model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
        self.mj_model.vis.headlight.specular = [0.6, 0.6, 0.6]

        # Set initial position from keyframe if available
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.q_init is None:
            mujoco.mj_resetDataKeyframe(
                self.mj_model, self.mj_data, self.mj_model.key("home").id
            )
        else:
            self.mj_data.qpos[:] = self.q_init
            mujoco.mj_forward(self.mj_model, self.mj_data)

        # Calculate forward kinematics to get initial end effector pose
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._setup_mocap_target()

    def _setup_placo(self):
        """Set up the placo inverse kinematics solver."""
        self.dt = self.mj_model.opt.timestep
        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = self.dt
        self.solver.add_kinetic_energy_regularization_task(1e-6)

        # Print all joint names from placo model
        print("Joint names in the Placo model:")
        for joint_name in self.placo_robot.model.names:
            if joint_name != "universe":  # Exclude the universe joint
                print(f"  {joint_name}")

        for name, config in self.end_effector_config.items():
            ee_xyz, ee_quat = self._get_end_effector_info(config["link_name"])
            ee_target = tf.quaternion_matrix(ee_quat)
            ee_target[:3, 3] = ee_xyz
            self.effector_task[name] = self.solver.add_frame_task(
                config["link_name"], ee_target
            )
            self.effector_task[name].configure(name, "soft", 1.0)
            manipulability = self.solver.add_manipulability_task(
                config["link_name"], "both", 1.0
            )
            manipulability.configure("manipulability", "soft", 5e-2)

        if self.floating_base:
            self.placo_robot.state.q = self.mj_data.qpos.copy()
        else:
            self.solver.mask_fbase(True)
            self.placo_robot.state.q[7:] = self.mj_data.qpos.copy()

        if self.visualize_placo:
            self.placo_robot.update_kinematics()
            self.placo_vis = robot_viz(self.placo_robot)
            self.placo_vis.display(self.placo_robot.state.q)
            for name, config in self.end_effector_config.items():
                robot_frame_viz(self.placo_robot, config["link_name"])
                frame_viz(
                    config["vis_target"], self.effector_task[name].T_world_frame
                )

    def _setup_mocap_target(self):
        """Find and set up the mocap target body."""
        for name, config in self.end_effector_config.items():
            vis_target = config["vis_target"]
            mocap_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, vis_target
            )
            if mocap_id == -1:
                raise ValueError(
                    f"Mocap body '{vis_target}' not found in the model."
                )

            if self.mj_model.body_mocapid[mocap_id] == -1:
                raise ValueError(
                    f"Body '{self.vis_target}' is not configured for mocap."
                )
            else:
                self.target_mocap_idx[name] = self.mj_model.body_mocapid[
                    mocap_id
                ]

            print(
                f"Mocap ID for '{vis_target}' body: {self.target_mocap_idx[name]}"
            )

    def xr_init(self):
        """Initialize XR tracking."""
        xr.init()

    def run(self):
        """Run the main teleoperation loop."""

        with mj_viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            # Set up viewer camera
            viewer.cam.azimuth = 0
            viewer.cam.elevation = -50
            viewer.cam.distance = 2.0
            viewer.cam.lookat = [0.2, 0, 0]

            while True:

                self.t += self.dt

                for name, config in self.end_effector_config.items():
                    xr_grip = xr.get_key_value_by_name(
                        config["control_trigger"]
                    )
                    active = xr_grip > 0.5

                    # Process current pose
                    if active:
                        if self.init_ee_xyz[name] is None:
                            self.init_ee_xyz[name], self.init_ee_quat[name] = (
                                self._get_end_effector_info(config["link_name"])
                            )
                            print(f"{name} is activated.")
                        xr_pose = xr.get_pose_by_name(config["pose_source"])
                        delta_xyz, delta_rot = self._process_xr_pose(
                            xr_pose, name
                        )

                        target_xyz, target_quat = apply_delta_pose(
                            self.init_ee_xyz[name],
                            self.init_ee_quat[name],
                            delta_xyz,
                            delta_rot,
                        )

                        target = tf.quaternion_matrix(target_quat)
                        target[:3, 3] = target_xyz

                        self.effector_task[name].T_world_frame = target

                        target_mocap_id = self.target_mocap_idx[name]
                        if target_mocap_id != -1:
                            self.mj_data.mocap_pos[target_mocap_id] = target_xyz
                            self.mj_data.mocap_quat[target_mocap_id] = (
                                target_quat
                            )
                    else:
                        if self.init_ee_xyz[name] is not None:
                            # Reset the initial pose when the controller is released
                            self.init_ee_xyz[name] = None
                            self.init_ee_quat[name] = None
                            self.init_controller_xyz[name] = None
                            self.init_controller_quat[name] = None
                            print(f"{name} is deactivated.")

                self._update_kinematics()

                # Step simulation and update viewer
                mujoco.mj_step(self.mj_model, self.mj_data)
                viewer.sync()

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
        controller_quat = tf.quaternion_multiply(R_quat, controller_quat)

        if self.init_controller_xyz[src_name] is None:
            # First time processing the pose
            self.init_controller_xyz[src_name] = controller_xyz
            self.init_controller_quat[src_name] = controller_quat

            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])
        else:
            # Calculate relative transformation from init pose
            delta_xyz = (
                controller_xyz - self.init_controller_xyz[src_name]
            ) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(
                self.init_controller_quat[src_name], controller_quat
            )

        return delta_xyz, delta_rot

    def _update_kinematics(self):
        # Solve IK
        self.solver.solve(True)
        self.placo_robot.update_kinematics()
        if self.floating_base:
            q = self.placo_robot.state.q
        else:
            q = self.placo_robot.state.q[7:]

        self.mj_data.ctrl = q

        if self.visualize_placo:
            self.placo_vis.display(self.placo_robot.state.q)

            for name, config in self.end_effector_config.items():
                robot_frame_viz(self.placo_robot, config["link_name"])
                frame_viz(
                    config["vis_target"], self.effector_task[name].T_world_frame
                )
