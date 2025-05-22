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


class MujocoTeleopControllerSingleEffector:
    def __init__(
        self,
        xml_path,
        robot_urdf_path,
        end_effector_name="wrist_3_link",
        vis_target="target",
        floating_base=False,
        R_headset_world=R_HEADSET_TO_WORLD,
        visualize_placo=False,
        scale_factor=1.0,
    ):
        """
        Initialize the teleoperation controller.

        Args:
            xml_path: Path to the MuJoCo XML scene file
            robot_urdf_path: Path to the robot URDF file for IK solving
            end_effector_name: Name of the end effector link
            vis_target: Name of the mocap target body in MuJoCo scene
            floating_base: Whether the robot has a floating base
            R_headset_world: Rotation matrix to transform headset coordinates to world coordinates
            visualize_placo: Whether to visualize the placo IK solver
            scale_factor: Scale factor for teleoperation
        """
        self.xml_path = xml_path
        self.robot_urdf_path = robot_urdf_path
        self.end_effector_name = end_effector_name
        self.vis_target = vis_target
        self.floating_base = floating_base

        # To be initialized later
        self.mj_model = None
        self.mj_data = None
        self.placo_robot = None
        self.solver = None
        self.effector_task = None
        self.dt = None
        self.t = 0
        self.target_mocap_idx = -1

        # Initial poses
        self.init_ee_xyz = None
        self.init_ee_quat = None
        self.init_controller_xyz = None
        self.init_controller_quat = None

        # Coordinate transformation
        self.R_headset_world = R_headset_world

        # placo visualization
        self.visualize_placo = visualize_placo
        self.scale_factor = scale_factor

    def initialize(self):
        """Set up the MuJoCo simulation and the IK solver."""
        # Load MuJoCo model
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        # Configure scene lighting
        self.mj_model.vis.headlight.ambient = [0.4, 0.4, 0.4]
        self.mj_model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
        self.mj_model.vis.headlight.specular = [0.6, 0.6, 0.6]

        # Set initial position from keyframe if available
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        try:
            mujoco.mj_resetDataKeyframe(
                self.mj_model, self.mj_data, self.mj_model.key("home").id
            )
        except Exception as e:
            print(f"Warning: Could not reset to 'home' keyframe: {e}")

        # Calculate forward kinematics to get initial end effector pose
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Setup placo IK solver
        self._setup_placo()

        # Setup mocap target
        self._setup_mocap_target()

        return self

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

    def _setup_placo(self):
        """Set up the placo inverse kinematics solver."""
        self.dt = self.mj_model.opt.timestep
        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.effector_task = self.solver.add_frame_task(
            self.end_effector_name, np.eye(4)
        )
        self.effector_task.configure(self.end_effector_name, "soft", 1.0)
        self.solver.dt = self.dt
        if self.floating_base:
            self.placo_robot.state.q = self.mj_data.qpos.copy()
        else:
            self.solver.mask_fbase(True)
            self.placo_robot.state.q[7:] = self.mj_data.qpos.copy()
        self.placo_robot.update_kinematics()
        if self.visualize_placo:
            self.placo_vis = robot_viz(self.placo_robot)
            self.placo_vis.display(self.placo_robot.state.q)
            robot_frame_viz(self.placo_robot, self.end_effector_name)
            frame_viz("target", self.effector_task.T_world_frame)

    def _setup_mocap_target(self):
        """Find and set up the mocap target body."""
        mocap_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.vis_target
        )
        if mocap_id == -1:
            raise ValueError(
                f"Mocap body '{self.vis_target}' not found in the model."
            )

        if self.mj_model.body_mocapid[mocap_id] == -1:
            raise ValueError(
                f"Body '{self.vis_target}' is not configured for mocap."
            )
        else:
            self.target_mocap_idx = self.mj_model.body_mocapid[mocap_id]

        print(f"Mocap ID for '{self.vis_target}' body: {self.target_mocap_idx}")

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

                # Get XR controller data
                xr_grip = xr.get_right_grip()
                active = xr_grip > 0.5

                # Process current pose
                if active:
                    if self.init_ee_xyz is None:
                        self.init_ee_xyz, self.init_ee_quat = (
                            self._get_end_effector_info(self.end_effector_name)
                        )
                    xr_pose = xr.get_right_controller_pose()
                    delta_xyz, delta_rot = self._process_xr_pose(xr_pose)

                    target_xyz, target_quat = apply_delta_pose(
                        self.init_ee_xyz,
                        self.init_ee_quat,
                        delta_xyz,
                        delta_rot,
                    )

                    target = tf.quaternion_matrix(target_quat)
                    target[:3, 3] = target_xyz

                    self.effector_task.T_world_frame = target
                    self._update_kinematics()

                    # Update mocap body pose
                    if self.target_mocap_idx != -1:
                        self.mj_data.mocap_pos[self.target_mocap_idx] = (
                            target_xyz
                        )

                        self.mj_data.mocap_quat[self.target_mocap_idx] = (
                            target_quat
                        )
                else:
                    self.init_controller_xyz = None
                    self.init_controller_quat = None
                    self.init_ee_quat = None
                    self.init_ee_xyz = None

                # Step simulation and update viewer
                mujoco.mj_step(self.mj_model, self.mj_data)
                viewer.sync()

    def _process_xr_pose(self, xr_pose):
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

        if self.init_controller_xyz is None:
            # First time processing the pose
            self.init_controller_xyz = controller_xyz
            self.init_controller_quat = controller_quat

            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])
        else:
            # Calculate relative transformation from init pose
            delta_xyz = (
                controller_xyz - self.init_controller_xyz
            ) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(
                self.init_controller_quat, controller_quat
            )

        return delta_xyz, delta_rot

    def _update_kinematics(self):
        # Solve IK
        self.solver.solve(True)
        self.placo_robot.update_kinematics()
        q = self.placo_robot.state.q

        # Update robot control
        if self.floating_base:
            self.mj_data.ctrl = q
        else:
            self.mj_data.ctrl = q[7:]

        if self.visualize_placo:
            self.placo_vis.display(self.placo_robot.state.q)
            robot_frame_viz(self.placo_robot, self.end_effector_name)
            frame_viz("target", self.effector_task.T_world_frame)
