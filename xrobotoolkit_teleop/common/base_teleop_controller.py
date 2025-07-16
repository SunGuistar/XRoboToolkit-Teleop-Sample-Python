import abc
import threading
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

from xrobotoolkit_teleop.common.data_logger import DataLogger
from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.utils.geometry import (
    apply_delta_pose,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.parallel_gripper_utils import (
    calc_parallel_gripper_position,
)


class BaseTeleopController(abc.ABC):
    def __init__(
        self,
        robot_urdf_path: str,
        end_effector_config: Dict[str, Dict[str, Any]],
        floating_base: bool,
        R_headset_world: np.ndarray,
        scale_factor: float,
        q_init: np.ndarray,
        dt: float,
        enable_log_data: bool = False,
        log_dir: str = "logs",
        log_freq: float = 50,
    ):
        self.robot_urdf_path = robot_urdf_path
        self.end_effector_config = end_effector_config
        self.floating_base = floating_base
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.q_init = q_init
        self.dt = dt
        self.xr_client = XrClient()

        self.enable_log_data = enable_log_data
        self.log_dir = log_dir
        self.log_freq = log_freq
        if enable_log_data:
            self.data_logger = DataLogger(log_dir=log_dir)

        # Initial poses
        self.ref_ee_xyz = {name: None for name in end_effector_config.keys()}
        self.ref_ee_quat = {name: None for name in end_effector_config.keys()}
        self.ref_controller_xyz = {name: None for name in end_effector_config.keys()}
        self.ref_controller_quat = {name: None for name in end_effector_config.keys()}
        self.effector_task = {}
        self.active = {}
        self.gripper_pos_target = {}
        for name, config in end_effector_config.items():
            if "gripper_config" in config:
                gripper_config = config["gripper_config"]
                self.gripper_pos_target[name] = {
                    joint_name: joint_pos
                    for joint_name, joint_pos in zip(gripper_config["joint_names"], gripper_config["open_pos"])
                }

        self._stop_event = threading.Event()

        self._robot_setup()
        self._placo_setup()

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

        if self.ref_controller_xyz[src_name] is None:
            self.ref_controller_xyz[src_name] = controller_xyz
            self.ref_controller_quat[src_name] = controller_quat

            delta_xyz = np.zeros(3)
            delta_rot = np.array([0.0, 0.0, 0.0])
        else:
            delta_xyz = (controller_xyz - self.ref_controller_xyz[src_name]) * self.scale_factor
            delta_rot = quat_diff_as_angle_axis(self.ref_controller_quat[src_name], controller_quat)

        return delta_xyz, delta_rot

    def _placo_setup(self):
        """Set up the placo inverse kinematics solver."""
        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        print("Joint names in the Placo model:")
        for joint_name in self.placo_robot.model.names:
            print(f"  {joint_name}")

        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = self.dt
        # self.solver.add_kinetic_energy_regularization_task(1e-6)

        # Set initial configuration
        if self.q_init is not None:
            if self.floating_base:
                self.placo_robot.state.q = self.q_init.copy()
            else:
                self.solver.mask_fbase(True)
                self.placo_robot.state.q[7:] = self.q_init.copy()
        else:
            if not self.floating_base:
                self.solver.mask_fbase(True)
            self.placo_robot.state.q[:7] = np.array([0, 0, 0, 0, 0, 0, 1])  # Identity quaternion for base

        self.placo_robot.update_kinematics()

        # Set up end effector tasks
        for name, config in self.end_effector_config.items():
            ee_xyz, ee_quat = self._get_link_pose(config["link_name"])
            ee_target = tf.quaternion_matrix(ee_quat)
            ee_target[:3, 3] = ee_xyz
            self.effector_task[name] = self.solver.add_frame_task(config["link_name"], ee_target)
            self.effector_task[name].configure(name, "soft", 1.0)
            manipulability = self.solver.add_manipulability_task(config["link_name"], "both", 1.0)
            manipulability.configure("manipulability", "soft", 1e-2)

        self.placo_robot.update_kinematics()

    def _update_ik(self):
        """
        This is the core IK logic block. It reads from XR, updates Placo tasks,
        and solves the kinematics.
        """
        self._update_robot_state()
        self.placo_robot.update_kinematics()

        for src_name, config in self.end_effector_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            self.active[src_name] = xr_grip_val > 0.9

            if self.active[src_name]:
                if self.ref_ee_xyz[src_name] is None:
                    print(f"{src_name} is activated.")
                    self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name] = self._get_link_pose(config["link_name"])

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot = self._process_xr_pose(xr_pose, src_name)
                target_xyz, target_quat = apply_delta_pose(
                    self.ref_ee_xyz[src_name], self.ref_ee_quat[src_name], delta_xyz, delta_rot
                )
                target_pose = tf.quaternion_matrix(target_quat)
                target_pose[:3, 3] = target_xyz
                self.effector_task[src_name].T_world_frame = target_pose
            else:
                if self.ref_ee_xyz[src_name] is not None:
                    print(f"{src_name} is deactivated.")
                    self.ref_ee_xyz[src_name] = None
                    self.ref_controller_xyz[src_name] = None

        try:
            self.solver.solve(True)
        except RuntimeError as e:
            print(f"IK solver failed: {e}")

    def _init_placo_viz(self):
        self.placo_vis = robot_viz(self.placo_robot)
        webbrowser.open(self.placo_vis.viewer.url())
        self.placo_vis.display(self.placo_robot.state.q)
        for name, config in self.end_effector_config.items():
            robot_frame_viz(self.placo_robot, config["link_name"])
            frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

    def _update_placo_viz(self):
        self.placo_vis.display(self.placo_robot.state.q)
        for name, config in self.end_effector_config.items():
            robot_frame_viz(self.placo_robot, config["link_name"])
            frame_viz(f"vis_target_{name}", self.effector_task[name].T_world_frame)

    def _update_gripper_target(self):
        for gripper_name in self.end_effector_config.keys():
            if "gripper_config" not in self.end_effector_config[gripper_name]:
                continue

            gripper_config = self.end_effector_config[gripper_name]["gripper_config"]
            gripper_config = self.end_effector_config[gripper_name]["gripper_config"]
            gripper_type = gripper_config["type"]
            if gripper_type == "parallel":
                trigger_value = self.xr_client.get_key_value_by_name(gripper_config["gripper_trigger"])
                for joint_name, open_pos, close_pos in zip(
                    gripper_config["joint_names"],
                    gripper_config["open_pos"],
                    gripper_config["close_pos"],
                ):
                    # Calculate the target position based on the trigger value
                    gripper_pos = calc_parallel_gripper_position(open_pos, close_pos, trigger_value)
                    self.gripper_pos_target[gripper_name][joint_name] = gripper_pos
                    self.gripper_pos_target[gripper_name][joint_name] = gripper_pos
            else:
                # TODO: add dexterous hand support
                raise ValueError(f"Unsupported gripper type: {gripper_type}")

    def _log_data(self):
        """
        Logs the current state of the robot, including joint positions, end effector poses,
        and any other relevant data
        """
        if self.enable_log_data:
            raise NotImplementedError

    # ---------------------------------------------------------
    # --- Abstract Methods (to be implemented by subclasses) ---
    # ---------------------------------------------------------

    @abc.abstractmethod
    def _robot_setup(self):
        """Initializes the specific backend (connects to robot, starts sim, etc.)."""
        raise NotImplementedError

    @abc.abstractmethod
    def _update_robot_state(self):
        """Reads the current joint states from the robot/sim and updates self.placo_robot.state.q."""
        raise NotImplementedError

    @abc.abstractmethod
    def _send_command(self):
        """Sends the calculated target joint positions from self.placo_robot.state.q to the robot/sim."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_link_pose(self, link_name):
        """Gets the current world pose for a given link name."""
        raise NotImplementedError

    @abc.abstractmethod
    def run(self):
        """
        The main entry point. Subclasses must implement this to define their
        execution model (single-threaded or multi-threaded).
        """
        raise NotImplementedError
