import os
import threading
import time
import webbrowser

import meshcat.transformations as tf
import numpy as np
import placo
from placo_utils.visualization import (
    frame_viz,
    robot_frame_viz,
    robot_viz,
)

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.interface.universal_robots import (
    CONTROLLER_DEADZONE,
    DEFAULT_ROBOT_IP,
    GRIPPER_FORCE,
    GRIPPER_SPEED,
    LOOKAHEAD_TIME,
    MAX_ACCELERATION,
    MAX_VELOCITY,
    RIGHT_INITIAL_JOINT_DEG,
    SERVO_GAIN,
    SERVO_TIME,
    URController,
)
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
    apply_delta_pose,
    quat_diff_as_angle_axis,
)
from xrobotoolkit_teleop.utils.parallel_gripper_utils import (
    calc_parallel_gripper_position,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

DEFAULT_SINGLE_ARM_URDF_PATH = os.path.join(ASSET_PATH, "universal_robots_ur5e/ur5e.urdf")
DEFAULT_SCALE_FACTOR = 0.8
PLACO_ARM_Q_SLICE = slice(7, 13)

LEFT_CONTROLLER_MANIPULATOR_CONFIG = {
    "arm": {
        "link_name": "tool0",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_trigger": "left_trigger",
    },
}

RIGHT_CONTROLLER_MANIPULATOR_CONFIG = {
    "arm": {
        "link_name": "tool0",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_trigger": "right_trigger",
    },
}


class SingleArmURController:
    """Teleoperation controller for one UR5e arm driven by a PICO VR controller."""

    def __init__(
        self,
        xr_client: XrClient,
        robot_urdf_path: str = DEFAULT_SINGLE_ARM_URDF_PATH,
        manipulator_config: dict = RIGHT_CONTROLLER_MANIPULATOR_CONFIG,
        robot_ip: str = DEFAULT_ROBOT_IP,
        initial_joint_deg: np.ndarray = RIGHT_INITIAL_JOINT_DEG,
        max_velocity: float = MAX_VELOCITY,
        max_acceleration: float = MAX_ACCELERATION,
        servo_time: float = SERVO_TIME,
        lookahead_time: float = LOOKAHEAD_TIME,
        servo_gain: float = SERVO_GAIN,
        gripper_force: float = GRIPPER_FORCE,
        gripper_speed: float = GRIPPER_SPEED,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = False,
        use_gripper: bool = False,
    ):
        self.xr_client = xr_client
        self.robot_urdf_path = robot_urdf_path
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.visualize_placo = visualize_placo
        self.manipulator_config = manipulator_config
        self.arm_name = next(iter(manipulator_config.keys()))
        self.use_gripper = use_gripper
        self.robot_controller = None

        self.robot_controller = URController(
            robot_ip=robot_ip,
            initial_joint_positions=np.deg2rad(initial_joint_deg),
            max_velocity=max_velocity,
            max_acceleration=max_acceleration,
            servo_time=servo_time,
            lookahead_time=lookahead_time,
            servo_gain=servo_gain,
            gripper_force=gripper_force,
            gripper_speed=gripper_speed,
            use_gripper=use_gripper,
        )

        self.placo_robot = placo.RobotWrapper(self.robot_urdf_path)
        self.solver = placo.KinematicsSolver(self.placo_robot)
        self.solver.dt = servo_time
        self.solver.mask_fbase(True)
        self.solver.add_kinetic_energy_regularization_task(1e-6)

        self.effector_task = {}
        self.init_ee_xyz = {}
        self.init_ee_quat = {}
        self.init_controller_xyz = {}
        self.init_controller_quat = {}
        for name, config in self.manipulator_config.items():
            initial_pose = np.eye(4)
            self.effector_task[name] = self.solver.add_frame_task(config["link_name"], initial_pose)
            self.effector_task[name].configure(f"{name}_frame", "soft", 1.0)
            manipulability = self.solver.add_manipulability_task(config["link_name"], "both", 1.0)
            manipulability.configure(f"{name}_manipulability", "soft", 5e-2)
            self.init_ee_xyz[name] = None
            self.init_ee_quat[name] = None
            self.init_controller_xyz[name] = None
            self.init_controller_quat[name] = None

        q_init = self.robot_controller.get_current_joint_positions()
        self.gripper_pos = 0
        self.placo_robot.state.q[PLACO_ARM_Q_SLICE] = q_init
        self.placo_robot.update_kinematics()
        self.target_q = q_init.copy()
        for name, config in self.manipulator_config.items():
            self.effector_task[name].T_world_frame = self.placo_robot.get_T_world_frame(config["link_name"])

        if self.visualize_placo:
            self.placo_robot.update_kinematics()
            self.placo_vis = robot_viz(self.placo_robot)
            time.sleep(0.5)
            meshcat_url = self.placo_vis.viewer.url()
            print(f"Automatically opening meshcat at: {meshcat_url}")
            webbrowser.open(meshcat_url)
            self.placo_vis.display(self.placo_robot.state.q)
            for name, config in self.manipulator_config.items():
                robot_frame_viz(self.placo_robot, config["link_name"])
                frame_viz(
                    f"vis_target_{name}",
                    self.effector_task[name].T_world_frame,
                )

    def _process_xr_pose(self, xr_pose, arm_name: str):
        controller_xyz = np.array([xr_pose[0], xr_pose[1], xr_pose[2]])
        controller_quat = np.array(
            [
                xr_pose[6],
                xr_pose[3],
                xr_pose[4],
                xr_pose[5],
            ]
        )

        controller_xyz = self.R_headset_world @ controller_xyz

        R_transform = np.eye(4)
        R_transform[:3, :3] = self.R_headset_world
        R_quat = tf.quaternion_from_matrix(R_transform)
        controller_quat = tf.quaternion_multiply(
            tf.quaternion_multiply(R_quat, controller_quat),
            tf.quaternion_conjugate(R_quat),
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

    def calc_target_joint_position(self):
        """Calculate target joint positions from PICO controller pose and grip input."""
        current_q = self.robot_controller.get_current_joint_positions()
        self.placo_robot.state.q[PLACO_ARM_Q_SLICE] = current_q
        self.placo_robot.update_kinematics()

        for name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)
            if self.use_gripper:
                trigger_val = self.xr_client.get_key_value_by_name(config["gripper_trigger"])
                self.gripper_pos = int(
                    calc_parallel_gripper_position(
                        self.robot_controller.gripper.get_open_position(),
                        self.robot_controller.gripper.get_closed_position(),
                        trigger_val,
                    )
                )

            if active:
                if self.init_ee_xyz[name] is None:
                    T_world_ee_current = self.placo_robot.get_T_world_frame(config["link_name"])
                    self.init_ee_xyz[name] = T_world_ee_current[:3, 3].copy()
                    self.init_ee_quat[name] = tf.quaternion_from_matrix(T_world_ee_current)
                    print(
                        f"{name} activated. Current EE xyz: {self.init_ee_xyz[name]}, "
                        f"quat: {self.init_ee_quat[name]}."
                    )

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])
                delta_xyz, delta_rot_angle_axis = self._process_xr_pose(xr_pose, name)

                target_xyz, target_quat = apply_delta_pose(
                    self.init_ee_xyz[name],
                    self.init_ee_quat[name],
                    delta_xyz,
                    delta_rot_angle_axis,
                )

                target_transform = tf.quaternion_matrix(target_quat)
                target_transform[:3, 3] = target_xyz
                self.effector_task[name].T_world_frame = target_transform

            else:
                if self.init_ee_xyz[name] is not None:
                    print(f"{name} deactivated.")
                    self.init_ee_xyz[name] = None
                    self.init_ee_quat[name] = None
                    self.init_controller_xyz[name] = None
                    self.init_controller_quat[name] = None
                T_world_ee_current = self.placo_robot.get_T_world_frame(config["link_name"])
                self.effector_task[name].T_world_frame = T_world_ee_current

        try:
            self.solver.solve(True)
            self.target_q = self.placo_robot.state.q[PLACO_ARM_Q_SLICE].copy()

            if self.visualize_placo and hasattr(self, "placo_vis"):
                self.placo_vis.display(self.placo_robot.state.q)
                for name, config in self.manipulator_config.items():
                    robot_frame_viz(self.placo_robot, config["link_name"])
                    frame_viz(
                        f"vis_target_{name}",
                        self.effector_task[name].T_world_frame,
                    )

        except RuntimeError as e:
            print(f"IK solver failed: {e}. Returning last known good joint positions.")
        except Exception as e:
            print(f"An unexpected error occurred in IK: {e}. Returning last known good joint positions.")

    def reset(self):
        self.robot_controller.reset()

    def close(self):
        if self.robot_controller is not None:
            self.robot_controller.close()
            self.robot_controller = None

    def run_robot_thread(self, stop_event):
        print("Starting arm control thread...")
        servo_fail_reported = False
        while not stop_event.is_set():
            ok = self.robot_controller.servo_joints(self.target_q)
            if not ok and not servo_fail_reported:
                print(
                    "WARNING: servoJ rejected. Check Remote mode on the teach pendant "
                    "and that no other program holds RTDE control."
                )
                servo_fail_reported = True
            if self.use_gripper:
                self.robot_controller.gripper.move(
                    self.gripper_pos,
                    self.robot_controller.gripper_speed,
                    self.robot_controller.gripper_force,
                )
    def run_ik_thread(self, stop_event):
        print("Starting IK calculation thread...")
        while not stop_event.is_set():
            try:
                start = time.time()
                self.calc_target_joint_position()
                elapsed = time.time() - start
                time.sleep(max(0, 0.01 - elapsed))
            except Exception as e:
                print(f"Error in IK calculation: {e}")

