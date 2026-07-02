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
from xrobotoolkit_teleop.hardware.interface.dh_pgc_gripper import DHPGCGripperTeleopInterface
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
from xrobotoolkit_teleop.bc.config import BCConfig
from xrobotoolkit_teleop.bc.gripper_norm import trigger_to_gripper_cmd
from xrobotoolkit_teleop.bc.policy import BCPolicy, apply_policy_action
from xrobotoolkit_teleop.hardware.soft_constraint import (
    DEFAULT_DZ_DAMP,
    DEFAULT_KR,
    DEFAULT_KZ,
    SoftConstraintTeleopHelper,
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

FLANGE_CONTROLLER_MANIPULATOR_CONFIG = {
    "arm": {
        "link_name": "flange",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_trigger": "right_trigger",
    },
}

LEFT_FLANGE_CONTROLLER_MANIPULATOR_CONFIG = {
    "arm": {
        "link_name": "flange",
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_trigger": "left_trigger",
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
        use_dh_gripper: bool = False,
        dh_gripper_port: str = "/dev/ttyUSB0",
        dh_gripper_force: int = 50,
        dh_gripper_speed: int = 50,
        dh_gripper_slave_id: int = 1,
        dh_gripper_baudrate: int = 115200,
        dh_gripper_standard_position: bool = False,
        policy_mode: bool = False,
        bc_config: BCConfig | None = None,
        enable_soft_constraint: bool = False,
        soft_constraint_kz: float = DEFAULT_KZ,
        soft_constraint_kr: float = DEFAULT_KR,
        soft_constraint_dz_damp: float = DEFAULT_DZ_DAMP,
    ):
        self.xr_client = xr_client
        self.robot_urdf_path = robot_urdf_path
        self.R_headset_world = R_headset_world
        self.scale_factor = scale_factor
        self.visualize_placo = visualize_placo
        self.manipulator_config = manipulator_config
        self.arm_name = next(iter(manipulator_config.keys()))
        self.use_gripper = use_gripper
        self.use_dh_gripper = use_dh_gripper
        self.dh_gripper = None
        self.robot_controller = None

        if use_gripper and use_dh_gripper:
            raise ValueError("use_gripper (Robotiq) and use_dh_gripper (PGC-300-60) are mutually exclusive.")

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
            use_gripper=use_gripper and not use_dh_gripper,
        )

        self._dh_gripper_kwargs = None
        if use_dh_gripper:
            self._dh_gripper_kwargs = dict(
                port=dh_gripper_port,
                baudrate=dh_gripper_baudrate,
                slave_id=dh_gripper_slave_id,
                force=dh_gripper_force,
                speed=dh_gripper_speed,
                standard_position_mapping=dh_gripper_standard_position,
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
        self._control_lock = threading.Lock()
        self.control_active = False
        self._servo_engaged = False
        self._pending_gripper_trigger: float = 0.0

        # BC deployment hooks (V1: recording uses sample_bc_frame; policy optional)
        self.policy_mode = policy_mode
        self.bc_config = bc_config or BCConfig()
        self.policy: BCPolicy | None = None
        self._policy_image: np.ndarray | None = None
        self._policy_image_lock = threading.Lock()
        self._reset_in_progress = False
        self.enable_soft_constraint = enable_soft_constraint
        self._soft_constraint = SoftConstraintTeleopHelper(
            kz=soft_constraint_kz,
            kr=soft_constraint_kr,
            dz_damp=soft_constraint_dz_damp,
        )
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

    def _get_processed_controller_pose(self, xr_pose) -> tuple[np.ndarray, np.ndarray]:
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
        return controller_xyz, controller_quat

    def _resync_cumulative_teleop_after_soft_off(self) -> None:
        for name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            if xr_grip_val > (1.0 - CONTROLLER_DEADZONE):
                T_world_ee = self.placo_robot.get_T_world_frame(config["link_name"])
                self.init_ee_xyz[name] = T_world_ee[:3, 3].copy()
                self.init_ee_quat[name] = tf.quaternion_from_matrix(T_world_ee)
                self.init_controller_xyz[name] = None
                self.init_controller_quat[name] = None

    def _update_soft_constraint_buttons(self) -> None:
        if not self.enable_soft_constraint:
            return
        self._soft_constraint.update_buttons(
            a_pressed=self.xr_client.get_button_state_by_name("A"),
            b_pressed=self.xr_client.get_button_state_by_name("B"),
            placo_robot=self.placo_robot,
            resync_active_grip=self._resync_cumulative_teleop_after_soft_off,
        )

    def _compute_soft_constraint_target(
        self,
        name: str,
        config: dict,
        xr_pose,
        init_xyz: np.ndarray,
        init_quat: np.ndarray,
    ) -> np.ndarray:
        delta_xyz, delta_rot = self._process_xr_pose(xr_pose, name)
        target_xyz, target_quat = apply_delta_pose(init_xyz, init_quat, delta_xyz, delta_rot)
        t_user = tf.quaternion_matrix(target_quat)
        t_user[:3, 3] = target_xyz
        return self._soft_constraint.project_target(t_user)

    def set_policy(self, policy: BCPolicy | None) -> None:
        """Attach a BC policy for deployment (policy_mode must be True)."""
        self.policy = policy

    def set_policy_image(self, image_bgr: np.ndarray) -> None:
        """Update the latest main-camera frame for policy inference."""
        with self._policy_image_lock:
            self._policy_image = image_bgr.copy()

    def get_gripper_cmd_normalized(self) -> float:
        """VR trigger as gripper_cmd. 0=open, 1=close."""
        config = self.manipulator_config[self.arm_name]
        trigger_val = self.xr_client.get_key_value_by_name(config["gripper_trigger"])
        return trigger_to_gripper_cmd(trigger_val)

    def get_gripper_state_normalized(self) -> float:
        """Estimated gripper opening. 0=open, 1=close."""
        if self.dh_gripper is not None:
            return self.dh_gripper.get_normalized_state()
        if self.use_gripper and self.robot_controller.gripper is not None:
            open_pos = self.robot_controller.gripper.get_open_position()
            closed_pos = self.robot_controller.gripper.get_closed_position()
            from xrobotoolkit_teleop.bc.gripper_norm import register_to_normalized

            return register_to_normalized(self.gripper_pos, open_pos, closed_pos)
        return self.get_gripper_cmd_normalized()

    def get_observation_state(self) -> np.ndarray:
        """BC observation.state = [qpos(6), gripper_state(1)]. 0=open, 1=close."""
        qpos = self.robot_controller.get_current_joint_positions()
        gripper_state = self.get_gripper_state_normalized()
        return np.concatenate([qpos, [gripper_state]]).astype(np.float64)

    def sample_bc_frame(self) -> dict:
        """Sample robot fields for BC recording (no image)."""
        with self._control_lock:
            target_q = self.target_q.copy()
            control_active = self.control_active
        return {
            "qpos": self.robot_controller.get_current_joint_positions().astype(np.float64),
            "target_q": target_q.astype(np.float64),
            "gripper_state": self.get_gripper_state_normalized(),
            "gripper_cmd": self.get_gripper_cmd_normalized(),
            "control_active": control_active,
        }

    def _run_policy_step(self, current_q: np.ndarray) -> bool:
        """Policy → target_q when policy_mode and human grip is not active."""
        if self.policy is None:
            return False
        with self._policy_image_lock:
            image = None if self._policy_image is None else self._policy_image.copy()
        if image is None:
            return False

        obs = {
            "observation.images.main": image,
            "observation.state": self.get_observation_state(),
        }
        action = self.policy.predict(obs)
        target_q, gripper_cmd = apply_policy_action(
            action,
            current_q,
            max_delta_q=self.bc_config.max_delta_q,
            joint_limits_lower=self.bc_config.joint_limits_lower,
            joint_limits_upper=self.bc_config.joint_limits_upper,
        )
        with self._control_lock:
            self.target_q = target_q
        self._pending_gripper_trigger = gripper_cmd
        return True

    def calc_target_joint_position(self):
        """Calculate target joint positions from PICO controller pose and grip input."""
        current_q = self.robot_controller.get_current_joint_positions()
        self.placo_robot.state.q[PLACO_ARM_Q_SLICE] = current_q
        self.placo_robot.update_kinematics()
        self._update_soft_constraint_buttons()

        human_control_active = False
        for name, config in self.manipulator_config.items():
            xr_grip_val = self.xr_client.get_key_value_by_name(config["control_trigger"])
            active = xr_grip_val > (1.0 - CONTROLLER_DEADZONE)
            human_control_active = human_control_active or active
            if self.use_gripper:
                trigger_val = self.xr_client.get_key_value_by_name(config["gripper_trigger"])
                self.gripper_pos = int(
                    calc_parallel_gripper_position(
                        self.robot_controller.gripper.get_open_position(),
                        self.robot_controller.gripper.get_closed_position(),
                        trigger_val,
                    )
                )
            elif self.use_dh_gripper and (active or not self.policy_mode):
                # In policy_mode, gripper_cmd comes from policy unless human has grip active.
                self._pending_gripper_trigger = self.xr_client.get_key_value_by_name(config["gripper_trigger"])

            if active:
                if self.init_ee_xyz[name] is None:
                    T_world_ee_current = self.placo_robot.get_T_world_frame(config["link_name"])
                    self.init_ee_xyz[name] = T_world_ee_current[:3, 3].copy()
                    self.init_ee_quat[name] = tf.quaternion_from_matrix(T_world_ee_current)
                    print(
                        f"{name} activated. Current EE xyz: {self.init_ee_xyz[name]}, "
                        f"quat: {self.init_ee_quat[name]}."
                    )
                    if self.enable_soft_constraint and self._soft_constraint.active:
                        self._soft_constraint.set_grip_start(
                            self.placo_robot.get_T_world_frame(config["link_name"])
                        )

                xr_pose = self.xr_client.get_pose_by_name(config["pose_source"])

                if self.enable_soft_constraint and self._soft_constraint.active:
                    target_transform = self._compute_soft_constraint_target(
                        name,
                        config,
                        xr_pose,
                        self.init_ee_xyz[name],
                        self.init_ee_quat[name],
                    )
                else:
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
                self._soft_constraint.reset_motion_state()
                T_world_ee_current = self.placo_robot.get_T_world_frame(config["link_name"])
                self.effector_task[name].T_world_frame = T_world_ee_current

        with self._control_lock:
            self.control_active = human_control_active

        # Human grip always wins over policy (safety / takeover).
        if not human_control_active and self.policy_mode:
            if self._run_policy_step(current_q):
                return

        if not human_control_active:
            with self._control_lock:
                self.target_q = current_q.copy()
            return

        try:
            self.solver.solve(True)
            with self._control_lock:
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

    def init_dh_gripper(self) -> bool:
        """Connect and initialize DH gripper (call after UR RTDE is up).

        Returns True if the gripper is ready, False if skipped or unavailable (arm-only).
        """
        if not self.use_dh_gripper or self._dh_gripper_kwargs is None:
            return False
        if self.dh_gripper is not None:
            return True
        port = self._dh_gripper_kwargs.get("port", "?")
        try:
            self.dh_gripper = DHPGCGripperTeleopInterface(**self._dh_gripper_kwargs)
            return True
        except Exception as e:
            print(
                f"WARNING: DH gripper on {port} unavailable ({e}). "
                "Continuing in arm-only mode."
            )
            self.use_dh_gripper = False
            self._dh_gripper_kwargs = None
            return False

    def reset(self):
        self.robot_controller.reset()

    def reset_to_home(self, open_gripper: bool = True) -> None:
        """Stop servoJ, moveJ to home, sync teleop state. Safe to call during teleop."""
        with self._control_lock:
            self._reset_in_progress = True

        try:
            if self._servo_engaged:
                self.robot_controller.servo_stop()
                self._servo_engaged = False

            self.robot_controller.reset()
            q_home = self.robot_controller.initial_joint_positions.copy()

            if open_gripper:
                if self.dh_gripper is not None:
                    self.dh_gripper.open_gripper()
                    self._pending_gripper_trigger = 0.0
                elif self.use_gripper and self.robot_controller.gripper is not None:
                    self.robot_controller.open_gripper()

            with self._control_lock:
                self.target_q = q_home.copy()
                self.control_active = False
            self.placo_robot.state.q[PLACO_ARM_Q_SLICE] = q_home
            self.placo_robot.update_kinematics()

            for name, config in self.manipulator_config.items():
                self.init_ee_xyz[name] = None
                self.init_ee_quat[name] = None
                self.init_controller_xyz[name] = None
                self.init_controller_quat[name] = None
                self.effector_task[name].T_world_frame = self.placo_robot.get_T_world_frame(config["link_name"])
            self._soft_constraint.clear_all()
        finally:
            with self._control_lock:
                self._reset_in_progress = False

    def close(self):
        if self.dh_gripper is not None:
            self.dh_gripper.disconnect()
            self.dh_gripper = None
        if self.robot_controller is not None:
            if self._servo_engaged:
                self.robot_controller.servo_stop()
                self._servo_engaged = False
            self.robot_controller.close()
            self.robot_controller = None

    def run_robot_thread(self, stop_event):
        print("Starting arm control thread...")
        servo_fail_reported = False
        while not stop_event.is_set():
            with self._control_lock:
                if self._reset_in_progress:
                    time.sleep(0.02)
                    continue
                q_cmd = self.target_q.copy()

            ok = self.robot_controller.servo_joints(q_cmd)
            self._servo_engaged = True
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

    def run_dh_gripper_thread(self, stop_event):
        """Modbus gripper I/O off the IK/RTDE path to avoid arm stutter."""
        print("Starting DH gripper control thread...")
        while not stop_event.is_set():
            if self.dh_gripper is not None:
                self.dh_gripper.update_from_trigger(self._pending_gripper_trigger)
            time.sleep(0.02)

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

