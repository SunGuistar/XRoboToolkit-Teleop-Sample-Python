import os
import threading
import time
from typing import Dict

import cv2
import meshcat.transformations as tf
import numpy as np

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.hardware.interface.arx_r5 import ARXR5Interface
from xrobotoolkit_teleop.hardware.interface.realsense import RealSenseCameraInterface
from xrobotoolkit_teleop.utils.geometry import R_HEADSET_TO_WORLD
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

# Default paths and configurations for ARX R5
DEFAULT_ARX_R5_URDF_PATH = os.path.join(ASSET_PATH, "arx/R5a/R5a.urdf")
DEFAULT_DUAL_ARX_R5_URDF_PATH = os.path.join(ASSET_PATH, "arx/R5a/dual_R5a.urdf")
DEFAULT_SCALE_FACTOR = 1.0
CONTROLLER_DEADZONE = 0.1

# Default camera configuration
DEFAULT_RIGHT_WRIST_CAM_SERIAL = "218622272014"
DEFAULT_LEFT_WRIST_CAM_SERIAL = "218622272499"
DEFAULT_BASE_CAM_SERIAL = "215222077461"

CAM_SERIAL_DICT = {
    "left_wrist": DEFAULT_LEFT_WRIST_CAM_SERIAL,
    "right_wrist": DEFAULT_RIGHT_WRIST_CAM_SERIAL,
    "base": DEFAULT_BASE_CAM_SERIAL,
}

DEFAULT_CAN_PORTS = {
    "left_arm": "can1",
    "right_arm": "can3",
}

# Default end-effector configuration for a single ARX R5 arm
DEFAULT_ARX_R5_END_EFFECTOR_CONFIG = {
    "right_arm": {  # Using "right_arm" for consistency with base controller
        "link_name": "link6",  # URDF link name for the end-effector
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": ["joint7"],
            "open_pos": [4.9],
            "close_pos": [0.0],
        },
    },
}

DEFAULT_DUAL_ARX_R5_END_EFFECTOR_CONFIG = {
    "right_arm": {  # Using "right_arm" for consistency with base controller
        "link_name": "right_link6",  # URDF link name for the end-effector
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": ["right_joint7"],
            "open_pos": [4.9],
            "close_pos": [0.0],
        },
    },
    "left_arm": {  # Using "left_arm" for consistency with base controller
        "link_name": "left_link6",  # URDF link name for the end-effector
        "pose_source": "left_controller",
        "control_trigger": "left_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "left_trigger",
            "joint_names": ["left_joint7"],
            "open_pos": [4.9],
            "close_pos": [0.0],
        },
    },
}


class ARXR5TeleopController(BaseTeleopController):
    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_DUAL_ARX_R5_URDF_PATH,
        end_effector_config: dict = DEFAULT_DUAL_ARX_R5_END_EFFECTOR_CONFIG,
        can_ports: Dict[str, str] = DEFAULT_CAN_PORTS,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = False,
        control_rate_hz: int = 50,
        enable_log_data: bool = True,
        log_dir: str = "logs/arx_r5",
        log_freq: float = 50,
        enable_camera: bool = True,
        camera_serial_dict: Dict[str, str] = CAM_SERIAL_DICT,
        camera_width: int = 424,
        camera_height: int = 240,
        camera_fps: int = 60,
        enable_camera_depth: bool = False,
    ):
        self.can_ports = can_ports
        self.control_rate_hz = control_rate_hz

        # Initialize camera interface
        self.camera_interface = None
        self.camera_serial_dict = camera_serial_dict
        if enable_camera:
            print("Initializing camera...")
            try:
                self.camera_interface = RealSenseCameraInterface(
                    width=camera_width,
                    height=camera_height,
                    fps=camera_fps,
                    serial_numbers=list(camera_serial_dict.values()),
                    enable_depth=enable_camera_depth,
                )
                self.camera_interface.start()
                print("Camera initialized successfully.")
            except Exception as e:
                print(f"Error initializing camera: {e}")
                self.camera_interface = None

        super().__init__(
            robot_urdf_path=robot_urdf_path,
            end_effector_config=end_effector_config,
            floating_base=False,
            R_headset_world=R_headset_world,
            scale_factor=scale_factor,
            q_init=None,
            dt=1.0 / control_rate_hz,
            enable_log_data=enable_log_data,
            log_dir=log_dir,
            log_freq=log_freq,
        )

        self._start_time = 0

        self.visualize_placo = visualize_placo
        if self.visualize_placo:
            self._init_placo_viz()

        self._prev_b_button_state = False
        self._is_logging = False

    def _placo_setup(self):
        super()._placo_setup()
        self.placo_arm_joint_slice: Dict[str, slice] = {}
        for arm_name, config in self.end_effector_config.items():
            ee_link_name = config["link_name"]
            arm_prefix = ee_link_name.replace("link6", "")
            arm_joint_names = [f"{arm_prefix}joint{i}" for i in range(1, 7)]
            self.placo_arm_joint_slice[arm_name] = slice(
                self.placo_robot.get_joint_offset(arm_joint_names[0]),
                self.placo_robot.get_joint_offset(arm_joint_names[-1]) + 1,
            )

    def _robot_setup(self):
        """Initializes the ARX R5 hardware interfaces for both arms."""
        self.arm_controllers: Dict[str, ARXR5Interface] = {}
        for arm_name, can_port in self.can_ports.items():
            print(f"Setting up ARX R5 {arm_name} on CAN port: {can_port}")
            arm = ARXR5Interface(can_port=can_port, dt=1.0 / self.control_rate_hz)
            self.arm_controllers[arm_name] = arm

        print("Going to home position...")
        for arm in self.arm_controllers.values():
            arm.go_home()

        time.sleep(1)  # Wait for the arms to reach home
        print("Arms are at home.")

    def _update_robot_state(self):
        """Reads current joint states from the arms and updates Placo."""
        for arm_name, controller in self.arm_controllers.items():
            q_slice = self.placo_arm_joint_slice[arm_name]
            self.placo_robot.state.q[q_slice] = controller.get_joint_positions()[:6]

    def _send_command(self):
        """Sends the solved joint targets to the hardware controllers."""
        for arm_name, controller in self.arm_controllers.items():
            if self.active.get(arm_name, False):
                q_des = self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]].copy()
                controller.set_joint_positions(q_des)

            if "gripper_config" in self.end_effector_config[arm_name]:
                gripper_config = self.end_effector_config[arm_name]["gripper_config"]
                joint_name = gripper_config["joint_names"][0]  # Assume first joint is representative
                gripper_target = self.gripper_pos_target[arm_name][joint_name]
                controller.set_catch_pos(gripper_target)

    def _get_link_pose(self, link_name: str):
        """Gets the current world pose for a given link name from Placo."""
        T_world_link = self.placo_robot.get_T_world_frame(link_name)
        pos = T_world_link[:3, 3]
        quat = tf.quaternion_from_matrix(T_world_link)
        return pos, quat

    def _log_data(self):
        """Logs the current state of the robot and camera."""
        if not self.enable_log_data:
            return

        timestamp = time.time() - self._start_time
        data_entry = {
            "timestamp": timestamp,
            "qpos": {arm: c.get_joint_positions() for arm, c in self.arm_controllers.items()},
            "qvel": {arm: c.get_joint_velocities() for arm, c in self.arm_controllers.items()},
            "qpos_des": {
                arm: self.placo_robot.state.q[self.placo_arm_joint_slice[arm]].copy() for arm in self.arm_controllers
            },
            "gripper_target": {
                arm: self.gripper_pos_target[arm].copy() if "gripper_config" in self.end_effector_config[arm] else None
                for arm in self.arm_controllers
            },
        }

        if self.camera_interface:
            cam_dict = {}
            frames = self.camera_interface.get_frames()
            for name, serial in self.camera_serial_dict.items():
                if serial in frames:
                    cam_dict[name] = frames[serial]
            if cam_dict:
                data_entry["image"] = cam_dict

        self.data_logger.add_entry(data_entry)

    def _ik_thread(self, stop_event: threading.Event):
        """Dedicated thread for running the IK solver."""
        while not stop_event.is_set():
            start_time = time.time()
            self._update_robot_state()
            self._update_gripper_target()
            self._update_ik()
            if self.visualize_placo:
                self._update_placo_viz()
            elapsed_time = time.time() - start_time
            if elapsed_time < 1.0 / self.control_rate_hz:
                time.sleep(1.0 / self.control_rate_hz - elapsed_time)
        print("IK loop has stopped.")

    def _control_thread(self, stop_event: threading.Event):
        """Dedicated thread for sending commands to hardware."""
        while not stop_event.is_set():
            start_time = time.time()
            self._send_command()
            elapsed_time = time.time() - start_time
            if elapsed_time < 1.0 / self.control_rate_hz:
                time.sleep(1.0 / self.control_rate_hz - elapsed_time)

        for _, controller in self.arm_controllers.items():
            controller.go_home()
        time.sleep(1)
        print("Control loop has stopped.")

    def _data_logging_thread(self, stop_event: threading.Event):
        """Dedicated thread for data logging."""
        while not stop_event.is_set():
            start_time = time.time()
            self._check_logging_button()
            if self._is_logging:
                self._log_data()
            elapsed_time = time.time() - start_time
            if elapsed_time < 1.0 / self.log_freq:
                time.sleep(1.0 / self.log_freq - elapsed_time)
        print("Data logging thread has stopped.")

    def _check_logging_button(self):
        """Checks for the 'B' button press to toggle data logging."""
        # Assuming 'b_button' is available from the right controller state
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

    def _camera_thread(self, stop_event: threading.Event):
        """Dedicated thread for managing the camera lifecycle and streaming."""
        if not self.camera_interface:
            return

        print("Camera thread started, waiting for logging to be enabled.")
        window_name = "RealSense Cameras"
        window_created = False

        try:
            while not stop_event.is_set():
                if self._is_logging:
                    if not window_created:
                        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
                        window_created = True

                    self.camera_interface.update_frames()
                    frames_dict = self.camera_interface.get_frames()
                    all_camera_rows = []

                    # Use the configured camera dictionary to maintain a consistent order
                    for serial in self.camera_serial_dict.values():
                        if serial not in frames_dict:
                            continue

                        frames = frames_dict[serial]
                        images_in_row = []

                        color_image = frames.get("color")
                        if color_image is not None:
                            # Ensure color image is 3-channel for stacking with colormap
                            if len(color_image.shape) == 2:
                                color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)
                            images_in_row.append(color_image)

                        depth_image = frames.get("depth")
                        if depth_image is not None:
                            depth_colormap = cv2.applyColorMap(
                                cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                            )
                            images_in_row.append(depth_colormap)

                        if images_in_row:
                            all_camera_rows.append(np.hstack(images_in_row))

                    if all_camera_rows:
                        # Pad rows to the same width for vertical stacking
                        max_width = max(row.shape[1] for row in all_camera_rows)
                        padded_rows = []
                        for row in all_camera_rows:
                            if row.shape[1] < max_width:
                                # Assuming 3 channels for color/colormap and consistent height
                                padding = np.zeros((row.shape[0], max_width - row.shape[1], 3), dtype=np.uint8)
                                padded_row = np.hstack([row, padding])
                                padded_rows.append(padded_row)
                            else:
                                padded_rows.append(row)

                        # Vertically stack all camera rows into a single image
                        combined_image = np.vstack(padded_rows)
                        cv2.imshow(window_name, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

                    # Process GUI events
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    if window_created:
                        cv2.destroyWindow(window_name)
                        window_created = False
                    # Sleep to avoid busy-waiting when not logging
                    time.sleep(0.1)
        finally:
            self.camera_interface.stop()
            if window_created:
                cv2.destroyAllWindows()
            print("Camera thread has stopped.")

    def run(self):
        """Main entry point that starts all threads."""
        self._start_time = time.time()
        self._stop_event = threading.Event()
        threads = []

        # Start core threads
        ik_thread = threading.Thread(target=self._ik_thread, args=(self._stop_event,))
        control_thread = threading.Thread(target=self._control_thread, args=(self._stop_event,))
        threads.extend([ik_thread, control_thread])

        # Start optional threads
        if self.enable_log_data:
            log_thread = threading.Thread(target=self._data_logging_thread, args=(self._stop_event,))
            threads.append(log_thread)
        if self.camera_interface:
            camera_thread = threading.Thread(target=self._camera_thread, args=(self._stop_event,))
            threads.append(camera_thread)

        for t in threads:
            t.daemon = True
            t.start()

        print("Teleoperation running. Press Ctrl+C to exit.")
        try:
            while not self._stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received.")
        finally:
            print("Shutting down...")
            self._stop_event.set()
            for t in threads:
                t.join(timeout=2.0)
            print("All threads have been shut down.")
