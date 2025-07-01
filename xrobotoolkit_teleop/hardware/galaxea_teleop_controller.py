import os
import threading
import time
from typing import Dict

import cv2
import meshcat.transformations as tf
import numpy as np
import rospy

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.hardware.interface.galaxea import A1XController
from xrobotoolkit_teleop.hardware.interface.realsense import RealSenseCameraInterface
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH

# Default paths and configurations for a single right arm
DEFAULT_SINGLE_A1X_URDF_PATH = os.path.join(ASSET_PATH, "galaxea/A1X/a1x.urdf")
DEFAULT_DUAL_A1X_URDF_PATH = os.path.join(ASSET_PATH, "galaxea/A1X/dual_a1x.urdf")
DEFAULT_SCALE_FACTOR = 1.0
CONTROLLER_DEADZONE = 0.1

DEFAULT_LEFT_WRIST_CAM_SERIAL = "218622272014"
DEFAULT_RIGHT_WRIST_CAM_SERIAL = "218622272499"
DEFAULT_BASE_CAM_SERIAL = "215222077461"

CAM_SERIAL_DICT = {
    "left_wrist": DEFAULT_LEFT_WRIST_CAM_SERIAL,
    "right_wrist": DEFAULT_RIGHT_WRIST_CAM_SERIAL,
    "base": DEFAULT_BASE_CAM_SERIAL,
}

DEFAULT_END_EFFECTOR_CONFIG = {
    "left_arm": {
        "link_name": "gripper_link",
        "pose_source": "right_controller",
        "control_trigger": "right_grip",
        "gripper_config": {
            "type": "parallel",
            "gripper_trigger": "right_trigger",
            "joint_names": [
                "right_gripper_finger_joint1",
            ],
            "open_pos": [
                -2.0,
            ],
            "close_pos": [
                0.0,
            ],
        },
    },
}

DEFAULT_DUAL_END_EFFECTOR_CONFIG = {
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
                -2.0,
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
                -2.0,
            ],
            "close_pos": [
                0.0,
            ],
        },
    },
}


class GalaxeaA1XTeleopController(BaseTeleopController):
    def __init__(
        self,
        robot_urdf_path: str = DEFAULT_DUAL_A1X_URDF_PATH,
        end_effector_config: dict = DEFAULT_DUAL_END_EFFECTOR_CONFIG,
        R_headset_world: np.ndarray = R_HEADSET_TO_WORLD,
        scale_factor: float = DEFAULT_SCALE_FACTOR,
        visualize_placo: bool = False,
        ros_rate_hz: int = 100,
        enable_log_data: bool = True,
        log_dir: str = "logs/galaxea",
        log_freq: float = 50,
        enable_camera: bool = True,
        camera_serial_dict: Dict[str, str] = CAM_SERIAL_DICT,
        camera_width: int = 424,
        camera_height: int = 240,
        camera_fps: int = 60,
        enable_camera_depth: bool = False,
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
        if self.visualize_placo:
            self._init_placo_viz()

        self.camera_interface = None
        self.camera_serial_dict = camera_serial_dict
        if enable_camera:
            print("Initializing camera...")
            try:
                self.camera_interface = RealSenseCameraInterface(
                    width=camera_width,
                    height=camera_height,
                    fps=camera_fps,
                    serial_numbers=camera_serial_dict.values(),
                    enable_depth=enable_camera_depth,
                )
                self.camera_interface.start()
                self.camera_interface.update_frames()
                print("Camera initialized successfully.")
            except Exception as e:
                print(f"Error initializing camera: {e}")
                self.camera_interface = None

        self._prev_b_button_state = False
        self._is_logging = False

    def _placo_setup(self):
        super()._placo_setup()
        for arm_name, config in self.end_effector_config.items():
            ee_link_name = config["link_name"]
            arm_prefix = ee_link_name.replace("gripper_link", "")
            arm_joint_names = [f"{arm_prefix}arm_joint{i}" for i in range(1, 7)]
            self.placo_arm_joint_slice[arm_name] = slice(
                self.placo_robot.get_joint_offset(arm_joint_names[0]),
                self.placo_robot.get_joint_offset(arm_joint_names[-1]) + 1,
            )

    def _robot_setup(self):
        rospy.init_node("galaxea_teleop_controller", anonymous=True)

        self.arm_controllers: Dict[str, A1XController] = {}
        self.placo_arm_joint_slice: Dict[str, slice] = {}

        for arm_name, config in self.end_effector_config.items():
            arm_prefix = arm_name.replace("_arm", "")
            controller = A1XController(
                arm_control_topic=f"/motion_control/control_arm_{arm_prefix}",
                gripper_control_topic=f"/motion_control/control_gripper_{arm_prefix}",
                arm_state_topic=f"/hdas/feedback_arm_{arm_prefix}",
                rate_hz=1.0 / self.dt,
            )
            self.arm_controllers[arm_name] = controller

        print("Waiting for initial joint states from Galaxea arms...")
        all_controllers_ready = False
        while not rospy.is_shutdown() and not all_controllers_ready:
            all_controllers_ready = all(controller.timestamp > 0 for controller in self.arm_controllers.values())
            rospy.sleep(0.1)
        print("All controllers received initial state.")

    def _update_robot_state(self):
        """Reads current joint states from all hardware controllers and updates Placo."""
        for arm_name, controller in self.arm_controllers.items():
            self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]] = controller.qpos

    def _send_command(self):
        """Sends the solved joint targets to the hardware controllers."""
        for arm_name, controller in self.arm_controllers.items():
            if self.active.get(arm_name, False):
                controller.q_des = self.placo_robot.state.q[self.placo_arm_joint_slice[arm_name]].copy()

            if "gripper_config" in self.end_effector_config[arm_name]:
                controller.q_des_gripper = [
                    self.gripper_pos_target[arm_name][gripper_joint]
                    for gripper_joint in self.gripper_pos_target[arm_name].keys()
                ]

            controller.publish_arm_control()
            controller.publish_gripper_control()

    def _get_link_pose(self, link_name: str):
        """Gets the current world pose for a given link name from Placo."""
        T_world_link = self.placo_robot.get_T_world_frame(link_name)
        pos = T_world_link[:3, 3]
        quat = tf.quaternion_from_matrix(T_world_link)
        return pos, quat

    def _log_data(self):
        if self.enable_log_data:
            """
            Logs the current state of the robot, including joint positions, end effector poses,
            and any other relevant data.
            """

            timestamp = time.time() - self._start_time
            data_entry = {
                "timestamp": timestamp,
                "qpos": {arm: controller.qpos for arm, controller in self.arm_controllers.items()},
                "qvel": {arm: controller.qvel for arm, controller in self.arm_controllers.items()},
                "qpos_des": {arm: controller.q_des for arm, controller in self.arm_controllers.items()},
                "gripper_qpos": {arm: controller.qpos_gripper for arm, controller in self.arm_controllers.items()},
                "gripper_qpos_des": {arm: controller.q_des_gripper for arm, controller in self.arm_controllers.items()},
            }
            if self.camera_interface:
                cam_dict = {}
                frames_dict = self.camera_interface.get_frames()
                for name, serial in self.camera_serial_dict.items():
                    frame_data = frames_dict[serial]
                    if frame_data:
                        cam_dict[name] = frame_data
                if cam_dict:
                    data_entry["image"] = cam_dict
            self.data_logger.add_entry(data_entry)

    def _ik_thread(self, stop_event: threading.Event):
        """Dedicated thread for running the IK solver."""
        rate = rospy.Rate(1.0 / self.dt)
        while not stop_event.is_set():
            self._update_robot_state()
            self._update_gripper_target()
            self._update_ik()
            if self.visualize_placo:
                self._update_placo_viz()
            rate.sleep()
        print("IK loop has stopped.")

    def _control_thread(self, stop_event: threading.Event):
        """Dedicated thread for sending commands to hardware."""
        rate = rospy.Rate(self.ros_rate_hz)  # High-frequency control loop
        while not stop_event.is_set():
            self._send_command()
            rate.sleep()
        print("Control loop has stopped.")

    def _data_logging_thread(self, stop_event: threading.Event):
        rate = rospy.Rate(self.log_freq)
        while not stop_event.is_set():
            if self.enable_log_data:
                self._check_logging_button()
                if self._is_logging:
                    self._log_data()
            rate.sleep()
        # if self._is_logging:
        #     self.data_logger.save()  # Save data when logging stops
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

        print("Camera streaming started.")
        window_name = "RealSense Cameras"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

        try:
            while not stop_event.is_set():
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
                cv2.waitKey(int(1000.0 / self.camera_interface.fps))
        finally:
            self.camera_interface.stop()
            cv2.destroyAllWindows()
            print("Camera thread has stopped.")

    def run(self):
        """
        Main entry point that starts the multi-threaded IK and control loops.
        """
        self._start_time = time.time()  # Record the start time for logging
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

        if self.camera_interface:
            camera_thread = threading.Thread(target=self._camera_thread, args=(self._stop_event,))
            camera_thread.daemon = True
            camera_thread.start()

        print("Teleoperation running. Press Ctrl+C to exit.")
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
            if self.camera_interface:
                camera_thread.join(timeout=2.0)
            print("All threads have been shut down.")
