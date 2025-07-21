import threading
import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage, Image

from .base_camera import BaseCameraInterface


class RosCameraInterface(BaseCameraInterface):
    """
    An interface to handle one or more cameras through ROS topics.
    """

    def __init__(self, camera_topics: dict, enable_depth: bool = True):
        """
        Initializes the ROS camera interface.

        Args:
            camera_topics (dict): A dictionary where keys are camera names and
                                  values are another dictionary with 'color' and 'depth' topic names.
                                  e.g., {"left": {"color": "/topic/left_color", "depth": "/topic/left_depth"}}
            enable_depth (bool): Whether to enable the depth stream.
        """
        self.camera_topics = camera_topics
        self.enable_depth = enable_depth
        self.bridge = CvBridge()

        self.frames_dict = {}
        self.frames_lock = threading.Lock()
        self.subscribers = []

    def start(self):
        """Starts the camera subscribers."""
        for name, topics in self.camera_topics.items():
            if "color" not in topics:
                print(f"Warning: 'color' topic not specified for camera {name}. Skipping.")
                continue

            color_topic = topics["color"]
            self.subscribers.append(
                rospy.Subscriber(
                    color_topic,
                    CompressedImage,
                    self._color_callback,
                    callback_args=name,
                )
            )

            if self.enable_depth and "depth" in topics:
                depth_topic = topics["depth"]
                self.subscribers.append(
                    rospy.Subscriber(
                        depth_topic, Image, self._depth_callback, callback_args=name
                    )
                )
            print(f"Subscribed to topics for camera: {name}")

    def _color_callback(self, msg, camera_name):
        np_arr = np.frombuffer(msg.data, np.uint8)
        color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        with self.frames_lock:
            if camera_name not in self.frames_dict:
                self.frames_dict[camera_name] = {}
            self.frames_dict[camera_name]["color"] = color_image

    def _depth_callback(self, msg, camera_name):
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with self.frames_lock:
            if camera_name not in self.frames_dict:
                self.frames_dict[camera_name] = {}
            self.frames_dict[camera_name]["depth"] = depth_image

    def stop(self):
        """Stops the camera subscribers."""
        for sub in self.subscribers:
            sub.unregister()
        print("Unregistered all camera subscribers.")

    def update_frames(self):
        """
        No-op for ROS-based interface, as frames are updated by callbacks.
        """
        pass

    def get_frames(self):
        """
        Returns the last received frames from all cameras.

        Returns:
            dict: A dictionary of frames.
        """
        with self.frames_lock:
            return self.frames_dict.copy()

    def get_frame(self, camera_name: str):
        """
        Returns the last received frame from a specific camera.

        Args:
            camera_name (str): The name of the camera.

        Returns:
            dict: A dictionary containing 'color' and 'depth' numpy arrays.
        """
        with self.frames_lock:
            return self.frames_dict.get(camera_name, {}).copy()
