import argparse

import meshcat.transformations as tf
import numpy as np
import pinocchio
import placo
from placo_utils.visualization import (
    robot_frame_viz,
    robot_viz,
)

from xrobotoolkit_teleop.hardware.ur import (
    LEFT_INITIAL_JOINT,
    RIGHT_INITIAL_JOINT,
)


def main():
    urdf_path = "assets/universal_robots_ur5e/dual_ur5e.urdf"
    robot = placo.RobotWrapper(urdf_path)
    robot.state.q[7:] = np.concatenate(
        (
            LEFT_INITIAL_JOINT,
            RIGHT_INITIAL_JOINT,
        )
    )  # Set initial joint positions for both arms
    robot.update_kinematics()

    print(f"left initial joint: {LEFT_INITIAL_JOINT}")
    print(f"right initial joint: {RIGHT_INITIAL_JOINT}")

    viz = robot_viz(robot)
    while True:
        viz.display(robot.state.q)
        robot_frame_viz(robot, "right_base_link")
        robot_frame_viz(robot, "left_base_link")
        robot_frame_viz(robot, "right_tool0")
        robot_frame_viz(robot, "left_tool0")


if __name__ == "__main__":
    main()
