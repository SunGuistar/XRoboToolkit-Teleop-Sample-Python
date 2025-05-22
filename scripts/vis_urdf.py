import pinocchio
import placo
import argparse
import meshcat.transformations as tf
from placo_utils.visualization import (
    robot_viz,
    robot_frame_viz,
)


def main():
    urdf_path = "assets/universal_robots_ur5e/dual_ur5e.urdf"
    robot = placo.RobotWrapper(urdf_path)
    robot.update_kinematics()

    viz = robot_viz(robot)
    while True:
        viz.display(robot.state.q)
        robot_frame_viz(robot, "right_base_link")
        robot_frame_viz(robot, "left_base_link")
        robot_frame_viz(robot, "right_tool0")
        robot_frame_viz(robot, "left_tool0")


if __name__ == "__main__":
    main()
