import os

from xrobotoolkit_teleop.simulation.placo_teleop_controller import (
    PlacoTeleopController,
)
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH


def main():

    # 14 dof: only arms
    # robot_urdf_path = os.path.join(ASSET_PATH, "X7S/X7S_arm_only.urdf")

    # 17 dof: 2 arms, body tilting (joint2), head (joint3 and joint4)
    # robot_urdf_path = os.path.join(
    #     ASSET_PATH, "X7S/X7S_fixed_vertical_no_gripper.urdf"
    # )

    # 22 dof: all joints available
    robot_urdf_path = os.path.join(ASSET_PATH, "arx/X7S/X7S.urdf")

    config = {
        "right_hand": {
            "link_name": "link20",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
        },
        "left_hand": {
            "link_name": "link11",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
        },
    }

    # Create and initialize the teleoperation controller
    controller = PlacoTeleopController(
        robot_urdf_path=robot_urdf_path,
        end_effector_config=config,
        scale_factor=1.5,
    )

    # additional constraints hardcoded here for now
    joints_task = controller.solver.add_joints_task()
    joints_task.set_joints({joint: 0.0 for joint in controller.placo_robot.joint_names()})
    joints_task.configure("joints_regularization", "soft", 5e-4)

    if "joint2" in controller.placo_robot.joint_names():
        controller.placo_robot.set_joint_limits("joint2", -0.5, 0.1)  # to avoid excessive tilting of torso
        controller.solver.enable_velocity_limits(True)

    controller.run()


if __name__ == "__main__":
    main()
