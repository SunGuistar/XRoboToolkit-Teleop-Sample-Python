import os

from teleop_demo_python.simulation.mujoco_teleop_controller import (
    MujocoTeleopController,
)
from teleop_demo_python.utils.path_utils import ASSET_PATH


def main():
    xml_path = os.path.join(ASSET_PATH, "galaxea/A1X/dual_a1x.xml")
    robot_urdf_path = os.path.join(ASSET_PATH, "galaxea/A1X/dual_a1x.urdf")

    config = {
        "right_hand": {
            "link_name": "right_arm_link6",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
        },
        "left_hand": {
            "link_name": "left_arm_link6",
            "pose_source": "left_controller",
            "control_trigger": "left_grip",
        },
    }

    # Create and initialize the teleoperation controller
    controller = MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        end_effector_config=config,
        scale_factor=1.5,
        visualize_placo=True,
    )

    controller.run()


if __name__ == "__main__":
    main()
