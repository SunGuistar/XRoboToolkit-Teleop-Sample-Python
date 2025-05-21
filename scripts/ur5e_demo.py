import os
from teleop_demo_mujoco.mujoco_teleop_controller import MujocoTeleopController
from teleop_demo_mujoco.path_utils import ASSET_PATH


def main():
    xml_path = os.path.join(ASSET_PATH, "universal_robots_ur5e/scene.xml")
    robot_urdf_path = os.path.join(
        ASSET_PATH, "universal_robots_ur5e/ur5e.urdf"
    )

    # Create and initialize the teleoperation controller
    controller = MujocoTeleopController(
        xml_path=xml_path,
        robot_urdf_path=robot_urdf_path,
        end_effector_name="tool0",
        vis_target="target",
        visualize_placo=True,
    )

    controller.initialize()
    controller.xr_init()
    controller.run()


if __name__ == "__main__":
    main()
