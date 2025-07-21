import tyro
from xrobotoolkit_teleop.hardware.galaxea_teleop_controller import (
    DEFAULT_END_EFFECTOR_CONFIG,
    DEFAULT_SINGLE_A1X_URDF_PATH,
    GalaxeaA1XTeleopController,
)


def main(
    robot_urdf_path: str = DEFAULT_SINGLE_A1X_URDF_PATH,
    scale_factor: float = 1.5,
):
    """
    Main function to run the Galaxea A1X teleoperation.
    """
    controller = GalaxeaA1XTeleopController(
        robot_urdf_path=robot_urdf_path,
        end_effector_config=DEFAULT_END_EFFECTOR_CONFIG,
        scale_factor=scale_factor,
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)