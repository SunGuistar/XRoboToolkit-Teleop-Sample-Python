import tyro
from xrobotoolkit_teleop.hardware.galaxea_teleop_controller import (
    DEFAULT_DUAL_A1X_URDF_PATH,
    DEFAULT_DUAL_END_EFFECTOR_CONFIG,
    GalaxeaA1XTeleopController,
)


def main(
    robot_urdf_path: str = DEFAULT_DUAL_A1X_URDF_PATH,
    scale_factor: float = 1.5,
    log_freq: int = 10,
    log_dir: str = "logs/galaxea_with_command",
    enable_log_data: bool = True,
    enable_camera: bool = True,
):
    """
    Main function to run the dual Galaxea A1X teleoperation.
    """
    controller = GalaxeaA1XTeleopController(
        robot_urdf_path=robot_urdf_path,
        end_effector_config=DEFAULT_DUAL_END_EFFECTOR_CONFIG,
        scale_factor=scale_factor,
        log_freq=log_freq,
        log_dir=log_dir,
        enable_log_data=enable_log_data,
        enable_camera=enable_camera,
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)