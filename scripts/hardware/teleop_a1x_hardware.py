import tyro
from xrobotoolkit_teleop.hardware.galaxea_teleop_controller import (
    DEFAULT_END_EFFECTOR_CONFIG,
    DEFAULT_SINGLE_A1X_URDF_PATH,
    GalaxeaA1XTeleopController,
)


def main(
    robot_urdf_path: str = DEFAULT_SINGLE_A1X_URDF_PATH,
    scale_factor: float = 1.5,
    enable_camera: bool = True,
    enable_log_data: bool = True,
    visualize_placo: bool = False,
    control_rate_hz: int = 100,
    log_dir: str = "logs/galaxea_a1x",
):
    """
    Main function to run the Galaxea A1X teleoperation.
    """
    controller = GalaxeaA1XTeleopController(
        robot_urdf_path=robot_urdf_path,
        end_effector_config=DEFAULT_END_EFFECTOR_CONFIG,
        scale_factor=scale_factor,
        enable_camera=enable_camera,
        enable_log_data=enable_log_data,
        visualize_placo=visualize_placo,
        control_rate_hz=control_rate_hz,
        log_dir=log_dir,
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
