import tyro
from xrobotoolkit_teleop.hardware.galaxea_r1_lite_teleop_controller import (
    DEFAULT_DUAL_A1X_URDF_PATH,
    DEFAULT_END_EFFECTOR_CONFIG,
    GalaxeaR1LiteTeleopController,
)


def main(
    robot_urdf_path: str = DEFAULT_DUAL_A1X_URDF_PATH,
    scale_factor: float = 1.5,
    enable_log_data: bool = True,
    visualize_placo: bool = False,
    control_rate_hz: int = 100,
    log_dir: str = "logs/galaxea_r1_lite",
):
    """
    Main function to run the Galaxea R1 Lite teleoperation.
    """
    controller = GalaxeaR1LiteTeleopController(
        robot_urdf_path=robot_urdf_path,
        end_effector_config=DEFAULT_END_EFFECTOR_CONFIG,
        scale_factor=scale_factor,
        enable_log_data=enable_log_data,
        visualize_placo=visualize_placo,
        control_rate_hz=control_rate_hz,
        log_dir=log_dir,
    )
    controller.run()


if __name__ == "__main__":
    tyro.cli(main)
