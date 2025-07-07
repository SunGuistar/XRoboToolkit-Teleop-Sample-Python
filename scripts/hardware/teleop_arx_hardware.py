from xrobotoolkit_teleop.hardware.arx_r5_teleop_controller import (
    DEFAULT_ARX_R5_END_EFFECTOR_CONFIG,
    DEFAULT_ARX_R5_URDF_PATH,
    ARXR5TeleopController,
)


def main():
    """
    Main function to run the ARX R5 teleoperation.
    """
    controller = ARXR5TeleopController(
        robot_urdf_path=DEFAULT_ARX_R5_URDF_PATH,
        end_effector_config=DEFAULT_ARX_R5_END_EFFECTOR_CONFIG,
        scale_factor=1.5,
        enable_camera=True,
        enable_log_data=True,
        can_ports={"right_arm": "can3"},
        visualize_placo=False,
        control_rate_hz=50,
        log_dir="logs/arx_r5",
    )
    controller.run()


if __name__ == "__main__":
    main()
