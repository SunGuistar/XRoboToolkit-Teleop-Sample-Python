from xrobotoolkit_teleop.hardware.galaxea_teleop_controller import (
    DEFAULT_DUAL_A1X_URDF_PATH,
    DEFAULT_DUAL_END_EFFECTOR_CONFIG,
    GalaxeaA1XTeleopController,
)


def main():
    controller = GalaxeaA1XTeleopController(
        robot_urdf_path=DEFAULT_DUAL_A1X_URDF_PATH,
        end_effector_config=DEFAULT_DUAL_END_EFFECTOR_CONFIG,
        scale_factor=1.5,
        log_freq=10,
        log_dir="logs/galaxea_with_command",
        enable_log_data=True,
        enable_camera=True,
    )
    controller.run()


if __name__ == "__main__":
    main()
