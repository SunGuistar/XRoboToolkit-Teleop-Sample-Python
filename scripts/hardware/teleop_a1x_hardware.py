from xrobotoolkit_teleop.hardware.galaxea_teleop_controller import (
    DEFAULT_END_EFFECTOR_CONFIG,
    DEFAULT_SINGLE_A1X_URDF_PATH,
    GalaxeaA1XTeleopController,
)


def main():
    controller = GalaxeaA1XTeleopController(
        robot_urdf_path=DEFAULT_SINGLE_A1X_URDF_PATH,
        end_effector_config=DEFAULT_END_EFFECTOR_CONFIG,
        scale_factor=1.5,
    )
    controller.run()


if __name__ == "__main__":
    main()
