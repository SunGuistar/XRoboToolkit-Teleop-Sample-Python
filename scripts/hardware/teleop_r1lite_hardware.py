from xrobotoolkit_teleop.hardware.galaxea_r1_lite_teleop_controller import (
    DEFAULT_DUAL_A1X_URDF_PATH,
    DEFAULT_END_EFFECTOR_CONFIG,
    GalaxeaR1LiteTeleopController,
)


def main():
    controller = GalaxeaR1LiteTeleopController(
        robot_urdf_path=DEFAULT_DUAL_A1X_URDF_PATH,
        end_effector_config=DEFAULT_END_EFFECTOR_CONFIG,
        scale_factor=1.5,
        log_freq=50,
        enable_log_data=True,
    )
    controller.run()


if __name__ == "__main__":
    main()
