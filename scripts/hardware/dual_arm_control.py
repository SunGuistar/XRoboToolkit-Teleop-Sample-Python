from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware import DualArmURController


def main():
    xr_client = XrClient()
    controller = DualArmURController(xr_client)
    controller.run()


if __name__ == "__main__":
    main()
