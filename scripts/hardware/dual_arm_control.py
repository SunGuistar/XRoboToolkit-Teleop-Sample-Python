from xrobotoolkit_teleop.hardware import DualArmURController
from xrobotoolkit_teleop.utils.xr_client import XrClient


def main():
    xr_client = XrClient()
    controller = DualArmURController(xr_client)
    controller.run()


if __name__ == "__main__":
    main()
