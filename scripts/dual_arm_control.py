from teleop_demo_python.hardware import DualArmURController
from teleop_demo_python.utils.pico_client import PicoClient


def main():
    pico_client = PicoClient()
    controller = DualArmURController(pico_client)
    controller.run()


if __name__ == "__main__":
    main()
