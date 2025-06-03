from teleop_demo_python.hardware.dynamixel import DynamixelHeadController
from teleop_demo_python.utils.pico_client import PicoClient

import time
import threading


def main():
    pico_client = PicoClient()

    controller = DynamixelHeadController(pico_client)

    stop_signal = threading.Event()

    control_thread = threading.Thread(
        target=controller.run_thread,
        args=(stop_signal,),
    )
    control_thread.start()

    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping head control...")
            stop_signal.set()
            control_thread.join()
            break


if __name__ == "__main__":
    main()
