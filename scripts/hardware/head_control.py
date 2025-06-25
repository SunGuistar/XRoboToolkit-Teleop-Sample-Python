import threading
import time

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware import DynamixelHeadController


def main():
    xr_client = XrClient()

    controller = DynamixelHeadController(xr_client)

    stop_signal = threading.Event()

    control_thread = threading.Thread(
        target=controller.run_thread,
        args=(stop_signal,),
    )
    control_thread.start()

    while True:
        try:
            time.sleep(0.01)
        except KeyboardInterrupt:
            print("Stopping head control...")
            stop_signal.set()
            control_thread.join()
            break


if __name__ == "__main__":
    main()
