import threading

import rospy

from xrobotoolkit_teleop.hardware.galaxea_teleop_controller import GalaxeaA1XTeleopController
from xrobotoolkit_teleop.utils.xr_client import XrClient


def main():
    xr_client = XrClient()
    controller = GalaxeaA1XTeleopController(xr_client=xr_client)

    stop_event = threading.Event()
    rospy.on_shutdown(lambda: stop_event.set())
    left_arm_thread = threading.Thread(
        target=controller.run_control_thread,
        args=(stop_event,),
    )

    ik_thread = threading.Thread(
        target=controller.run_ik_thread,
        args=(stop_event,),
    )

    left_arm_thread.start()
    ik_thread.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    finally:
        stop_event.set()

    left_arm_thread.join()
    ik_thread.join()


if __name__ == "__main__":
    main()
