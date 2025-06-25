import threading

import rospy

from xrobotoolkit_teleop.common.xr_client import XrClient
from xrobotoolkit_teleop.hardware.galaxea_teleop_controller import GalaxeaDualA1XTeleopController


def main():
    xr_client = XrClient()
    controller = GalaxeaDualA1XTeleopController(xr_client=xr_client)

    stop_event = threading.Event()
    rospy.on_shutdown(lambda: stop_event.set())
    left_arm_thread = threading.Thread(
        target=controller.run_left_arm_control_thread,
        args=(stop_event,),
    )

    right_arm_thread = threading.Thread(
        target=controller.run_right_arm_control_thread,
        args=(stop_event,),
    )

    ik_thread = threading.Thread(
        target=controller.run_ik_thread,
        args=(stop_event,),
    )

    left_arm_thread.start()
    right_arm_thread.start()
    ik_thread.start()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
    finally:
        stop_event.set()

    left_arm_thread.join()
    right_arm_thread.join()
    ik_thread.join()


if __name__ == "__main__":
    main()
