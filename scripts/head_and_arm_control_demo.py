import argparse
import threading
import time

from teleop_demo_python.hardware import (
    DualArmURController,
    DynamixelHeadController,
)
from teleop_demo_python.utils.pico_client import PicoClient


def main():
    parser = argparse.ArgumentParser(description="Run head and dual arm teleoperation control.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Run the reset procedure for the dual arm controller.",
    )
    args = parser.parse_args()

    pico_client = PicoClient()
    head_controller = DynamixelHeadController(pico_client)
    arm_controller = DualArmURController(pico_client, visualize_placo=False)

    if args.reset:
        print("Reset flag detected. Running arm controller reset procedure...")
        try:
            arm_controller.reset()
            print("Arm controller reset procedure completed.")
        except Exception as e:
            print(f"Error during arm_controller.reset(): {e}")
    else:
        print("No reset flag detected. Proceeding with normal operation.")
        arm_controller.calc_target_joint_position()

        stop_signal = threading.Event()
        head_thread = threading.Thread(
            target=head_controller.run_thread,
            args=(stop_signal,),
        )
        left_arm_thread = threading.Thread(
            target=arm_controller.run_left_controller_thread,
            args=(stop_signal,),
        )
        right_arm_thread = threading.Thread(
            target=arm_controller.run_right_controller_thread,
            args=(stop_signal,),
        )
        ik_thread = threading.Thread(
            target=arm_controller.run_ik_thread,
            args=(stop_signal,),
        )

        # Start the threads
        head_thread.start()
        left_arm_thread.start()
        right_arm_thread.start()
        ik_thread.start()

        while not stop_signal.is_set():
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                print("KeyboardInterrupt detected. Exiting...")
                stop_signal.set()  # Trigger the stop signal for all threads

        head_thread.join()
        left_arm_thread.join()
        right_arm_thread.join()
        ik_thread.join()

    head_controller.close()
    arm_controller.close()
    print("All controllers have been stopped and disconnected.")


if __name__ == "__main__":
    main()
