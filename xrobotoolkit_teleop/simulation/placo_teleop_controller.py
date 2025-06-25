import time
from typing import Any, Dict

from xrobotoolkit_teleop.common.base_teleop_controller import BaseTeleopController
from xrobotoolkit_teleop.utils.geometry import (
    R_HEADSET_TO_WORLD,
)


class PlacoTeleopController(BaseTeleopController):
    """
    Placo teleoperation controller for a robot using inverse kinematics.

    This controller allows teleoperation of a robot by mapping XR controller poses
    to end effector movements using the Placo IK solver.
    """

    def __init__(
        self,
        robot_urdf_path: str,
        end_effector_config: Dict[str, Dict[str, Any]],
        floating_base=False,
        R_headset_world=R_HEADSET_TO_WORLD,
        scale_factor=1.0,
        q_init=None,
        dt=0.01,
    ):
        super().__init__(
            robot_urdf_path,
            end_effector_config,
            floating_base,
            R_headset_world,
            scale_factor,
            q_init,
            dt,
        )

    def _robot_setup(self):
        """
        In this case the robot is a visualization of the Placo robot.
        """
        self._init_placo_viz()

    def _send_command(self):
        self._update_placo_viz()

    def _cleanup(self):
        pass

    def _update_robot_state(self):
        pass

    def run(self):
        """
        Run the main teleoperation loop.
        This method is inherited from BaseTeleopController and implements the teleoperation logic.
        """
        while not self._stop_event.is_set():
            try:
                start_time = time.time()
                self._update_ik()
                self._send_command()
                end_time = time.time()
                time.sleep(max(0, self.dt - (end_time - start_time)))
            except KeyboardInterrupt:
                print("\nTeleoperation stopped.")
                self._stop_event.set()
