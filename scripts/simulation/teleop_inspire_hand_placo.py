import os
import webbrowser

import numpy as np
import placo
from dex_retargeting.constants import HandType, RetargetingType, RobotName
from placo_utils.visualization import (
    robot_viz,
)

from xrobotoolkit_teleop.utils.dex_hand_utils import DexHandTracker, pico_hand_state_to_mediapipe
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH
from xrobotoolkit_teleop.utils.xr_client import XrClient


def main():
    robot = placo.RobotWrapper(
        os.path.join(ASSET_PATH, "inspire_hand/inspire_hand_right.urdf"),
    )
    viz = robot_viz(robot, "Inspire Hand")
    webbrowser.open(viz.viewer.url())

    xr_client = XrClient()

    dextracker = DexHandTracker(
        robot_name=RobotName.inspire,
        urdf_path=os.path.join(ASSET_PATH, "inspire_hand/inspire_hand_right.urdf"),
        hand_type=HandType.right,
        retargeting_type=RetargetingType.vector,
    )

    while True:
        viz.display(robot.state.q)
        right_hand_state = np.array(xr_client.get_hand_tracking_state("right"))
        if np.all(right_hand_state == 0):
            print("all zero, ignore")
            continue
        mediapipe_hand_state = pico_hand_state_to_mediapipe(right_hand_state)
        qpos = dextracker.retarget(mediapipe_hand_state)
        robot.state.q[7:] = qpos


if __name__ == "__main__":
    main()
