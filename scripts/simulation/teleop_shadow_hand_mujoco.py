import os

import mujoco
import numpy as np
import pinocchio as pin
from dex_retargeting.constants import HandType, RetargetingType, RobotName
from mujoco import viewer as mj_viewer

from xrobotoolkit_teleop.utils.dex_hand_utils import DexHandTracker, pico_hand_state_to_mediapipe
from xrobotoolkit_teleop.utils.mujoco_utils import calc_mujoco_ctrl_from_qpos, calc_mujoco_qpos_from_pin_q
from xrobotoolkit_teleop.utils.path_utils import ASSET_PATH
from xrobotoolkit_teleop.utils.xr_client import XrClient


def main():
    urdf_path = os.path.join(ASSET_PATH, "shadow_hand/shadow_hand_right.urdf")
    xml_path = os.path.join(ASSET_PATH, "shadow_hand/xml/scene_right.xml")

    mj_model = mujoco.MjModel.from_xml_path(xml_path)
    mj_data = mujoco.MjData(mj_model)
    pin_model = pin.buildModelFromUrdf(urdf_path)

    xr_client = XrClient()
    dextracker = DexHandTracker(
        robot_name=RobotName.shadow,
        urdf_path=urdf_path,
        hand_type=HandType.right,
        retargeting_type=RetargetingType.vector,
    )
    with mj_viewer.launch_passive(mj_model, mj_data) as viewer:
        # Set up viewer camera
        viewer.cam.azimuth = 180
        viewer.cam.elevation = -30
        viewer.cam.distance = 1.0
        viewer.cam.lookat = [0.2, 0, 0.2]

        while True:
            right_hand_state = np.array(xr_client.get_hand_tracking_state("right"))
            mediapipe_hand_state = pico_hand_state_to_mediapipe(right_hand_state)
            pin_q = dextracker.retarget(mediapipe_hand_state)
            mj_qpos = calc_mujoco_qpos_from_pin_q(mj_model, pin_model, pin_q)
            mj_ctrl = calc_mujoco_ctrl_from_qpos(mj_model, mj_qpos)
            mj_data.ctrl[:] = mj_ctrl
            mujoco.mj_step(mj_model, mj_data)
            viewer.sync()


if __name__ == "__main__":
    main()
