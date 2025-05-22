from typing import Optional
import typer
import mujoco
from mujoco import viewer as mj_viewer
import numpy as np


def main():
    model_path = "assets/universal_robots_ur5e/dual_ur5e.xml"
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)

    viewer = mj_viewer.launch_passive(model, data)

    while True:
        mujoco.mj_step(model, data)
        viewer.sync()
    viewer.close()


if __name__ == "__main__":
    main()
