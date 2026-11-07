# XRoboToolkit-Teleop-Sample-Python

Pico teleoperation demo written in python for both mujoco simulation and robot hardware.

> **版权说明 / Attribution**: 本项目改造自 [XR-Robotics/XRoboToolkit-Teleop-Sample-Python](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python)，在原始遥操作示例的基础上新增了行为克隆（BC）数据采集与策略部署、共享自主视觉辅助、RealSense VR 视频推流、单臂 UR5e / FAIRINO FR3 遥操作等功能，原始代码版权归 XR-Robotics 所有（MIT License）。
>
> This project is modified from [XR-Robotics/XRoboToolkit-Teleop-Sample-Python](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python). It builds on the original teleoperation sample with added behavior-cloning data collection and policy deployment, shared-autonomy vision assistance, RealSense VR streaming, and single-arm UR5e / FAIRINO FR3 teleoperation. Original code is copyright XR-Robotics (MIT License).

## Overview

This project provides a framework for controlling robots in robot hardware and MuJoCo simulation through XR (VR/AR) input devices. It allows users to manipulate robot arms using natural hand movements captured through XR controllers.

On top of the original teleoperation sample, this repository now also includes:

- **Behavior-cloning (BC) data collection and deployment pipeline** — record episodes (camera images + robot states + actions) directly during teleoperation at 10 Hz, batch-evaluate episode quality, convert to [LeRobot](https://github.com/huggingface/lerobot) datasets, and deploy trained policies back on the robot through a safety-filtered servo interface.
- **Shared-autonomy vision assistance** — classical-CV perception (HSV color thresholding + ROI + contour scoring, no deep learning) for desktop target detection, gripper detection, pixel alignment error overlay in VR, fixed-camera extrinsic calibration, depth sampling, and automatic hover-pose generation for `MoveL` approaches.
- **RealSense → VR video streaming** — stream one or two Intel RealSense cameras (H.264 over TCP) to the PICO headset in picture-in-picture layout, in the same process as teleoperation (the PC Service allows only one SDK client, so streaming and teleop share one `XrClient` connection).
- **Single-arm setups with dexterous hands** — single UR5e or FAIRINO FR3 arm teleoperation with DH-Robotics PGC-300-60 / Robotiq grippers (or a trigger-proxied virtual gripper), a Linker Hand L20 dexterous-hand interface with bring-up/diagnostic scripts, world-frame soft motion constraints (for wrench-turning style tasks), and cached home poses.
- **New robot support** — FAIRINO FR3 (法奥), Flexiv Rizon4s, Unitree G1, ARX A1X, plus a hardware interface layer (`xrobotoolkit_teleop/hardware/interface/`) wrapping RealSense cameras, Linker Hand L20, DH-Robotics and Robotiq grippers, Dynamixel, ARX R5, Galaxea R1 Lite, and UR robots.

## Installation
1. Download and install [XRoboToolkit PC Service](https://github.com/XR-Robotics/XRoboToolkit-PC-Service). Run the installed program before running the following demo.

2.  **Clone the repository:**
    ```bash
    git clone https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python.git
    cd XRoboToolkit-Teleop-Sample-Python
    ```

3.  **Installation**
    **Note:** The setup scripts are currently only tested on Ubuntu 22.04.
    It is recommended to setup a Conda environment and install the project using the included script.
    ```bash
    bash setup_conda.sh --conda <optional_env_name>
    conda activate <env_name>
    bash setup_conda.sh --install
    ```

    If installing on system python:
    ```bash
    bash setup.sh
    ```

    Newer features additionally require `pyrealsense2` (RealSense cameras), `tyro` (CLI), and `lerobot` (dataset conversion) — all included in `requirements_xrobot.txt`. The FAIRINO FR3 demo uses the official Fairino Python SDK in `FR3/fairino-python-sdk`, and the Linker Hand L20 uses its own Python SDK (CAN).

## Usage
Use the following instructions to run example scripts. For a more detailed description, please refer to [`teleop_details.md`](teleop_details.md).

### Running the MuJoCo Simulation Demo

To run the teleoperation demo with a UR5e robot in MuJoCo simulation:

```bash
python scripts/simulation/teleop_dual_ur5e_mujoco.py
```
This script initializes the [`MujocoTeleopController`](xrobotoolkit_teleop/simulation/mujoco_teleop_controller.py) with the UR5e model and starts the teleoperation loop.

Other simulation demos:
- Dual ARX A1X arms with parallel grippers in MuJoCo: `python scripts/simulation/teleop_dual_a1x_mujoco.py`
- Flexiv Rizon4s in MuJoCo: `python scripts/simulation/teleop_flexiv_rizon4s_mujoco.py`
- Tool/TCP rotation demo on UR5e in MuJoCo: `python scripts/simulation/tool_rotate_ur5e_mujoco.py`

### Running the Placo Visualization Demo

To run the teleoperation demo with a UR5e robot in Placo visualization:

```bash
python scripts/simulation/teleop_x7s_placo.py
```
This script initializes the [`PlacoTeleopController`](xrobotoolkit_teleop/simulation/placo_teleop_controller.py) with the X7S robot and starts the teleoperation loop.

Other Placo demos:
- Flexiv Rizon4s: `python scripts/simulation/teleop_flexiv_rizon4s_placo.py`
- Unitree G1 dual-arm humanoid: `python scripts/simulation/teleop_unitree_g1_placo.py`

### Running Dexterous Hand Teleop Simulation
- Shadow hand simulation in Mujoco
    ```bash
    python scripts/simulation/teleop_shadow_hand_mujoco.py
    ```

- Shadow hand in Placo visualization
    ```bash
    python scripts/simulation/teleop_shadow_hand_placo.py
    ```

- Inspire hand in Placo Visualization
    ```bash
    scripts/simulation/teleop_inspire_hand_placo.py
    ```

### Running the Hardware Demo (Dual UR5 Arms and Dynamixel-based Head)

To run the teleoperation demo with the physical dual UR arms and Dynamixel-based head:

1.  **Normal Operation:**
    ```bash
    python scripts/hardware/teleop_dual_ur5e_hardware.py
    ```
    This script initializes the [`DynamixelHeadController`](xrobotoolkit_teleop/hardware/dynamixel_head_controller.py) and [`DualArmURController`](xrobotoolkit_teleop/hardware/dual_arm_ur_controller.py) and starts the teleoperation loops for both head tracking and arm control.

2.  **Resetting Arm Positions:**
    If you need to reset the UR arms to their initial/home positions and initialize the robotiq grippers, you can run the script with the `--reset` flag:
    ```bash
    python scripts/hardware/teleop_dual_ur5e_hardware.py --reset
    ```
    This will execute the reset procedure defined in the [`DualArmURController`](xrobotoolkit_teleop/hardware/dual_arm_ur_controller.py) and then exit.

3.  **Visualizing IK results:**
    To visualize the inverse kinematics solution with placo during teleoperation, run the script with the `--visualize_placo` flag.
    ```bash
    python scripts/hardware/teleop_dual_ur5e_hardware.py --visualize_placo
    ```

### Running Single UR5e + Dexterous Hand Demo (with BC data collection)

Single-arm teleoperation of a UR5e in a dexterous-hand workcell (DH-Robotics PGC-300-60 or Robotiq gripper, or a trigger-proxied virtual gripper), RealSense cameras, and built-in BC episode recording:

```bash
python scripts/hardware/teleop_single_ur5e_hand.py
```

Useful flags:
- `--quality balanced` — higher resolution (auto-clamps fps for a D435 + D435I dual setup)
- `--no-pip` — main camera only (recorded episodes contain no wrist/PIP image)
- `--no-camera` — arm teleop only, no BC recording
- `--main-serial <SN> --pip-serial <SN>` — manually select RealSense cameras (by default connected cameras are auto-detected; D435I preferred as main, D435 as PIP)
- `--save-home` — save the current arm pose as the fixed Home pose (`teleop_single_ur5e_hand.home.json`); episodes auto-return to Home after saving

This script initializes the [`SingleArmURController`](xrobotoolkit_teleop/hardware/single_arm_ur_controller.py) and the [`BCEpisodeRecorder`](xrobotoolkit_teleop/bc/recorder.py); see [Behavior-Cloning Data Pipeline](#behavior-cloning-data-pipeline) below.

Button map (default right controller):
- **Hold Grip** — move the arm (human input always overrides the policy)
- **A** — enable world-frame soft constraint (lock z / rx / ry; only x / y / rz free) for wrench-turning style motions
- **B** — release the constraint and clear the reference
- **X** — start / stop BC episode recording (auto-saves and returns Home)
- **Y** — discard the current recording (and return Home)

### Running Single UR5e + RealSense VR Stream (teleop with live video in the headset)

Teleoperation and RealSense→VR streaming in one process (one XRoboToolkit SDK connection — running them in separate terminals would break arm tracking):

```bash
python scripts/hardware/teleop_single_ur5e_vr_stream.py \
  --main-serial 406122070449 \
  --pip-serial 148522072134 \
  --quality balanced \
  --pip-scale 0.38
```

- The stream protocol is TCP 4-byte length + H.264 NALU to the PICO headset (`ZEDMINI → Listen`); start the script first, then tap Listen on the headset when the terminal shows "waiting for Listen".
- **A** swaps main / PIP cameras; add `--debug` to overlay the shared-autonomy alignment visualization in the PIP corner.
- BC recording (X / Y) works the same as in `teleop_single_ur5e_hand.py`.

The standalone streamer (no teleop) is `python scripts/visualization/realsense_vr_stream.py`; see [RealSense VR Streaming](#realsense-vr-streaming).

### Running FAIRINO FR3 Hardware Demo

Single FAIRINO FR3 (法奥) arm teleoperation — a model port of the single-UR5e pipeline: the control chain (VR controller → XRoboToolkit → `SingleArmFR3Controller` → Placo IK → real-time joint servo) is identical, with RTDE `servoJ` replaced by the Fairino SDK `ServoJ` (`ServoMoveStart` / `ServoJ` / `ServoMoveEnd`, joint units in degrees) and joint states read from the CNDE status stream:

```bash
python FR3/teleop_single_fr3_hand.py
```

It supports the same flags as the UR5e single-arm demo (`--quality`, `--no-pip`, `--no-camera`, `--robot-ip`, `--save-home`, ...). FR3-related tooling in `FR3/`:
- `FR3/play_episode.py`, `FR3/play_episode_2.py` — replay recorded BC episodes on the arm via `ServoJ` / `MoveJ`
- `FR3/analyze_fr3_raw.py` — inspect raw FR3 BC episodes
- `FR3/evoRl.py` — deploy a policy from a remote EvoRL inference server over ZMQ (REQ/REP JSON: images + state in, target joint action out)
- `FR3/merge_rl_correction.py`, `FR3/pad_episode_frames.py` — merge failed/corrected RL episodes and append padding frames for retraining data

### Running ARX R5 Hardware Demo

To run the teleoperation demo with dual ARX R5 robotic arms:

```bash
python scripts/hardware/teleop_dual_arx_r5_hardware.py
```

Single ARX R5 arm variant: `python scripts/hardware/teleop_arx_hardware.py`.

This script initializes the [`ARXR5TeleopController`](xrobotoolkit_teleop/hardware/arx_r5_teleop_controller.py) for dual arm control with built-in grippers.

### Running Galaxea R1 Lite Humanoid Demo

To run the teleoperation demo with the Galaxea R1 Lite humanoid robot:

```bash
python scripts/hardware/teleop_r1lite_hardware.py
```

This script initializes the [`GalaxeaR1LiteTeleopController`](xrobotoolkit_teleop/hardware/galaxea_r1_lite_teleop_controller.py) for mobile manipulator control, the controller communicates with the robot hardware via ROS.

### DH-Robotics PGC-300-60 Gripper CLI

Command-line control of the DH-Robotics (大寰) PGC-300-60 gripper via RS485 or Modbus-RTU-over-UDP, with proportional VR-trigger-style control wrapped in [`dh_pgc_gripper.py`](xrobotoolkit_teleop/hardware/interface/dh_pgc_gripper.py):

```bash
python Jiazhua/pgc300_60_gripper.py --port /dev/ttyUSB0 --status
python Jiazhua/pgc300_60_gripper.py --udp-host <gateway-ip> --position 50 --force 50
```

## Behavior-Cloning Data Pipeline

The `xrobotoolkit_teleop/bc/` package implements a V1 pipeline for collecting imitation-learning data during teleoperation and deploying policies safely.

### Conventions

- `observation.state = [qpos(6), gripper_state(1)]` — gripper 0.0 = fully open, 1.0 = fully closed
- `action = [target_q(6), gripper_cmd(1)]` — gripper 0.0 = open, 1.0 = close
- UR5e joint limits from the datasheet are baked into [`bc/config.py`](xrobotoolkit_teleop/bc/config.py) (`BCConfig`)

### Recording episodes

Recording is built into the single-arm demos (`teleop_single_ur5e_hand.py`, `teleop_single_ur5e_vr_stream.py`, `FR3/teleop_single_fr3_hand.py`):

- A dual-thread recorder ([`BCEpisodeRecorder`](xrobotoolkit_teleop/bc/recorder.py)) samples at 10 Hz: one thread captures frames (main + PIP images, qpos, target_q, gripper state/cmd), a second thread encodes JPEG and accumulates; stopping writes the episode atomically to `logs/ur5e_bc/raw/episode_XXXXX.pkl` (or `logs/fr3_bc/raw/` for FR3).
- **X** starts/stops a recording (auto-save, then `moveJ` back to Home); **Y** discards the current episode.
- Gripper command is proxied from the VR trigger (no physical parallel gripper needed for the dexterous-hand setup).
- After saving, episodes can be renumbered contiguously:
  ```bash
  python scripts/renumber_ur5e_bc_episodes.py --raw-dir logs/ur5e_bc/raw
  ```

### Inspecting and evaluating episodes

```bash
# Replay / inspect a single raw episode (GUI or --no-gui)
python scripts/check_ur5e_bc_episode.py logs/ur5e_bc/raw/episode_00000.pkl

# Batch quality evaluation: timing, control coverage, pick-place gripper pattern,
# joint/action quality, frozen-camera detection (main vs pip); writes a JSON report
python scripts/evaluate_ur5e_bc_raw.py --raw-dir logs/ur5e_bc/raw --json-out logs/ur5e_bc/eval_report.json
```

### Converting to LeRobot

Convert raw episodes to a LeRobot v2 dataset (images stored as video, tabular fields as parquet):

```bash
python scripts/convert_ur5e_raw_to_lerobot.py \
  --raw-dir logs/ur5e_bc/raw \
  --repo-id local/ur5e_block_pick \
  --output-dir datasets/ur5e_block_pick

# Include failed episodes and idle frames:
python scripts/convert_ur5e_raw_to_lerobot.py --include-failed --include-inactive
```

Example converted datasets are included under `datasets/` (e.g. `ur5e_block_pick_120`, `ur5e_wrench_screw_111_lerobotV2`). This enables training with LeRobot pipelines and integration with openpi-style fine-tuning examples.

### Policy deployment

[`bc/policy.py`](xrobotoolkit_teleop/bc/policy.py) defines a `BCPolicy` protocol (`obs dict → 7-dim action`) and [`apply_policy_action`](xrobotoolkit_teleop/bc/policy.py), which routes every policy command through [`safety_filter_target_q`](xrobotoolkit_teleop/bc/safety.py): per-step joint delta clamping (`max_delta_q`) and joint-limit clipping before `servoJ`. A `NullPolicy` placeholder holds the current configuration; human teleop input always takes priority over the policy. A remote-policy example (ZMQ client to an EvoRL GPU server) is provided in `FR3/evoRl.py`.

## Shared Autonomy (Vision Assistance)

The `xrobotoolkit_teleop/shared_autonomy/` package provides classical-CV perception helpers (no deep learning, V1):

- **Target detection** ([`target_detection.py`](xrobotoolkit_teleop/shared_autonomy/target_detection.py)) — HSV color thresholding + normalized ROI + contour scoring with temporal EMA smoothing, tuned via [`shared_autonomy/config.py`](xrobotoolkit_teleop/shared_autonomy/config.py) (color presets: target / silver / white / black)
- **Gripper detection** ([`gripper_detection.py`](xrobotoolkit_teleop/shared_autonomy/gripper_detection.py)) and **pixel alignment error** ([`alignment.py`](xrobotoolkit_teleop/shared_autonomy/alignment.py)) between gripper and target
- **Camera extrinsic calibration** ([`camera_extrinsic.py`](xrobotoolkit_teleop/shared_autonomy/camera_extrinsic.py)) — fixed side camera: 3D camera→base transform saved from a single TCP-touch calibration
- **Depth sampling** ([`depth_sample.py`](xrobotoolkit_teleop/shared_autonomy/depth_sample.py)) and **hover pose construction** ([`hover_pose.py`](xrobotoolkit_teleop/shared_autonomy/hover_pose.py)) — build `MoveL` hover poses above a detected target base

Calibration and debugging scripts:

```bash
# Calibrate fixed side camera (move TCP to target center, press Enter)
python scripts/hardware/calibrate_fixed_camera.py --serial <camera SN>

# Tune HSV / ROI / area parameters with live mask view
python scripts/hardware/debug_target_detection.py --serial <camera SN> --show-mask

# Debug alignment overlay (gripper vs target pixel error)
python scripts/hardware/debug_alignment_vision.py --serial <camera SN>

# Debug target depth sampling / verify hover MoveL target
python scripts/hardware/debug_target_depth.py --serial <camera SN>
python scripts/hardware/test_moveL_target_hover.py
```

Related task tooling for the wrench-screwing task lives in `scripts/hardware/wrench_data/` (TCP pose recording into `tool_tcp.json`, arc-motion primitives in MuJoCo and on hardware, home recording), and [`soft_constraint.py`](xrobotoolkit_teleop/hardware/soft_constraint.py) implements the world-frame motion constraint (free x / y / rz, locked z / rx / ry) used during wrench turning.

## RealSense VR Streaming

[`scripts/visualization/realsense_vr_stream.py`](scripts/visualization/realsense_vr_stream.py) streams Intel RealSense color video to a PICO VR headset over XRoboToolkit (TCP, 4-byte big-endian length + H.264 Annex-B NALU):

```bash
# Single camera
python scripts/visualization/realsense_vr_stream.py

# Dual camera picture-in-picture (main view + corner inset)
python scripts/visualization/realsense_vr_stream.py \
  --main-serial <SN> --pip-serial <SN> --quality balanced --pip-scale 0.38
```

Key options: `--ip` (headset IP; USB PICO default `192.168.88.129`), `--layout screen|sbs` (VR "big screen" panel vs side-by-side stereo; on the headset press **B** to toggle 2D/stereo), `--quality high|balanced|low`, `--mode server` fallback. Start the script first and tap `ZEDMINI → Listen` on the headset only when the terminal shows "waiting for Listen". Helper scripts: `rs_cam_streaming.py` / `view_realsense.py` (PC-side camera check), `vr_stream_diagnose.py` (diagnostics), `vis_mjcf.py` / `vis_urdf.py` (model visualization).

## Data Collection (generic teleop logging)

The framework also automatically logs teleoperation sessions when running hardware demos. Data collection includes:

- **Robot joint states** and end effector poses
- **Camera streams** from multiple viewpoints
- **User input data** from XR controllers
- **Timestamp synchronization** across all data streams

#### Starting Data Collection

1. **Run any hardware teleoperation script:**
   ```bash
   python scripts/hardware/teleop_dual_arx_r5_hardware.py
   ```

2. **Press B button** on the VR controller to start/stop logging
   - First press: Start data logging
   - Second press: Stop logging and save data to disk

3. **Emergency stop:** Press right joystick click to discard current session

#### Data Storage

Collected data is saved as `.pkl` files in the `logs/` directory with timestamps:
```
logs/
├── <robot_name>/
│   └── teleop_log_YYYYMMDD_HHMMSS_<session_id>.pkl
└── <another_robot>/
    ├── teleop_log_YYYYMMDD_HHMMSS_<session_id>.pkl
    └── teleop_log_YYYYMMDD_HHMMSS_<session_id>.pkl
```

#### Validating Collected Data

Use the provided analysis script to verify data integrity and examine collected datasets:

```bash
python scripts/misc/test_data_log_analysis.py logs/<robot_name>/teleop_log_YYYYMMDD_HHMMSS_1.pkl
```

This script will:
- Display available data fields and their types
- Verify robot states and camera images are properly saved
- Show sample entries and data statistics
- Count total logged entries

## Teleoperation Guide

### Tracking Modes

The teleoperation system supports multiple tracking modes for controlling robot end effectors:

#### 1. Controller Tracking (Default)
- **Description**: Uses VR/AR controller poses to control robot end effectors
- **Use Case**: Primary method for precise manipulation tasks
- **Configuration**: Set `pose_source` to `"left_controller"` or `"right_controller"`
- **Tracking**: Full 6DOF pose (position + orientation) or 3DOF position-only

#### 2. Hand Tracking
- **Description**: Uses hand pose estimation from XR cameras
- **Use Case**: Natural hand gesture control

#### 3. Head Tracking
- **Description**: Uses headset pose for controlling specific robot components
- **Use Case**: Head/neck control for humanoid robots or camera orientation

#### 4. Motion Tracker Tracking
- **Description**: Uses additional motion tracking devices for controlling auxiliary robot links
- **Use Case**: Multi-point control (e.g., elbow position while controlling end effector)
- **Configuration**: Add `motion_tracker` config with device serial and target link
- **Note**: Not recommended for 6DOF arms like UR5e; better suited for redundant arms

### Controller Button Functions

When using VR controllers for teleoperation, the following button mappings apply:

#### **Grip Buttons**
- **Left Grip** (`left_grip`): Activates left arm teleoperation
- **Right Grip** (`right_grip`): Activates right arm teleoperation
- **Function**: Hold to enable arm control, release to deactivate

#### **Trigger Buttons**
- **Left Trigger** (`left_trigger`): Controls left gripper/hand
- **Right Trigger** (`right_trigger`): Controls right gripper/hand
- **Function**: Analog control (0.0 = fully open, 1.0 = fully closed)

#### **System Buttons**
- **A Button**: Reserved for system functions (single-arm demos: enable soft constraint / swap cameras)
- **B Button**: Toggle data logging on/off
  - Press once: Start logging
  - Press again: Stop logging and save data

#### **Joysticks/Touchpads**
- **Left Joystick**: Linear velocity commands for mobile robots
- **Right Joystick**: Angular velocity commands for mobile robots
- **Right Axis Click**: stop data logging (discards current data)

Note: in the single-arm BC demos, logging uses **X** (start/stop recording) and **Y** (discard episode) instead of B; see [Behavior-Cloning Data Pipeline](#behavior-cloning-data-pipeline).

## Repository Layout

- `xrobotoolkit_teleop/common/` — base controller classes (`BaseTeleopController`, `HardwareTeleopController`), XR client, generic data logger
- `xrobotoolkit_teleop/hardware/` — hardware controllers (dual UR5e, single UR5e, FR3, ARX R5, Galaxea R1 Lite, Dynamixel head), soft constraints, and `interface/` low-level wrappers
- `xrobotoolkit_teleop/simulation/` — MuJoCo / Placo teleop controllers
- `xrobotoolkit_teleop/bc/` — BC recording, safety filtering, policy deployment
- `xrobotoolkit_teleop/shared_autonomy/` — vision-based shared-autonomy helpers
- `xrobotoolkit_teleop/utils/` — geometry, image, MuJoCo and path utilities
- `assets/` — URDF/MJCF models and meshes per robot
- `scripts/` — runnable entry points (`hardware/`, `simulation/`, `visualization/`, `misc/`)
- `datasets/` — example converted LeRobot datasets
- `FR3/` — FR3 teleop entry point and FR3-specific tooling; `Jiazhua/` — PGC-300-60 gripper CLI

## Dependencies
XR Robotics dependencies:
- [`xrobotookit_sdk`](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind): Python binding for XRoboToolkit PC Service SDK, MIT License

Robotics Simulation and Solver
- [`mujoco`](https://github.com/google-deepmind/mujoco): robotics simulation, Apache 2.0 License
- [`placo`](https://github.com/rhoban/placo): inverse kinematics, MIT License

Hardware Control
- [`dynamixel_sdk`](https://github.com/ROBOTIS-GIT/DynamixelSDK.git): Dynamixel control functions, Apache-2.0 License
- [`ur_rtde`](https://gitlab.com/sdurobotics/ur_rtde): interface for controlling and receiving data from a UR robot, MIT License
- [`ARX R5 SDK`](https://github.com/zhigenzhao/R5/tree/dev/python_pkg): Interface for controlling ARX R5 robotic arms
- [`pyrealsense2`](https://github.com/IntelRealSense/librealsense): Intel RealSense camera interface

Data and CLI
- [`lerobot`](https://github.com/huggingface/lerobot): dataset format and training pipeline for imitation learning, Apache 2.0 License
- [`tyro`](https://github.com/brentyi/tyro): type-driven CLI construction, MIT License

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project is a modification of [XR-Robotics/XRoboToolkit-Teleop-Sample-Python](https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python); the original work is copyright XR-Robotics and licensed under the MIT License. 本项目改造自 XR-Robotics/XRoboToolkit-Teleop-Sample-Python，原始代码版权归 XR-Robotics 所有。
