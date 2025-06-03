# Teleop Demo Python

Pico teleoperation demo written in python for both mujoco simulation and robot hardware.

## Overview

This project provides a framework for controlling robots in robot hardware and MuJoCo simulation through XR (VR/AR) input devices. It allows users to manipulate robot arms using natural hand movements captured through XR controllers.

## Dependencies

The main dependencies are listed in the [`pyproject.toml`](pyproject.toml) file and include:
-   `numpy`
-   `meshcat`
-   [`mujoco`](https://github.com/google-deepmind/mujoco)
-   [`placo`](https://github.com/rhoban/placo) (inverse kinematics library)
-   [`pyroboticsservice`](https://github.com/XR-Robotics/RoboticsService-Python) (Python binding for xr-robot sdk)

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/XR-Robotics/teleop_demo_python.git
    cd teleop_demo_python
    ```

2.  **Installation**
    
    The project includes a script to help set up a suitable Conda environment and install all dependencies.
    ```bash
    bash setup_conda.sh --conda <optional_env_name>
    conda activate <env_name>
    bash setup_conda.sh --install
    ```

    If installing on system python
    ```bash
    bash setup.sh
    ```

3. Download and install `roboticsservice`
    ```bash
    dpkg -i /path/to/roboticsservice_1.0.0.0_amd64.deb
    ```

## Usage

### Running the MuJoCo Simulation Demo

To run the teleoperation demo with a UR5e robot in MuJoCo simulation:

```bash
python scripts/teleop_dual_ur5e_mujoco.py
```
This script initializes the [`MujocoTeleopController`](teleop_demo_python/mujoco_teleop_controller.py) with the UR5e model and starts the teleoperation loop.

### Running the Hardware Demo (Dual UR Arms and Dynamixel Head)

To run the teleoperation demo with the physical dual UR arms and Dynamixel-based head:

1.  **Normal Operation:**
    ```bash
    python scripts/teleop_dual_ur5e_hardware.py
    ```
    This script initializes the [`DynamixelHeadController`](teleop_demo_python/hardware/dynamixel.py) and [`DualArmURController`](teleop_demo_python/hardware/ur.py) and starts the teleoperation loops for both head tracking and arm control.

2.  **Resetting Arm Positions:**
    If you need to reset the UR arms to their initial/home positions and initialize the robotiq grippers, you can run the script with the `--reset` flag:
    ```bash
    python scripts/teleop_dual_ur5e_hardware.py --reset
    ```
    This will execute the reset procedure defined in the [`DualArmURController`](teleop_demo_python/hardware/ur.py) and then exit.

3.  **Visualizing IK results:**
    To visualize the inverse kinematics solution with placo during teleoperation, run the script with the `--visualize_placo` flag.
    ```bash
    python scripts/teleop_dual_ur5e_hardware.py --visualize_placo
    ```