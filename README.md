# Teleop Demo MuJoCo

Teleoperation demo with MuJoCo and UR5 robot.

## Overview

This project provides a framework for controlling robots in MuJoCo simulation through XR (VR/AR) input devices. It allows users to manipulate robot arms using natural hand movements captured through XR controllers.

The current implementation focuses on a UR5e robot arm, providing real-time control through position-based teleoperation.

## Features

-   Real-time teleoperation of robot arms in MuJoCo physics simulation.
-   Intuitive control using XR controllers.
-   Coordinate frame transformations between XR and robot spaces.
-   Support for Universal Robots UR5e (expandable to other robots).
-   Inverse kinematics solving using Placo.

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
    git clone https://github.com/XR-Robotics/teleop_demo_mujoco.git
    cd teleop_demo_mujoco
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

### Running the Demo

To run the teleoperation demo with a UR5e robot:

```bash
python scripts/ur5e_dual_arm_demo.py
```
This script initializes the [`MujocoTeleopController`](teleop_demo_mujoco/mujoco_teleop_controller.py) with the UR5e model and starts the teleoperation loop.
