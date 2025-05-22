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
-   `mujoco`
-   `numpy`
-   `meshcat` (for transformations)
-   `placo`
-   An XR interface library `pyroboticsservice`

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/XR-Robotics/teleop_demo_mujoco.git
    cd teleop_demo_mujoco
    ```

2.  **Set up Conda Environment (Recommended for Linux - Ubuntu 22.04/24.04):**
    The project includes a script to help set up a suitable Conda environment.
    ```bash
    bash setup_conda.sh <your_desired_env_name>
    ```

    After the script completes, activate the environment:
    ```bash
    conda activate <your_desired_env_name> # or 'xr-mujoco' if default was used
    ```

3.  **Install the package:**
    If you are not using the `setup_conda.sh` script, or for other systems, ensure you have a Python environment (>=3.7) with `pip` and `setuptools`. Then install the project in editable mode:
    ```bash
    pip install -e .
    ```

## Usage

### Running the Demo

To run the teleoperation demo with a UR5e robot:

```bash
python scripts/ur5e_dual_arm_demo.py
```
This script initializes the [`MujocoTeleopController`](teleop_demo_mujoco/mujoco_teleop_controller.py) with the UR5e model and starts the teleoperation loop.
