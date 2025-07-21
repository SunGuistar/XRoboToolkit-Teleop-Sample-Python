# Gemini Project Configuration (`GEMINI.md`)

This file provides project-specific guidance and configuration for the Gemini agent to ensure its actions are aligned with the project's standards, conventions, and requirements.

## 1. Project Overview

*   **Purpose:** Teleoperation demo with MuJoCo and UR5 robot.
*   **Tech Stack:** Python, MuJoCo, ROS (for some parts), Pybind11, UR RTDE.
*   **Key Features:** Real-time teleoperation of various robot arms (UR5e, ARX, Galaxea) in both simulation and hardware. Supports bimanual setups and dexterous hands (Shadow Hand, Inspire Hand).

---

## 2. Build, Test, and Run Commands

Provide the exact shell commands needed to build, test, and run the project.

*   **Build:**
    ```sh
    # The command to setup new conda environment for this project
    ./setup_conda.sh --conda

    # the command to activate the conda environment
    conda activate xr-robotics
    
    # the command to install the project within the conda env
    ./setup_conda.sh --install
    ```

*   **Run Main Application / Simulation:**
    ```sh
    # Command to run a primary simulation or hardware script
    python scripts/simulation/teleop_dual_ur5e_mujoco.py
    ```

*   **Linting & Formatting:**
    ```sh
    # Commands to check for code style and formatting issues
    black .
    isort .
    flake8 .
    ```

---

## 3. Dependency Management

*   **Primary Tool:** pip and shell scripts.
*   **Installation Command:**
    ```sh
    # The command to setup new conda environment for this project
    ./setup_conda.sh --conda

    # the command to activate the conda environment
    conda activate xr-robotics

    # the command to install the project within the conda env
    ./setup_conda.sh --install
    ```
*   **Configuration File(s):** `pyproject.toml`, `setup.sh`

---

## 4. Coding Style and Conventions

*   **Code Style:** PEP 8
*   **Formatter:** `black`, `isort`
*   **Linter:** `flake8`
*   **Naming Conventions:** `snake_case` for variables and functions, `PascalCase` for classes.
*   **Key Architectural Patterns:** The project uses a controller-based architecture. A `BaseTeleopController` is extended for different simulation (`MujocoTeleopController`, `PlacoTeleopController`) and hardware environments. Hardware interfaces are abstracted in `xrobotoolkit_teleop/hardware/interface/`.

---

## 5. Key Files and Directories

List important files or directories that the agent should be aware of.

*   `pyproject.toml`: Project metadata and Python package dependencies.
*   `setup.sh`: Main installation script that clones git dependencies and installs the project.
*   `assets/`: Contains all URDF models and meshes for robots.
*   `scripts/hardware/`: High-level teleoperation scripts for physical hardware.
*   `scripts/simulation/`: High-level teleoperation scripts for simulations (e.g., MuJoCo, Placo).
*   `xrobotoolkit_teleop/`: The core Python package containing all teleoperation logic.
*   `xrobotoolkit_teleop/common/base_teleop_controller.py`: The abstract base class for all teleop controllers.
*   `xrobotoolkit_teleop/hardware/interface/`: Low-level wrappers for hardware communication (robots, grippers, cameras).
*   `xrobotoolkit_teleop/simulation/`: Controllers for different simulation environments.
*   `dependencies/`: Directory where git dependencies are cloned during setup.

---

## 6. User Preferences & Notes

*   Use `black`, `isort`, and `flake8` for linting and formatting.
*   Prefer adding new teleoperation scripts in the `scripts/` directory, following the existing hardware/simulation structure.
*   Always use absolute paths for assets when loading URDFs, leveraging the `path_utils` where appropriate.

---

## 7. Technical Implementation Details

This section provides more detailed descriptions of the teleoperation examples, sourced from `teleop_details.md`.

### XR Client
- Connects to the XR device using the Python binding of [xrobotoolkit-sdk](https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind).
- The client is implemented in `xrobotoolkit_teleop/utils/xr_client.py`.

### Placo
- [PlaCo](https://placo.readthedocs.io/en/latest/) is used for whole-body inverse kinematics and dynamics based on a Quadratic Programming solver.
- **Main Placo Tasks Used:**
  - **Frame Task:** Constrains the position and orientation of an end-effector frame.
  - **Manipulability Task:** Ensures stable robot motion near singularities or workspace limits.
  - **Kinetic Energy Regularization:** Minimizes the system's kinetic energy.
  - **Joint Task:** Regularizes the joint state towards a default state.

### Mujoco Simulation
- **Robot Definition:** Requires both `.xml` (for MuJoCo) and `.urdf` (for Placo) files, which must be consistent. The `.xml` can optionally define extra bodies for visualizing teleop targets.
- **Teleoperation Task Config:** A dictionary defines the teleop setup for each hand/arm.
    ```python
    config = {
        "right_hand": {
            "link_name": "right_tool0",
            "pose_source": "right_controller",
            "control_trigger": "right_grip",
            "vis_target": "right_target", # Optional: for MuJoCo visualization
        },
        # ... and so on for the left hand
    }
    ```
- **Parallel Gripper Control:** Gripper behavior is configured within the end-effector dictionary. It specifies the joint name, trigger key, and open/closed positions.
    ```python
    "gripper_config": {
        "type": "parallel",
        "gripper_trigger": "right_trigger",
        "joint_names": ["right_gripper_finger_joint1"],
        "open_pos": [0.05],
        "close_pos": [0.0],
    }
    ```
  - **Note:** In the MuJoCo `.xml`, one gripper joint is actuated, and others are linked via equality constraints.

### Hardware Teleoperation
- **Robot Definition:** Only a `.urdf` file is required.
- **Config:** Similar to the simulation config, it maps arms to controller inputs and defines the end-effector link.
- **Examples:**
  - **Dual UR5e:** `scripts/hardware/teleop_dual_ur5e_hardware.py`
  - **Galaxea A1X:** `scripts/hardware/teleop_dual_a1x_hardware.py`