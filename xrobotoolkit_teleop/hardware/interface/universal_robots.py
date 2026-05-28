import time

import numpy as np
import rtde_control
import rtde_receive

from xrobotoolkit_teleop.hardware.interface.robotiq_gripper import RobotiqGripper

# Default for single-arm teleop (teleop_single_ur5e_hardware.py)
DEFAULT_ROBOT_IP = "192.168.56.10"

# Legacy lab dual-arm setup (teleop_dual_ur5e_hardware.py)
LEFT_ROBOT_IP = "192.168.50.55"
RIGHT_ROBOT_IP = "192.168.50.195"

SERVO_TIME = 0.017
LOOKAHEAD_TIME = 0.1
SERVO_GAIN = 300.0
MAX_VELOCITY = 0.5
MAX_ACCELERATION = 1.0

GRIPPER_FORCE = 128
GRIPPER_SPEED = 255
CONTROLLER_DEADZONE = 0.1

LEFT_INITIAL_JOINT_DEG = np.array([165.26, -47.50, 118.93, -38.96, 87.51, 149.56])
# Home pose for single UR5e @ 192.168.56.10 (J1..J6 = shoulder_pan .. wrist_3).
# Tuned so forearm ~horizontal (J2 shoulder_lift +15 deg) and elbow -45 deg on J3
# (teach-pendant CCW on elbow; UR joint delta -51.54 -> -96.54) for wrist vs table.
RIGHT_INITIAL_JOINT_DEG = np.array([-173.59, -110, -70, -90, 90, -90])

RTDE_CONNECT_RETRIES = 3
RTDE_RETRY_DELAY_S = 1.5


def _disconnect_rtde_control(rtde_c) -> None:
    if rtde_c is None:
        return
    try:
        rtde_c.servoStop()
    except Exception:
        pass
    try:
        rtde_c.stopScript()
    except Exception:
        pass
    try:
        rtde_c.disconnect()
    except Exception:
        pass


def connect_rtde(robot_ip: str, retries: int = RTDE_CONNECT_RETRIES) -> tuple:
    """Open RTDE receive + control with retries (clears stale external-control sessions)."""
    last_error: Exception | None = None
    rtde_r = None
    rtde_c = None

    for attempt in range(1, retries + 1):
        _disconnect_rtde_control(rtde_c)
        rtde_c = None
        try:
            if rtde_r is None:
                rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
            time.sleep(0.3)
            rtde_c = rtde_control.RTDEControlInterface(robot_ip)
            return rtde_c, rtde_r
        except RuntimeError as exc:
            last_error = exc
            print(f"RTDE control connect attempt {attempt}/{retries} failed: {exc}")
            time.sleep(RTDE_RETRY_DELAY_S)

    raise RuntimeError(
        f"Failed to connect RTDE control to UR at {robot_ip} after {retries} attempts: {last_error}\n"
        "Checklist:\n"
        f"  1. Ping the robot: ping {robot_ip}\n"
        "  2. Teach pendant: Remote control ON, program stopped, no protective stop\n"
        "  3. No other PC/script holding RTDE (re-run after closing old teleop)\n"
        "  4. On pendant: stop any running program, then retry\n"
        "  5. PC on same subnet (e.g. 192.168.56.x)\n"
        "  6. Quick test: python scripts/hardware/test_ur_connection.py --demo-motion=False"
    ) from last_error


class URController:
    def __init__(
        self,
        robot_ip: str,
        initial_joint_positions: np.ndarray,
        max_velocity: float = MAX_VELOCITY,
        max_acceleration: float = MAX_ACCELERATION,
        servo_time: float = SERVO_TIME,
        lookahead_time: float = LOOKAHEAD_TIME,
        servo_gain: float = SERVO_GAIN,
        gripper_force: float = GRIPPER_FORCE,
        gripper_speed: float = GRIPPER_SPEED,
        use_gripper: bool = True,
    ):
        self.robot_ip = robot_ip
        self.initial_joint_positions = initial_joint_positions
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.servo_time = servo_time
        self.lookahead_time = lookahead_time
        self.servo_gain = servo_gain
        self.gripper_force = gripper_force
        self.gripper_speed = gripper_speed
        self.use_gripper = use_gripper

        self.rtde_c = None
        self.rtde_r = None
        self.gripper = None

        self.rtde_c, self.rtde_r = connect_rtde(robot_ip)
        print(f"Connected to UR robot at {robot_ip}")

        if self.use_gripper:
            self.gripper = RobotiqGripper()
            try:
                self.gripper.connect(robot_ip, 63352)
                print("Gripper connected.")
            except Exception as e:
                self.close()
                raise RuntimeError(
                    f"Connected to UR at {robot_ip}, but Robotiq gripper on port 63352 failed: {e}\n"
                    "Pass use_gripper=False to skip the gripper."
                ) from e
        else:
            print("Gripper disabled (arm-only mode).")

    def reset(self):
        print(f"Moving to initial joint positions (rad): {self.initial_joint_positions}")
        ok = self.rtde_c.moveJ(self.initial_joint_positions)
        if not ok:
            raise RuntimeError(
                "moveJ failed (RTDE control script may have stopped).\n"
                "  - Check teach pendant: Remote mode, no protective stop\n"
                "  - Target joints may be invalid for this arm; read current q with:\n"
                "      python scripts/hardware/test_ur_connection.py\n"
                "  - Teleop without --reset uses the current pose as the starting point"
            )
        print("Reached initial position.")
        if self.use_gripper:
            self.gripper.activate()
            print("Gripper activated.")

    def servo_joints(self, joint_positions: np.ndarray) -> bool:
        t_start = self.rtde_c.initPeriod()
        ok = self.rtde_c.servoJ(
            joint_positions,
            self.max_velocity,
            self.max_acceleration,
            self.servo_time,
            self.lookahead_time,
            self.servo_gain,
        )
        self.rtde_c.waitPeriod(t_start)
        return ok

    def servo_stop(self) -> None:
        if self.rtde_c is not None:
            try:
                self.rtde_c.servoStop()
            except Exception:
                pass

    def open_gripper(self):
        if not self.use_gripper:
            return
        self.gripper.move_and_wait_for_pos(
            self.gripper.get_open_position(),
            self.gripper_speed,
            self.gripper_force,
        )

    def close_gripper(self):
        if not self.use_gripper:
            return
        self.gripper.move_and_wait_for_pos(
            self.gripper.get_closed_position(),
            self.gripper_speed,
            self.gripper_force,
        )

    def get_current_joint_positions(self) -> np.ndarray:
        return np.array(self.rtde_r.getActualQ())

    def get_current_tcp_pose(self) -> np.ndarray:
        return np.array(self.rtde_r.getActualTCPPose())

    def close(self):
        if self.rtde_c is not None:
            try:
                self.rtde_c.servoStop()
                self.rtde_c.stopScript()
            except Exception:
                pass
            self.rtde_c = None
        if self.rtde_r is not None:
            self.rtde_r = None
        if self.gripper is not None:
            try:
                self.gripper.disconnect()
            except Exception:
                pass
            self.gripper = None
        print("UR controller closed.")
