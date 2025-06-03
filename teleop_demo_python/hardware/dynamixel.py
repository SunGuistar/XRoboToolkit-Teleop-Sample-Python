import numpy as np
import time
import threading
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
from teleop_demo_python.utils.pico_client import PicoClient
import meshcat.transformations as tf

# Protocol version
PROTOCOL_VERSION = 2.0

DYNAMIXEL_DEGREE_PER_UNIT = 0.0879
DEFAULT_DEVICE_NAME = "/dev/ttyUSB0"
DEFAULT_BAUDRATE = 4000000

# Control table addresses (common to most Dynamixel motors)
ADDR_LED_RED = 65
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132


class DynamixelController:
    """Generic Dynamixel motor controller for basic operations."""

    def __init__(
        self,
        device_name: str = DEFAULT_DEVICE_NAME,
        baudrate: int = DEFAULT_BAUDRATE,
        protocol_version: float = PROTOCOL_VERSION,
    ):
        self.device_name = device_name
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(protocol_version)
        self.is_closed = True

        # Open port
        if not self.port_handler.openPort():
            raise Exception(f"Failed to open port {device_name}")
        self.is_closed = False
        print(f"Successfully opened port {device_name}")

        # Set baudrate
        if not self.port_handler.setBaudRate(baudrate):
            self.close()
            raise Exception(f"Failed to set baudrate to {baudrate}")
        print(f"Successfully set baudrate to {baudrate}")

    def enableTorque(self, motor_id: int):
        """Enable torque for a specific motor."""
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 1
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(
                f"Torque enable failed for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
            )
        elif dxl_error != 0:
            print(
                f"Torque enable error for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}"
            )

    def disableTorque(self, motor_id: int):
        """Disable torque for a specific motor."""
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 0
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(
                f"Torque disable failed for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
            )
        elif dxl_error != 0:
            print(
                f"Torque disable error for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}"
            )

    def turnOnLED(self, motor_id: int):
        """Turn on LED for a specific motor."""
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_LED_RED, 1
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(
                f"LED turn on failed for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
            )
        elif dxl_error != 0:
            print(
                f"LED turn on error for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}"
            )

    def turnOffLED(self, motor_id: int):
        """Turn off LED for a specific motor."""
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
            self.port_handler, motor_id, ADDR_LED_RED, 0
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(
                f"LED turn off failed for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
            )
        elif dxl_error != 0:
            print(
                f"LED turn off error for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}"
            )

    def setGoalPosition(self, motor_id: int, position: int):
        """Set goal position for a specific motor."""
        dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(
            self.port_handler, motor_id, ADDR_GOAL_POSITION, position
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(
                f"Failed to write position for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
            )
            return False
        elif dxl_error != 0:
            print(
                f"Error in writing position for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}"
            )
            return False
        return True

    def getPresentPosition(self, motor_id: int):
        """Get current position of a specific motor."""
        dxl_present_position, dxl_comm_result, dxl_error = (
            self.packet_handler.read4ByteTxRx(
                self.port_handler, motor_id, ADDR_PRESENT_POSITION
            )
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(
                f"Failed to read position for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}"
            )
            return None
        elif dxl_error != 0:
            print(
                f"Error in reading position for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}"
            )
            return None
        return dxl_present_position

    def close(self):
        """Closes the port and performs cleanup."""
        if not self.is_closed:
            print("Closing Dynamixel controller...")
            try:
                if (
                    hasattr(self.port_handler, "is_using")
                    and self.port_handler.is_using
                ):
                    self.port_handler.closePort()
                    print("Dynamixel port closed.")
            except Exception as e:
                print(f"Error during port close: {e}")
            finally:
                self.is_closed = True


class DynamixelHeadController:
    """Specific controller for yaw-pitch head control using two Dynamixel motors."""

    def __init__(
        self,
        pico_client: PicoClient,
        device_name: str = None,
        baudrate: int = None,
    ):
        # Head control specific constants
        self.YAW_MOTOR_ID = 3
        self.PITCH_MOTOR_ID = 1
        self.YAW_CENTER = 1521
        self.PITCH_CENTER = 2753

        # Store dependencies
        self.pico_client = pico_client
        self.tf = tf

        # Initialize the generic Dynamixel controller
        device_name = device_name or DEFAULT_DEVICE_NAME
        baudrate = baudrate or DEFAULT_BAUDRATE
        self.controller = DynamixelController(device_name, baudrate)

        # Initialize motors for head control
        self._initialize_head_motors()

    def _initialize_head_motors(self):
        """Initialize yaw and pitch motors for head control."""
        print("Initializing head control motors...")

        # Enable torque for both motors
        self.controller.enableTorque(self.YAW_MOTOR_ID)
        self.controller.enableTorque(self.PITCH_MOTOR_ID)
        print("Torque enabled for YAW and PITCH motors.")

        # Turn on LEDs
        self.controller.turnOnLED(self.YAW_MOTOR_ID)
        self.controller.turnOnLED(self.PITCH_MOTOR_ID)
        print("LEDs turned on for YAW and PITCH motors.")

    def mapYawToDynamixelPosition(self, yaw: float) -> int:
        """Map yaw angle (degrees) to Dynamixel position."""
        position = self.YAW_CENTER + int(yaw / DYNAMIXEL_DEGREE_PER_UNIT)
        # print(f"Yaw {yaw:.2f}° -> Dynamixel position: {position}")
        return position

    def mapPitchToDynamixelPosition(self, pitch: float) -> int:
        """Map pitch angle (degrees) to Dynamixel position."""
        # Clamp pitch to valid range
        if pitch > 50:
            pitch = 50
        if pitch < -50:
            pitch = -50

        position = self.PITCH_CENTER - int(pitch / DYNAMIXEL_DEGREE_PER_UNIT)
        # print(f"Pitch {pitch:.2f}° -> Dynamixel position: {position}")
        return position

    def setHeadPosition(self, yaw: float, pitch: float):
        """Set head position using yaw and pitch angles in degrees."""
        yaw_position = self.mapYawToDynamixelPosition(yaw)
        pitch_position = self.mapPitchToDynamixelPosition(pitch)

        yaw_success = self.controller.setGoalPosition(
            self.YAW_MOTOR_ID, yaw_position
        )
        pitch_success = self.controller.setGoalPosition(
            self.PITCH_MOTOR_ID, pitch_position
        )

        return yaw_success and pitch_success

    def get_target_orientation(self) -> tuple:
        """
        Fetches the current head orientation from the Pico Robotics Service.
        Returns a tuple (yaw, pitch) in degrees.
        """
        try:
            head_pose = self.pico_client.get_pose_by_name("headset")
            quat = np.array(
                [head_pose[6], head_pose[3], head_pose[4], head_pose[5]]
            )  # [w, x, y, z]
            rot_matrix = self.tf.quaternion_matrix(quat)[:3, :3]
            euler = self.tf.euler_from_matrix(rot_matrix, "rzxy")
            currentYaw = euler[2] * 180.0 / np.pi
            currentPitch = euler[1] * 180.0 / np.pi

            return currentYaw, currentPitch

        except Exception as e:
            print(f"Error fetching head orientation: {e}")
            return 0.0, 0.0  # Default values in case of error

    def run(self, stop_event: threading.Event = threading.Event()):
        control_thread = threading.Thread(
            target=self.run_thread,
            args=(stop_event,),
        )
        control_thread.start()

        while not stop_event.is_set():
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                print("Stopping head control...")
                stop_event.set()

        control_thread.join()

    def run_thread(self, stop_event: threading.Event):
        """
        Main control loop for head tracking.
        :param get_orientation_callback: A function that returns a tuple (current_yaw_degrees, current_pitch_degrees).
        :param stop_event: A threading.Event object. The loop stops when this event is set.
        """
        print("Head control loop starting...")
        last_yaw = 0.0
        last_pitch = 0.0

        try:
            while not stop_event.is_set():
                current_yaw_raw, current_pitch_raw = (
                    self.get_target_orientation()
                )

                current_yaw = current_yaw_raw
                if current_yaw > 90.0 and current_yaw < 180.0:
                    current_yaw -= 180.0
                if current_yaw < -90.0:
                    current_yaw = 180.0 + current_yaw

                current_pitch = current_pitch_raw
                if current_pitch < -90.0:
                    current_pitch = -current_pitch - 180.0
                if current_pitch > 90.0:
                    current_pitch = 180.0 - current_pitch

                # print(
                #     f"Current Head Orientation: Yaw={current_yaw:.2f}°, Pitch={current_pitch:.2f}°"
                # )

                if (
                    abs(current_yaw - last_yaw) > 0.01
                    or abs(current_pitch - last_pitch) > 0.01
                ):
                    success = self.setHeadPosition(current_yaw, current_pitch)
                    if success:
                        last_yaw = current_yaw
                        last_pitch = current_pitch

                time.sleep(0.01)  # 10ms delay

        except Exception as e:
            print(f"Exception in head control loop: {e}")
        finally:
            self._cleanup_head_motors()

    def _cleanup_head_motors(self):
        """Cleanup head control motors."""
        print(
            "Head control loop stopping. Disabling torque and turning off LEDs..."
        )

        # Disable torque for both motors
        self.controller.disableTorque(self.YAW_MOTOR_ID)
        self.controller.disableTorque(self.PITCH_MOTOR_ID)

        # Turn off LEDs
        self.controller.turnOffLED(self.YAW_MOTOR_ID)
        self.controller.turnOffLED(self.PITCH_MOTOR_ID)
        print("Torque disabled and LEDs off for YAW and PITCH motors.")

    def close(self):
        """Close the head controller and cleanup resources."""
        self._cleanup_head_motors()
        self.controller.close()

    def __del__(self):
        """Ensures resources are released when the object is deleted."""
        self.close()
