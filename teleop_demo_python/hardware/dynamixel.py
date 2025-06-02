import numpy as np 
import time
import threading
from teleop_demo_python.utils.pico_client import PicoClient
from teleop_demo_python.utils.geometry import R_HEADSET_TO_WORLD
import meshcat.transformations as tf
from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS

# Dynamixel motor parameters
MOTOR_ID = 3
BAUDRATE = 4000000 # 4.5M causes an error when using python
DEVICE_NAME = "/dev/ttyUSB0"  # "COM3"  # Change this to your port name

# Protocol version
PROTOCOL_VERSION = 2.0

# Control table address
ADDR_LED_RED = 65
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

# Dynamixel motor IDs
YAW_MOTOR_ID = 3
PITCH_MOTOR_ID = 1

# Dynamixel position constants
YAW_CENTER = 1521
PITCH_CENTER = 2753
DYNAMIXEL_DEGREE_PER_UNIT = 0.0879

def mapYawToDynamixelPosition(yaw: float) -> int:
    # Map yaw from [-90, 90] to Dynamixel position
    position = YAW_CENTER + int(yaw / DYNAMIXEL_DEGREE_PER_UNIT)
    print(f"Yaw Dynamixel position: {position}")
    return position

# Function to map pitch angle to Dynamixel position
def mapPitchToDynamixelPosition(pitch: float) -> int:
    # Map pitch from [-50, 50] to Dynamixel position
    
    if pitch > 50:
        pitch = 50
    if pitch < -50:
        pitch = -50
    
    position = PITCH_CENTER - int(pitch / DYNAMIXEL_DEGREE_PER_UNIT)
    print(f"Pitch Dynamixel position: {position}")
    return position

class DynamixelHeadController:
    def __init__(self, 
                 pico_client: PicoClient, 
                 device_name: str = DEVICE_NAME, 
                 baudrate: int = BAUDRATE, 
                 protocol_version: float = PROTOCOL_VERSION):
        self.device_name = device_name
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        self.port_handler = PortHandler(device_name)
        self.packet_handler = PacketHandler(protocol_version)
        self.is_closed = True # Track if port is open
        self.pico_client = pico_client

        # Open port
        if not self.port_handler.openPort():
            raise Exception(f"Failed to open port {device_name}")
        self.is_closed = False
        print(f"Successfully opened port {device_name}")
        
        # Set baudrate
        if not self.port_handler.setBaudRate(baudrate):
            self.close() # Close port if baudrate setting fails
            raise Exception(f"Failed to set baudrate to {baudrate}")
        print(f"Successfully set baudrate to {baudrate}")

        # Enable torque for motors
        self.enableTorque(YAW_MOTOR_ID) # Assuming these methods should raise exceptions or return status
        self.enableTorque(PITCH_MOTOR_ID)
        print("Torque enabled for YAW and PITCH motors.")

        self.turnOnLED(YAW_MOTOR_ID)
        self.turnOnLED(PITCH_MOTOR_ID)
        print("LEDs turned on for YAW and PITCH motors.")

    def enableTorque(self, motor_id: int):
        # It's good practice for these methods to check for errors
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 1)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Torque enable failed for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Torque enable error for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}")

    def disableTorque(self, motor_id: int):
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, ADDR_TORQUE_ENABLE, 0)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Torque disable failed for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Torque disable error for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}")
    
    def turnOnLED(self, motor_id: int):
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, ADDR_LED_RED, 1)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"LED turn on failed for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"LED turn on error for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}")

    def turnOffLED(self, motor_id: int):
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, motor_id, ADDR_LED_RED, 0)
        if dxl_comm_result != COMM_SUCCESS:
            print(f"LED turn off failed for motor {motor_id}: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"LED turn off error for motor {motor_id}: {self.packet_handler.getRxPacketError(dxl_error)}")

    def control_loop_with_orientation_provider(self, get_orientation_callback, stop_event: threading.Event):
        """
        Main control loop for Dynamixel motors.
        Assumes __init__ has successfully set up the motors (port, baud, torque, LED).
        :param get_orientation_callback: A function that returns a tuple (current_yaw_degrees, current_pitch_degrees).
        :param stop_event: A threading.Event object. The loop stops when this event is set.
        """
        print("Dynamixel motor control loop starting...")
        last_yaw = 0.0
        last_pitch = 0.0

        try:
            while not stop_event.is_set():
                current_yaw_raw, current_pitch_raw = get_orientation_callback()

                # Apply C++ specific angle normalization logic
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
                
                # print(f"Yaw: {current_yaw:.2f}°, Pitch: {current_pitch:.2f}°")

                # Check if position needs to be updated
                if abs(current_yaw - last_yaw) > 0.01 or abs(current_pitch - last_pitch) > 0.01:
                    yaw_position = mapYawToDynamixelPosition(current_yaw)
                    pitch_position = mapPitchToDynamixelPosition(current_pitch)
                    
                    # Write Yaw position
                    dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(self.port_handler, YAW_MOTOR_ID, ADDR_GOAL_POSITION, yaw_position)
                    if dxl_comm_result != COMM_SUCCESS:
                        print(f"Failed to write yaw position: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
                    elif dxl_error != 0:
                        print(f"Error in writing yaw position: {self.packet_handler.getRxPacketError(dxl_error)}")
                    
                    # Write Pitch position
                    dxl_comm_result, dxl_error = self.packet_handler.write4ByteTxRx(self.port_handler, PITCH_MOTOR_ID, ADDR_GOAL_POSITION, pitch_position)
                    if dxl_comm_result != COMM_SUCCESS:
                        print(f"Failed to write pitch position: {self.packet_handler.getTxRxResult(dxl_comm_result)}")
                    elif dxl_error != 0:
                        print(f"Error in writing pitch position: {self.packet_handler.getRxPacketError(dxl_error)}")
                    
                    last_yaw = current_yaw
                    last_pitch = current_pitch
                
                time.sleep(0.01)  # 10ms delay, similar to C++ example

        except Exception as e:
            print(f"Exception in Dynamixel control loop: {e}")
        finally:
            print("Dynamixel control loop stopping. Disabling torque and turning off LEDs...")
            # Disable torque for both motors
            self.disableTorque(YAW_MOTOR_ID)
            self.disableTorque(PITCH_MOTOR_ID)
            
            # Turn off LEDs
            self.turnOffLED(YAW_MOTOR_ID)
            self.turnOffLED(PITCH_MOTOR_ID)
            print("Torque disabled and LEDs off for YAW and PITCH motors.")

    def getHeadOrientationFromPico(self) -> tuple:
        """
        Fetches the current head orientation from the Pico Robotics Service.
        Returns a tuple (yaw, pitch) in degrees.
        """
        try:
            # Get head pose from pico client
            head_pose = self.pico_client.get_pose_by_name("headset")
            quat = np.array([head_pose[6], head_pose[3], head_pose[4], head_pose[5]])  # [w, x, y, z]
            
            # Convert quaternion to rotation matrix
            rot_matrix = tf.quaternion_matrix(quat)[:3, :3]
            
            euler = tf.euler_from_matrix(rot_matrix, 'rzxy')  
            
            currentYaw = euler[2] * 180.0 / np.pi  
            currentPitch = euler[1] * 180.0 / np.pi 
                
            return currentYaw, currentPitch

        except Exception as e:
            print(f"Error fetching head orientation: {e}")
            return 0.0, 0.0  # Default values in case of error

    def close(self):
        """Closes the port and performs any other necessary cleanup."""
        if not self.is_closed:
            print("Closing Dynamixel controller...")
            # Disable torque and turn off LEDs as a final safety measure if not already done
            try:
                if self.port_handler.is_open: # Check if port is open before trying to use it
                    self.disableTorque(YAW_MOTOR_ID)
                    self.disableTorque(PITCH_MOTOR_ID)
                    self.turnOffLED(YAW_MOTOR_ID)
                    self.turnOffLED(PITCH_MOTOR_ID)
            except Exception as e:
                print(f"Error during pre-close cleanup: {e}")
            finally:
                if self.port_handler.is_open:
                    self.port_handler.closePort()
                    print("Dynamixel port closed.")
                self.is_closed = True
        else:
            print("Dynamixel controller was already closed or not successfully opened.")

    def __del__(self):
        """Ensures resources are released when the object is deleted."""
        self.close()

# ... (if you have a main section for testing, it would go here)
# Example usage (conceptual, place in your main script):
# if __name__ == '__main__':
#     # Dummy orientation provider for testing
#     def get_dummy_orientation():
#         # Simulate changing orientation
#         # In a real application, this would get data from your headset/sensor
#         current_yaw = (time.time() * 10) % 180 - 90  # Sweeps -90 to 90
#         current_pitch = (time.time() * 5) % 100 - 50 # Sweeps -50 to 50
#         return current_yaw, current_pitch

#     stop_signal = threading.Event()
#     controller = None

#     try:
#         controller = DynamixelController(device_name=DEVICE_NAME, baudrate=BAUDRATE)
        
#         # Start the control loop in a new thread
#         control_thread = threading.Thread(
#             target=controller.control_loop_with_orientation_provider,
#             args=(get_dummy_orientation, stop_signal)
#         )
#         control_thread.start()
        
#         print("Main thread: Control loop started. Running for 10 seconds...")
#         time.sleep(10) # Let the loop run for some time
        
#         print("Main thread: Signaling control loop to stop.")
#         stop_signal.set()
#         control_thread.join() # Wait for the control loop thread to finish
#         print("Main thread: Control loop finished.")

#     except Exception as e:
#         print(f"An error occurred in the main execution: {e}")
#     finally:
#         if controller:
#             controller.close() # Explicitly close resources
#         print("Program terminated.")

