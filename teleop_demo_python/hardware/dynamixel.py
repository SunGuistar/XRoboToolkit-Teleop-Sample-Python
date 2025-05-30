import dynamixel_sdk as dxl

# Dynamixel motor parameters
MOTOR_ID = 3
BAUDRATE = 4500000
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

class DynamixelController:
    def __init__(self, device_name: str = DEVICE_NAME, baudrate: int = BAUDRATE, protocol_version: float = PROTOCOL_VERSION):
        self.device_name = device_name
        self.baudrate = baudrate
        self.protocol_version = protocol_version
        self.port_handler = dxl.PortHandler(device_name)
        self.packet_handler = dxl.PacketHandler(protocol_version)

        # Open port
        if not self.port_handler.openPort():
            raise Exception("Failed to open port")
        
        # Set baudrate
        if not self.port_handler.setBaudRate(baudrate):
            raise Exception("Failed to set baudrate")

        # Enable torque for motors
        self.enableTorque(YAW_MOTOR_ID)
        self.enableTorque(PITCH_MOTOR_ID)

        self.turnOnLED(YAW_MOTOR_ID)
        self.turnOnLED(PITCH_MOTOR_ID)

    def enableTorque(self, motor_id: int):
        dxl.write1ByteTxRx(self.port_handler, self.packet_handler, motor_id, ADDR_TORQUE_ENABLE, 1)

    def disableTorque(self, motor_id: int):
        dxl.write1ByteTxRx(self.port_handler, self.packet_handler, motor_id, ADDR_TORQUE_ENABLE, 0)
    
    def turnOnLED(self, motor_id: int):
        dxl.write1ByteTxRx(self.port_handler, self.packet_handler, motor_id, ADDR_LED_RED, 1)

    def turnOffLED(self, motor_id: int):
        dxl.write1ByteTxRx(self.port_handler, self.packet_handler, motor_id, ADDR_LED_RED, 0)

    