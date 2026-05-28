"""
Control script for DH-Robotics PGC-300-60 electric parallel gripper.

Communication: Modbus RTU over RS485 (default) or Modbus RTU over UDP (e.g. Adam gateway).

Register map (from DH gripper manual, PGE/PGC series):
  0x0100  INIT         write 1=initialize, 0xA5=full recalibration
  0x0101  FORCE        20-100 (%)
  0x0103  POSITION     0-1000 (per mille; mapping see POSITION_* below)
  0x0104  SPEED        1-100 (%)
  0x0200  INIT_STATE   read: 0=not initialized, 1=initialized
  0x0201  GRIP_STATE   read: 0=moving, 1=at position, 2=object caught, 3=object dropped
  0x0202  POSITION_FB  read: actual position (0-1000)

Dependencies (RS485):
  pip install pymodbus pyserial

Example:
  python Jiazhua/pgc300_60_gripper.py --port /dev/ttyUSB0 init
  python Jiazhua/pgc300_60_gripper.py --port /dev/ttyUSB0 open
  python Jiazhua/pgc300_60_gripper.py --port /dev/ttyUSB0 close --force 50
  python Jiazhua/pgc300_60_gripper.py --port /dev/ttyUSB0 set-position 500
  python Jiazhua/pgc300_60_gripper.py --udp-host 10.10.10.18 status
"""

from __future__ import annotations

import argparse
import inspect
import socket
import struct
import time
from enum import IntEnum
from typing import Any, Optional, Protocol, Union

# --- Modbus register addresses -------------------------------------------------

REG_INIT = 0x0100
REG_FORCE = 0x0101
REG_POSITION = 0x0103
REG_SPEED = 0x0104
REG_INIT_STATE = 0x0200
REG_GRIP_STATE = 0x0201
REG_POSITION_FB = 0x0202
REG_INIT_DIRECTION = 0x0301  # 0=home to open, 1=home to closed (saved in flash)

INIT_COMMAND = 0x01
INIT_FULL_CALIBRATION = 0xA5

POSITION_MIN = 0
POSITION_MAX = 1000

# Default mapping for this PGC-300-60: register 1000 = physically open, 0 = closed.
# Use --standard-position for manual mapping (0=open, 1000=closed).
POSITION_REGISTER_OPEN = POSITION_MAX
POSITION_REGISTER_CLOSED = POSITION_MIN
FORCE_MIN = 20
FORCE_MAX = 100
SPEED_MIN = 1
SPEED_MAX = 100

DEFAULT_BAUDRATE = 115200
DEFAULT_SLAVE_ID = 1
DEFAULT_SERIAL_TIMEOUT = 2.0
INTER_FRAME_DELAY_S = 0.05


class GripState(IntEnum):
    MOVING = 0
    AT_POSITION = 1
    OBJECT_CAUGHT = 2
    OBJECT_DROPPED = 3


class InitState(IntEnum):
    NOT_INITIALIZED = 0
    INITIALIZED = 1
    INITIALIZING = 2  # initialization in progress (DH manual 0x0200)


# --- Low-level Modbus transports -----------------------------------------------


def _modbus_crc(data: bytes) -> bytes:
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return struct.pack("<H", crc)


class ModbusTransport(Protocol):
    def read_holding_register(self, address: int) -> int: ...

    def write_holding_register(self, address: int, value: int) -> None: ...

    def close(self) -> None: ...


def _pymodbus_device_kwargs(client: Any, device_id: int) -> dict[str, int]:
    """Return device_id= or slave= depending on installed pymodbus version."""
    for method in (client.write_register, client.read_holding_registers):
        params = inspect.signature(method).parameters
        if "device_id" in params:
            return {"device_id": device_id}
        if "slave" in params:
            return {"slave": device_id}
    return {}


class ModbusRtuSerialTransport:
    """Modbus RTU client over RS485/USB serial."""

    def __init__(
        self,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        slave_id: int = DEFAULT_SLAVE_ID,
        timeout: float = DEFAULT_SERIAL_TIMEOUT,
        handle_local_echo: bool = False,
    ):
        try:
            from pymodbus.client import ModbusSerialClient
        except ImportError as exc:
            raise ImportError("RS485 mode requires: pip install pymodbus pyserial") from exc

        self._slave_id = slave_id
        client_kwargs: dict[str, Any] = dict(
            port=port,
            baudrate=baudrate,
            bytesize=8,
            parity="N",
            stopbits=1,
            timeout=timeout,
            handle_local_echo=handle_local_echo,
        )
        init_params = inspect.signature(ModbusSerialClient.__init__).parameters
        if "device_id" in init_params:
            client_kwargs["device_id"] = slave_id
        self._client = ModbusSerialClient(**client_kwargs)
        if not self._client.connect():
            raise ConnectionError(f"Failed to open serial port: {port}")
        # Per-call device_id only needed when client constructor does not take device_id
        if "device_id" in init_params:
            self._device_kwargs: dict[str, int] = {}
        else:
            self._device_kwargs = _pymodbus_device_kwargs(self._client, slave_id)
        time.sleep(INTER_FRAME_DELAY_S)

    def read_holding_register(self, address: int) -> int:
        result = self._client.read_holding_registers(address=address, count=1, **self._device_kwargs)
        if result.isError():
            raise IOError(f"Modbus read failed at 0x{address:04X}: {result}")
        time.sleep(INTER_FRAME_DELAY_S)
        return int(result.registers[0])

    def write_holding_register(self, address: int, value: int, retries: int = 3) -> None:
        last_error: Optional[Exception] = None
        for attempt in range(retries):
            try:
                result = self._client.write_register(address=address, value=value, **self._device_kwargs)
                if result.isError():
                    raise IOError(f"Modbus write failed at 0x{address:04X}: {result}")
                time.sleep(INTER_FRAME_DELAY_S)
                return
            except Exception as exc:
                last_error = exc
                time.sleep(INTER_FRAME_DELAY_S * (attempt + 1))
        raise IOError(f"Modbus write failed at 0x{address:04X} after {retries} attempts") from last_error

    def close(self) -> None:
        self._client.close()


class ModbusRtuUdpTransport:
    """Modbus RTU frames forwarded over UDP (e.g. Adam 485-to-Ethernet gateway)."""

    def __init__(self, host: str, port: int = 2562, slave_id: int = DEFAULT_SLAVE_ID, timeout: float = 2.0):
        self._host = host
        self._port = port
        self._slave_id = slave_id
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(timeout)

    def _exchange(self, request: bytes) -> bytes:
        self._socket.sendto(request, (self._host, self._port))
        response, _ = self._socket.recvfrom(1024)
        if len(response) < 4:
            raise ValueError("Modbus response too short")
        if response[-2:] != _modbus_crc(response[:-2]):
            raise ValueError("Modbus RTU CRC check failed")
        return response

    def read_holding_register(self, address: int) -> int:
        pdu = struct.pack(">BBHH", self._slave_id, 0x03, address, 1)
        response = self._exchange(pdu + _modbus_crc(pdu))
        if response[1] & 0x80:
            raise IOError(f"Modbus exception: {response[2]}")
        return struct.unpack(">H", response[3:5])[0]

    def write_holding_register(self, address: int, value: int) -> None:
        pdu = struct.pack(">BBHH", self._slave_id, 0x06, address, value)
        response = self._exchange(pdu + _modbus_crc(pdu))
        if response[1] & 0x80:
            raise IOError(f"Modbus exception: {response[2]}")

    def close(self) -> None:
        self._socket.close()


# --- High-level gripper API ----------------------------------------------------


class PGC30060Gripper:
    """Driver for DH-Robotics PGC-300-60 (60 mm stroke, Modbus RTU)."""

    def __init__(self, transport: ModbusTransport, standard_position_mapping: bool = False):
        self._transport = transport
        self._force_percent = 50
        self._speed_percent = 50
        if standard_position_mapping:
            self._register_open = POSITION_MIN
            self._register_closed = POSITION_MAX
        else:
            self._register_open = POSITION_REGISTER_OPEN
            self._register_closed = POSITION_REGISTER_CLOSED

    @classmethod
    def from_serial(
        cls,
        port: str,
        baudrate: int = DEFAULT_BAUDRATE,
        slave_id: int = DEFAULT_SLAVE_ID,
        handle_local_echo: bool = False,
        timeout: float = DEFAULT_SERIAL_TIMEOUT,
        standard_position_mapping: bool = False,
    ) -> "PGC30060Gripper":
        return cls(
            ModbusRtuSerialTransport(
                port=port,
                baudrate=baudrate,
                slave_id=slave_id,
                handle_local_echo=handle_local_echo,
                timeout=timeout,
            ),
            standard_position_mapping=standard_position_mapping,
        )

    @classmethod
    def from_udp(
        cls,
        host: str,
        port: int = 2562,
        slave_id: int = DEFAULT_SLAVE_ID,
        standard_position_mapping: bool = False,
    ) -> "PGC30060Gripper":
        return cls(
            ModbusRtuUdpTransport(host=host, port=port, slave_id=slave_id),
            standard_position_mapping=standard_position_mapping,
        )

    def register_value_for_open(self) -> int:
        return self._register_open

    def register_value_for_closed(self) -> int:
        return self._register_closed

    def physical_state(self) -> str:
        pos = self.read_position()
        if abs(pos - self._register_open) <= abs(pos - self._register_closed):
            return "open"
        return "closed"

    def disconnect(self) -> None:
        self._transport.close()

    def require_initialized(self) -> None:
        state = self.read_init_state()
        if state != InitState.INITIALIZED:
            raise RuntimeError(
                f"Gripper is not initialized (0x0200={int(state)}). "
                "Run 'init' after power-on before open/close."
            )

    def read_position_setpoint(self) -> int:
        return self._transport.read_holding_register(REG_POSITION)

    def read_init_direction(self) -> int:
        return self._transport.read_holding_register(REG_INIT_DIRECTION)

    def initialize(self, full_calibration: bool = False, wait: bool = True, timeout: float = 10.0) -> None:
        """Initialize gripper. Must run once after power-on before motion commands."""
        value = INIT_FULL_CALIBRATION if full_calibration else INIT_COMMAND
        self._transport.write_holding_register(REG_INIT, value)
        if wait:
            self.wait_initialized(timeout=timeout)

    def wait_initialized(self, timeout: float = 10.0, poll_interval: float = 0.1) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self.read_init_state()
            if state == InitState.INITIALIZED:
                return
            if state not in (InitState.NOT_INITIALIZED, InitState.INITIALIZING):
                raise RuntimeError(f"Unexpected init state during wait: {state.name} ({int(state)})")
            time.sleep(poll_interval)
        raise TimeoutError("Gripper initialization did not complete in time")

    def set_force(self, percent: Union[int, float], skip_if_unchanged: bool = True) -> None:
        percent = int(round(percent))
        if not FORCE_MIN <= percent <= FORCE_MAX:
            raise ValueError(f"Force must be in [{FORCE_MIN}, {FORCE_MAX}] %")
        if skip_if_unchanged and self._transport.read_holding_register(REG_FORCE) == percent:
            self._force_percent = percent
            return
        self._force_percent = percent
        self._transport.write_holding_register(REG_FORCE, percent)

    def set_speed(self, percent: Union[int, float], skip_if_unchanged: bool = True) -> None:
        percent = int(round(percent))
        if not SPEED_MIN <= percent <= SPEED_MAX:
            raise ValueError(f"Speed must be in [{SPEED_MIN}, {SPEED_MAX}] %")
        if skip_if_unchanged and self._transport.read_holding_register(REG_SPEED) == percent:
            self._speed_percent = percent
            return
        self._speed_percent = percent
        self._transport.write_holding_register(REG_SPEED, percent)

    def apply_motion_config(self, force: int, speed: int) -> None:
        """Set force/speed only when different from current values (fewer RTU writes)."""
        self.set_force(force, skip_if_unchanged=True)
        self.set_speed(speed, skip_if_unchanged=True)

    def set_position(
        self,
        position: Union[int, float],
        wait: bool = False,
        timeout: float = 5.0,
        verify: bool = True,
        position_tolerance: int = 40,
    ) -> GripState:
        """
        Set finger position.

        position: raw register value 0-1000 (use open()/close_gripper() for physical motion).
        """
        self.require_initialized()
        position = int(round(position))
        if not POSITION_MIN <= position <= POSITION_MAX:
            raise ValueError(f"Position must be in [{POSITION_MIN}, {POSITION_MAX}]")

        start_pos = self.read_position()
        self._transport.write_holding_register(REG_POSITION, position)
        time.sleep(0.02)
        setpoint = self.read_position_setpoint()
        if setpoint != position:
            raise IOError(f"Position register write failed: wrote {position}, read back {setpoint}")

        state = GripState.MOVING
        if wait:
            state = self.wait_motion_done(timeout=timeout, target_position=position)

        if verify and abs(start_pos - position) > position_tolerance:
            final_pos = self.read_position()
            if abs(final_pos - position) > position_tolerance:
                raise RuntimeError(
                    "Gripper did not reach the commanded position. "
                    f"start={start_pos}, target={position}, actual={final_pos}, "
                    f"grip_state={self.read_grip_state().name}, "
                    f"init_direction={self.read_init_direction()} (0=open, 1=closed). "
                    "If already at target after init, try 'open' first then 'close'. "
                    "If setpoint changed but fingers did not move, check 24V power and RS485 wiring."
                )
        return state

    def open(self, wait: bool = False, verify: bool = True) -> GripState:
        return self.set_position(self._register_open, wait=wait, verify=verify)

    def close_gripper(self, wait: bool = False, verify: bool = True) -> GripState:
        """Fully close fingers."""
        return self.set_position(self._register_closed, wait=wait, verify=verify)

    def read_init_state_raw(self) -> int:
        return self._transport.read_holding_register(REG_INIT_STATE)

    def read_init_state(self) -> InitState:
        value = self.read_init_state_raw()
        try:
            return InitState(value)
        except ValueError as exc:
            raise ValueError(f"Unknown init state from register 0x0200: {value}") from exc

    def read_grip_state(self) -> GripState:
        return GripState(self._transport.read_holding_register(REG_GRIP_STATE))

    def read_position(self) -> int:
        return self._transport.read_holding_register(REG_POSITION_FB)

    def wait_motion_done(
        self,
        timeout: float = 5.0,
        poll_interval: float = 0.05,
        target_position: Optional[int] = None,
        position_tolerance: int = 40,
    ) -> GripState:
        deadline = time.monotonic() + timeout
        saw_moving = False
        while time.monotonic() < deadline:
            state = self.read_grip_state()
            if state == GripState.MOVING:
                saw_moving = True
            if target_position is not None:
                if abs(self.read_position() - target_position) <= position_tolerance:
                    return self.read_grip_state()
            elif state != GripState.MOVING and saw_moving:
                return state
            elif state != GripState.MOVING and not saw_moving:
                # Never entered MOVING; keep polling in case feedback is slow
                pass
            time.sleep(poll_interval)
        raise TimeoutError("Gripper motion did not finish in time")

    def status_dict(self) -> dict:
        init_dir = self.read_init_direction()
        return {
            "initialized": self.read_init_state().name,
            "grip_state": self.read_grip_state().name,
            "physical_state": self.physical_state(),
            "position_actual": self.read_position(),
            "position_setpoint": self.read_position_setpoint(),
            "register_open": self._register_open,
            "register_closed": self._register_closed,
            "init_direction": init_dir,
            "init_direction_note": "0=home to open, 1=home to closed",
            "force_set": self._transport.read_holding_register(REG_FORCE),
            "speed_set": self._transport.read_holding_register(REG_SPEED),
        }


def print_status(gripper: PGC30060Gripper) -> None:
    for key, val in gripper.status_dict().items():
        print(f"  {key}: {val}")


# --- CLI -----------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Control DH-Robotics PGC-300-60 gripper")
    conn = parser.add_mutually_exclusive_group(required=True)
    conn.add_argument("--port", help="Serial port for RS485 (e.g. /dev/ttyUSB0, COM3)")
    conn.add_argument("--udp-host", help="UDP gateway host (Modbus RTU over UDP)")

    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--slave-id", type=int, default=DEFAULT_SLAVE_ID)
    parser.add_argument("--udp-port", type=int, default=2562)
    parser.add_argument(
        "--local-echo",
        action="store_true",
        help="Enable RS485 USB adapter local-echo handling (try if writes fail or reads are wrong)",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_SERIAL_TIMEOUT, help="Serial timeout (seconds)")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Do not verify finger position reached target after motion",
    )
    parser.add_argument(
        "--skip-config",
        action="store_true",
        help="Skip force/speed writes (only move position; use if force/speed writes timeout)",
    )
    parser.add_argument(
        "--standard-position",
        action="store_true",
        help="Use manual mapping: register 0=open, 1000=closed (default on this unit: 1000=open, 0=closed)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="Initialize gripper (required after power-on)")
    sub.add_parser("recalibrate", help="Full calibration (write 0xA5 to init register)")
    sub.add_parser("open", help="Fully open")
    sub.add_parser("close", help="Fully close")
    sub.add_parser("status", help="Print gripper status")

    pos = sub.add_parser("set-position", help="Move to position 0-1000")
    pos.add_argument("value", type=int)

    demo = sub.add_parser("demo", help="Open/close cycle for connection test")
    demo.add_argument("--cycles", type=int, default=3)

    for name in ("init", "open", "close", "set-position", "demo"):
        p = sub.choices[name]
        p.add_argument("--force", type=int, default=50, help="Grip force 20-100 %%")
        p.add_argument("--speed", type=int, default=50, help="Motion speed 1-100 %%")
        p.add_argument("--wait", action="store_true", help="Wait until motion completes")

    return parser


def main() -> None:
    args = _build_parser().parse_args()

    verify_motion = not args.no_verify

    if args.port:
        gripper: Optional[PGC30060Gripper] = PGC30060Gripper.from_serial(
            port=args.port,
            baudrate=args.baudrate,
            slave_id=args.slave_id,
            handle_local_echo=args.local_echo,
            timeout=args.timeout,
            standard_position_mapping=args.standard_position,
        )
    else:
        gripper = PGC30060Gripper.from_udp(
            host=args.udp_host,
            port=args.udp_port,
            slave_id=args.slave_id,
            standard_position_mapping=args.standard_position,
        )

    try:
        def maybe_apply_config() -> None:
            if not args.skip_config:
                gripper.apply_motion_config(args.force, args.speed)

        if args.command == "init":
            maybe_apply_config()
            gripper.initialize(full_calibration=False)
            print("Initialization completed. Status:")
            print_status(gripper)
        elif args.command == "recalibrate":
            maybe_apply_config()
            gripper.initialize(full_calibration=True)
            print("Full recalibration started.")
        elif args.command == "open":
            print("Before:")
            print_status(gripper)
            maybe_apply_config()
            gripper.open(wait=args.wait, verify=verify_motion)
            print("After:")
            print_status(gripper)
            print("Command: open")
        elif args.command == "close":
            print("Before:")
            print_status(gripper)
            maybe_apply_config()
            gripper.close_gripper(wait=args.wait, verify=verify_motion)
            print("After:")
            print_status(gripper)
            print("Command: close")
        elif args.command == "set-position":
            print("Before:")
            print_status(gripper)
            maybe_apply_config()
            gripper.set_position(args.value, wait=args.wait, verify=verify_motion)
            print("After:")
            print_status(gripper)
            print(f"Command: position={args.value}")
        elif args.command == "status":
            print_status(gripper)
        elif args.command == "demo":
            maybe_apply_config()
            gripper.initialize()
            for i in range(args.cycles):
                print(f"Cycle {i + 1}/{args.cycles}: open")
                gripper.open(wait=True, verify=verify_motion)
                time.sleep(0.5)
                print(f"Cycle {i + 1}/{args.cycles}: close")
                gripper.close_gripper(wait=True, verify=verify_motion)
                time.sleep(0.5)
            print("Demo finished.")
    finally:
        gripper.disconnect()


if __name__ == "__main__":
    main()
