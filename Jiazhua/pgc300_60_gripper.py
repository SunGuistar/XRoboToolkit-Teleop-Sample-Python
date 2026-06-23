"""
CLI for DH-Robotics PGC-300-60 gripper.

Driver implementation: xrobotoolkit_teleop.hardware.interface.pgc300_60_driver
"""

from xrobotoolkit_teleop.hardware.interface.pgc300_60_driver import (
    DEFAULT_BAUDRATE,
    DEFAULT_SERIAL_TIMEOUT,
    PGC30060Gripper,
    print_status,
)
def main() -> None:
    import argparse
    import time
    from typing import Optional

    parser = argparse.ArgumentParser(description="Control DH-Robotics PGC-300-60 gripper")
    conn = parser.add_mutually_exclusive_group(required=True)
    conn.add_argument("--port", help="Serial port for RS485 (e.g. /dev/ttyUSB0, COM3)")
    conn.add_argument("--udp-host", help="UDP gateway host (Modbus RTU over UDP)")

    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--slave-id", type=int, default=1)
    parser.add_argument("--udp-port", type=int, default=2562)
    parser.add_argument("--local-echo", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_SERIAL_TIMEOUT)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--skip-config", action="store_true")
    parser.add_argument("--standard-position", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("recalibrate")
    sub.add_parser("open")
    sub.add_parser("close")
    sub.add_parser("status")
    pos = sub.add_parser("set-position")
    pos.add_argument("value", type=int)
    demo = sub.add_parser("demo")
    demo.add_argument("--cycles", type=int, default=3)
    for name in ("init", "open", "close", "set-position", "demo"):
        p = sub.choices[name]
        p.add_argument("--force", type=int, default=50)
        p.add_argument("--speed", type=int, default=50)
        p.add_argument("--wait", action="store_true")

    args = parser.parse_args()
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
            gripper.initialize(full_calibration=False)
            maybe_apply_config()
            gripper.open(wait=args.wait, verify=verify_motion)
            print("Initialization completed. Status:")
            print_status(gripper)
        elif args.command == "recalibrate":
            gripper.initialize(full_calibration=True)
            maybe_apply_config()
            print("Full recalibration started.")
        elif args.command == "open":
            print("Before:")
            print_status(gripper)
            maybe_apply_config()
            gripper.open(wait=args.wait, verify=verify_motion)
            print("After:")
            print_status(gripper)
        elif args.command == "close":
            print("Before:")
            print_status(gripper)
            maybe_apply_config()
            gripper.close_gripper(wait=args.wait, verify=verify_motion)
            print("After:")
            print_status(gripper)
        elif args.command == "set-position":
            print("Before:")
            print_status(gripper)
            maybe_apply_config()
            gripper.set_position(args.value, wait=args.wait, verify=verify_motion)
            print("After:")
            print_status(gripper)
        elif args.command == "status":
            print_status(gripper)
        elif args.command == "demo":
            gripper.initialize()
            maybe_apply_config()
            gripper.open(wait=True, verify=verify_motion)
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
