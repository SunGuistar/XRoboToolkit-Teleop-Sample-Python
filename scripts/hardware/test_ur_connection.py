"""
Minimal RTDE connectivity test for a UR arm (no VR, no Placo, no gripper).

Use this before running teleop_single_ur5e_hardware.py to verify PC <-> robot communication.
"""

import sys
import time

import numpy as np
import rtde_control
import rtde_receive
import tyro

from xrobotoolkit_teleop.hardware.interface.universal_robots import DEFAULT_ROBOT_IP

UR5E_JOINT_NAMES = (
    "base (J0)",
    "shoulder_lift (J1)",
    "elbow (J2)",
    "wrist_1 (J3)",
    "wrist_2 (J4)",
    "wrist_3 (J5)",
)

# Small per-joint deltas (degrees): base/shoulder smaller for safety.
DEFAULT_JOINT_DELTAS_DEG = (5.0, 4.0, 4.0, 5.0, 5.0, 8.0)


def _print_robot_state(rtde_r: rtde_receive.RTDEReceiveInterface) -> None:
    q = np.array(rtde_r.getActualQ())
    qd = np.array(rtde_r.getActualQd())
    tcp = np.array(rtde_r.getActualTCPPose())
    print(f"  Joint q (deg):  {np.rad2deg(q)}")
    print(f"  Joint qd (deg/s): {np.rad2deg(qd)}")
    print(f"  TCP pose [x,y,z,rx,ry,rz]: {tcp}")
    try:
        print(f"  Robot mode: {rtde_r.getRobotMode()}")
        print(f"  Safety mode: {rtde_r.getSafetyMode()}")
        print(f"  Protective stop: {rtde_r.isProtectiveStopped()}")
        print(f"  Emergency stop: {rtde_r.isEmergencyStopped()}")
    except Exception as e:
        print(f"  (Extra state unavailable: {e})")


def _check_safe(rtde_r: rtde_receive.RTDEReceiveInterface) -> None:
    if rtde_r.isEmergencyStopped():
        raise RuntimeError("Emergency stop active.")
    if rtde_r.isProtectiveStopped():
        raise RuntimeError("Protective stop active.")


def _wait_until_q(
    rtde_r: rtde_receive.RTDEReceiveInterface,
    q_target: np.ndarray,
    timeout_s: float,
    tol_deg: float,
) -> bool:
    """Wait until actual joints are close to q_target."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        _check_safe(rtde_r)
        q = np.array(rtde_r.getActualQ())
        err_deg = np.max(np.abs(np.rad2deg(q - q_target)))
        if err_deg <= tol_deg:
            return True
        time.sleep(0.05)
    return False


def _movej_and_wait(
    rtde_c: rtde_control.RTDEControlInterface,
    rtde_r: rtde_receive.RTDEReceiveInterface,
    q_target: np.ndarray,
    move_speed: float,
    move_acceleration: float,
    wait_timeout_s: float,
    position_tol_deg: float,
) -> None:
    _check_safe(rtde_r)
    ok = rtde_c.moveJ(q_target, move_speed, move_acceleration)
    if not ok:
        raise RuntimeError("moveJ command rejected (is Remote control enabled?)")
    if not _wait_until_q(rtde_r, q_target, wait_timeout_s, position_tol_deg):
        q_actual = np.rad2deg(rtde_r.getActualQ())
        q_cmd = np.rad2deg(q_target)
        raise RuntimeError(
            f"Timed out waiting for joints (>{wait_timeout_s}s).\n"
            f"  Commanded (deg): {q_cmd}\n"
            f"  Actual   (deg): {q_actual}"
        )


def _run_all_joints_demo(
    rtde_c: rtde_control.RTDEControlInterface,
    rtde_r: rtde_receive.RTDEReceiveInterface,
    joint_deltas_deg: tuple[float, ...],
    move_speed: float,
    move_acceleration: float,
    wait_timeout_s: float,
    position_tol_deg: float,
    min_motion_deg: float,
    pause_between_joints_s: float,
    settle_s: float,
) -> None:
    """Move each joint slightly from a fixed home pose, then return home after every step."""
    if len(joint_deltas_deg) != 6:
        raise ValueError("joint_deltas_deg must have 6 values for UR5e.")

    print("\n[2b] All-joints demo motion (slow, small relative moves)")
    print("  SAFETY: Clear the workspace. Keep E-stop reachable. Stay outside reach.")
    print("  Starting in 3 seconds... (Ctrl+C to abort)")
    try:
        time.sleep(3.0)
    except KeyboardInterrupt:
        print("Aborted by user.")
        raise SystemExit(0) from None

    q_home = np.array(rtde_r.getActualQ())
    print(f"  Home pose (deg): {np.rad2deg(q_home)}")

    for joint_index, (joint_name, delta_deg) in enumerate(zip(UR5E_JOINT_NAMES, joint_deltas_deg)):
        q_offset = q_home.copy()
        q_offset[joint_index] += np.deg2rad(delta_deg)

        print(
            f"\n  -> Joint {joint_index} {joint_name}: "
            f"{np.rad2deg(q_home[joint_index]):.1f}° + {delta_deg:.1f}°"
        )
        _movej_and_wait(
            rtde_c,
            rtde_r,
            q_offset,
            move_speed,
            move_acceleration,
            wait_timeout_s,
            position_tol_deg,
        )
        time.sleep(settle_s)

        q_at_offset = np.array(rtde_r.getActualQ())
        moved_deg = np.rad2deg(q_at_offset[joint_index] - q_home[joint_index])
        print(f"     Verified delta on J{joint_index}: {moved_deg:+.2f}° (others should stay ~0)")

        if abs(moved_deg) < min_motion_deg:
            raise RuntimeError(
                f"Joint {joint_index} moved only {moved_deg:.2f}° (< {min_motion_deg}°). "
                "Check remote control / brakes / limits."
            )

        print(f"  <- Joint {joint_index}: return to home")
        _movej_and_wait(
            rtde_c,
            rtde_r,
            q_home,
            move_speed,
            move_acceleration,
            wait_timeout_s,
            position_tol_deg,
        )
        time.sleep(pause_between_joints_s)

    err_home = np.max(np.abs(np.rad2deg(np.array(rtde_r.getActualQ()) - q_home)))
    print(f"\nOK: all-joints demo finished. Max deviation from start pose: {err_home:.2f}°")


def _run_single_joint_demo(
    rtde_c: rtde_control.RTDEControlInterface,
    rtde_r: rtde_receive.RTDEReceiveInterface,
    joint_index: int,
    delta_deg: float,
    move_speed: float,
    move_acceleration: float,
    wait_timeout_s: float,
    position_tol_deg: float,
    min_motion_deg: float,
) -> None:
    q_home = np.array(rtde_r.getActualQ())
    q_out = q_home.copy()
    q_out[joint_index] += np.deg2rad(delta_deg)

    print(f"\n[2b] Single-joint demo: J{joint_index} delta {delta_deg:+.1f}°")
    print("  SAFETY: Clear the workspace. E-stop reachable. Starting in 2 s...")
    time.sleep(2.0)

    _movej_and_wait(rtde_c, rtde_r, q_out, move_speed, move_acceleration, wait_timeout_s, position_tol_deg)
    moved = np.rad2deg(rtde_r.getActualQ()[joint_index] - q_home[joint_index])
    if abs(moved) < min_motion_deg:
        raise RuntimeError(f"Joint {joint_index} moved only {moved:.2f}°")
    _movej_and_wait(rtde_c, rtde_r, q_home, move_speed, move_acceleration, wait_timeout_s, position_tol_deg)
    print("OK: single-joint demo finished.")


def main(
    robot_ip: str = DEFAULT_ROBOT_IP,
    test_control: bool = True,
    demo_motion: bool = True,
    demo_all_joints: bool = True,
    demo_joint: int = 5,
    demo_delta_deg: float = 15.0,
    move_speed: float = 0.15,
    move_acceleration: float = 0.25,
    wait_timeout_s: float = 25.0,
    position_tol_deg: float = 1.5,
    min_motion_deg: float = 1.0,
    pause_between_joints_s: float = 0.5,
    settle_s: float = 0.4,
    test_servo_hold: bool = False,
):
    """
    Args:
        robot_ip: UR controller IP address.
        test_control: Open RTDEControlInterface (required for motion).
        demo_motion: Run a slow relative motion test after connecting.
        demo_all_joints: Cycle J0–J5 with small offsets (default, safer visibility).
        demo_joint: Used only when demo_all_joints=False.
        demo_delta_deg: Offset for single-joint mode (degrees).
        move_speed: moveJ speed (rad/s); kept low for safety.
        move_acceleration: moveJ acceleration (rad/s^2).
        wait_timeout_s: Max wait per move until joints reach target.
        position_tol_deg: Position tolerance to consider a move complete.
        min_motion_deg: Fail if a joint moves less than this (detect false OK).
        pause_between_joints_s: Pause between joint tests.
        settle_s: Pause at offset pose before measuring.
        test_servo_hold: Optional servoJ hold test.
    """
    print(f"=== UR RTDE test @ {robot_ip} ===\n")

    print("[1/3] RTDE receive (read-only)...")
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface(robot_ip)
    except Exception as e:
        print(f"FAIL: cannot open RTDE receive: {e}")
        sys.exit(1)
    print("OK: RTDE receive connected.")
    _print_robot_state(rtde_r)

    rtde_c = None
    if test_control:
        print("\n[2/3] RTDE control...")
        try:
            rtde_c = rtde_control.RTDEControlInterface(robot_ip)
        except Exception as e:
            print(f"FAIL: cannot open RTDE control: {e}")
            rtde_r.disconnect()
            sys.exit(1)
        print("OK: RTDE control connected (external control script running).")

        if demo_motion:
            try:
                if demo_all_joints:
                    _run_all_joints_demo(
                        rtde_c,
                        rtde_r,
                        DEFAULT_JOINT_DELTAS_DEG,
                        move_speed,
                        move_acceleration,
                        wait_timeout_s,
                        position_tol_deg,
                        min_motion_deg,
                        pause_between_joints_s,
                        settle_s,
                    )
                else:
                    _run_single_joint_demo(
                        rtde_c,
                        rtde_r,
                        demo_joint,
                        demo_delta_deg,
                        move_speed,
                        move_acceleration,
                        wait_timeout_s,
                        position_tol_deg,
                        min_motion_deg,
                    )
            except RuntimeError as e:
                print(f"FAIL: demo motion: {e}")
                sys.exit(1)

        if test_servo_hold:
            print("\n[2c] servoJ hold at current joints for 1 second...")
            q_hold = np.array(rtde_r.getActualQ())
            try:
                for _ in range(60):
                    t_start = rtde_c.initPeriod()
                    ok = rtde_c.servoJ(q_hold, 0.2, 0.3, 0.017, 0.1, 300)
                    if not ok:
                        print("FAIL: servoJ returned False.")
                        break
                    rtde_c.waitPeriod(t_start)
                else:
                    print("OK: servoJ hold finished.")
            except Exception as e:
                print(f"FAIL during servoJ: {e}")
            finally:
                try:
                    rtde_c.servoStop()
                except Exception:
                    pass

    print("\n[3/3] Cleanup...")
    if rtde_c is not None:
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        try:
            rtde_c.disconnect()
        except Exception:
            pass
    try:
        rtde_r.disconnect()
    except Exception:
        pass

    print("\nDone. PC <-> arm RTDE read/write and motion verified.")
    print("Next teleop (no --reset):")
    print("  python scripts/hardware/teleop_single_ur5e_hardware.py")


if __name__ == "__main__":
    tyro.cli(main)
