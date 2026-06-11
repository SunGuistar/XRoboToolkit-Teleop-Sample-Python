"""Normalized gripper values for BC: 0=open, 1=close."""

from __future__ import annotations


def register_to_normalized(register_value: int, register_open: int, register_closed: int) -> float:
    """Map DH gripper register position to [0, 1]. 0=open, 1=close."""
    span = float(register_closed - register_open)
    if span == 0.0:
        return 0.0
    raw = (float(register_value) - float(register_open)) / span
    return float(max(0.0, min(1.0, raw)))


def trigger_to_gripper_cmd(trigger_val: float) -> float:
    """VR trigger → gripper_cmd. Released (0)=open, pressed (1)=close."""
    return float(max(0.0, min(1.0, trigger_val)))
