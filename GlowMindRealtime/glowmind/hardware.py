"""Optional serial LED output."""

from __future__ import annotations

import time
from typing import Optional

import serial


def open_serial(port: str, baud: int, settle_s: float = 2.0) -> Optional[serial.Serial]:
    try:
        ser = serial.Serial(port, baud)
        time.sleep(settle_s)
        print("LED serial connected")
        return ser
    except Exception as e:
        print("LED serial not available (camera will still run):", e)
        return None


def send_rgb(ser: Optional[serial.Serial], r: int, g: int, b: int) -> None:
    if ser is None:
        return
    ser.write(f"{r},{g},{b}\n".encode())
