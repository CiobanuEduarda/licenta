"""LED output via serial, or a no-op sink for headless runs and tests."""

from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

import serial

from glowmind.config import Settings

log = logging.getLogger(__name__)


@runtime_checkable
class LedSink(Protocol):
    def send_rgb(self, r: int, g: int, b: int) -> None: ...
    def close(self) -> None: ...


class NullLedSink:
    __slots__ = ()

    def send_rgb(self, r: int, g: int, b: int) -> None:
        return

    def close(self) -> None:
        return


class SerialLedSink:
    __slots__ = ("_ser",)

    def __init__(self, ser: serial.Serial) -> None:
        self._ser = ser

    def send_rgb(self, r: int, g: int, b: int) -> None:
        self._ser.write(f"{r},{g},{b}\n".encode())

    def close(self) -> None:
        self._ser.close()


def open_led_sink(settings: Settings, *, settle_s: float = 2.0) -> LedSink:
    try:
        ser = serial.Serial(settings.serial_port, settings.serial_baud)
        time.sleep(settle_s)
        log.info("LED serial connected on %s", settings.serial_port)
        return SerialLedSink(ser)
    except Exception as e:
        log.warning("LED serial not available (preview still runs): %s", e)
        return NullLedSink()
