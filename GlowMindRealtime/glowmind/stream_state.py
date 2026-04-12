"""Thread-safe snapshot of the latest pipeline outputs for REST/WebSocket clients."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LiveSnapshot:
    """Immutable view suitable for JSON serialization."""

    face_active: bool
    emotion: str
    valence_smoothed: float
    arousal_smoothed: float
    valence_display: float
    arousal_display: float
    led_r: int
    led_g: int
    led_b: int
    t: float  # time.time() when the snapshot was taken

    def to_json(self) -> dict:
        return asdict(self)


class LiveState:
    """Written by the camera thread; read by FastAPI handlers."""

    __slots__ = ("_lock", "_snap")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snap = LiveSnapshot(
            face_active=False,
            emotion="neutral",
            valence_smoothed=0.0,
            arousal_smoothed=0.0,
            valence_display=0.0,
            arousal_display=0.0,
            led_r=0,
            led_g=0,
            led_b=0,
            t=time.time(),
        )

    def update(
        self,
        *,
        face_active: bool,
        emotion: str,
        valence_smoothed: float,
        arousal_smoothed: float,
        valence_display: float,
        arousal_display: float,
        led_r: int,
        led_g: int,
        led_b: int,
    ) -> None:
        with self._lock:
            self._snap = LiveSnapshot(
                face_active=face_active,
                emotion=emotion,
                valence_smoothed=valence_smoothed,
                arousal_smoothed=arousal_smoothed,
                valence_display=valence_display,
                arousal_display=arousal_display,
                led_r=led_r,
                led_g=led_g,
                led_b=led_b,
                t=time.time(),
            )

    def to_json(self) -> dict:
        with self._lock:
            return self._snap.to_json()
