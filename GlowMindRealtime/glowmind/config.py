"""Runtime configuration (env-backed defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Tunable parameters for inference, display, and hardware."""

    device: str = "cpu"
    model_weights: str = "resnet50_va_finetune.pth"
    serial_port: str = "/dev/cu.usbmodem1101"
    serial_baud: int = 115200
    camera_index: int = 1
    camera_fallback_index: int = 0
    # Match AffectNet crop margin in training (face box + buffer around Haar box).
    face_bbox_buffer: float = 0.1
    valence_offset: float = 0.0
    valence_gain: float = 2.5
    arousal_offset: float = 0.0
    arousal_gain: float = 2.0
    led_brightness: float = 0.22
    ema_alpha: float = 0.25
    transition_speed: float = 4.0
    frame_blend: float = 0.033  # ~30 FPS assumption for color lerp

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            model_weights=os.environ.get("MODEL_WEIGHTS", "resnet50_va_finetune.pth"),
            serial_port=os.environ.get("SERIAL_PORT", "/dev/cu.usbmodem1101"),
            face_bbox_buffer=float(os.environ.get("FACE_BBOX_BUFFER", "0.1")),
        )
