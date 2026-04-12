"""Runtime configuration (env-backed defaults)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw, 10)
    except ValueError as e:
        raise ValueError(f"Invalid integer for {key}={raw!r}") from e


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"Invalid float for {key}={raw!r}") from e


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _parse_cors_origins(value: str) -> list[str]:
    v = value.strip()
    if v == "*":
        return ["*"]
    return [x.strip() for x in v.split(",") if x.strip()]


def _validate(settings: Settings) -> None:
    if settings.face_bbox_buffer < 0:
        raise ValueError("face_bbox_buffer must be >= 0")
    if not 0.0 <= settings.led_brightness <= 1.0:
        raise ValueError("led_brightness must be in [0, 1]")
    if not 0.0 < settings.ema_alpha <= 1.0:
        raise ValueError("ema_alpha must be in (0, 1]")
    if settings.transition_speed <= 0:
        raise ValueError("transition_speed must be > 0")
    if settings.frame_blend <= 0:
        raise ValueError("frame_blend must be > 0")
    if settings.camera_index < 0 or settings.camera_fallback_index < 0:
        raise ValueError("camera indices must be >= 0")
    if settings.serial_baud <= 0:
        raise ValueError("serial_baud must be > 0")
    if settings.api_enabled and not (1 <= settings.api_port <= 65535):
        raise ValueError("api_port must be between 1 and 65535")


@dataclass(frozen=True)
class Settings:
    """Tunable parameters for inference, display, and hardware."""

    device: str = "cpu"
    model_weights: str = "resnet50_va_finetune.pth"
    serial_port: str = "/dev/cu.usbmodem11101"
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
    # Local HTTP API (REST + WebSocket); bind is localhost-oriented by default.
    api_enabled: bool = True
    api_host: str = "127.0.0.1"
    api_port: int = 8765
    api_cors_origins: str = "*"

    @classmethod
    def from_env(cls) -> Settings:
        """Load settings from environment variables (see source for keys)."""
        s = cls(
            device=_env_str("DEVICE", "cpu"),
            model_weights=_env_str("MODEL_WEIGHTS", "resnet50_va_finetune.pth"),
            serial_port=_env_str("SERIAL_PORT", "/dev/cu.usbmodem11101"),
            serial_baud=_env_int("SERIAL_BAUD", 115200),
            camera_index=_env_int("CAMERA_INDEX", 1),
            camera_fallback_index=_env_int("CAMERA_FALLBACK_INDEX", 0),
            face_bbox_buffer=_env_float("FACE_BBOX_BUFFER", 0.1),
            valence_offset=_env_float("VALENCE_OFFSET", 0.0),
            valence_gain=_env_float("VALENCE_GAIN", 2.5),
            arousal_offset=_env_float("AROUSAL_OFFSET", 0.0),
            arousal_gain=_env_float("AROUSAL_GAIN", 2.0),
            led_brightness=_env_float("LED_BRIGHTNESS", 0.22),
            ema_alpha=_env_float("EMA_ALPHA", 0.25),
            transition_speed=_env_float("TRANSITION_SPEED", 4.0),
            frame_blend=_env_float("FRAME_BLEND", 0.033),
            api_enabled=_env_bool("API_ENABLED", True),
            api_host=_env_str("API_HOST", "127.0.0.1"),
            api_port=_env_int("API_PORT", 8765),
            api_cors_origins=_env_str("API_CORS_ORIGINS", "*"),
        )
        _validate(s)
        return s

    def cors_origin_list(self) -> list[str]:
        return _parse_cors_origins(self.api_cors_origins)
