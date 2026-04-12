"""Configuration parsing and validation."""

from __future__ import annotations

import pytest

from glowmind.config import Settings


def test_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "DEVICE",
        "MODEL_WEIGHTS",
        "SERIAL_PORT",
        "SERIAL_BAUD",
        "CAMERA_INDEX",
        "CAMERA_FALLBACK_INDEX",
        "FACE_BBOX_BUFFER",
        "VALENCE_OFFSET",
        "VALENCE_GAIN",
        "AROUSAL_OFFSET",
        "AROUSAL_GAIN",
        "LED_BRIGHTNESS",
        "EMA_ALPHA",
        "TRANSITION_SPEED",
        "FRAME_BLEND",
    ):
        monkeypatch.delenv(key, raising=False)
    s = Settings.from_env()
    assert s.device == "cpu"
    assert s.camera_index == 1


def test_from_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("CAMERA_INDEX", "2")
    monkeypatch.setenv("LED_BRIGHTNESS", "0.5")
    s = Settings.from_env()
    assert s.device == "cuda"
    assert s.camera_index == 2
    assert s.led_brightness == 0.5


def test_from_env_rejects_bad_float(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMA_ALPHA", "not-a-float")
    with pytest.raises(ValueError, match="EMA_ALPHA"):
        Settings.from_env()


def test_from_env_rejects_ema_alpha_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMA_ALPHA", "0")
    with pytest.raises(ValueError, match="ema_alpha"):
        Settings.from_env()
