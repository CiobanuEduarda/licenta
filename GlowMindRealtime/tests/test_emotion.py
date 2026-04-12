"""Unit tests for valence/arousal mapping and smoothing helpers."""

from __future__ import annotations

import pytest

from glowmind.emotion import ema, lerp_rgb, va_to_emotion


@pytest.mark.parametrize(
    ("v", "a", "expected"),
    [
        (0.0, 0.0, "neutral"),
        (0.9, 0.9, "excited"),
        (0.9, 0.45, "happy"),
        (0.9, -0.5, "calm"),
        (-0.9, 0.55, "anxious"),
        (-0.9, 0.45, "angry"),
        (-0.9, -0.55, "tired"),
        (-0.9, -0.45, "sad"),
    ],
)
def test_va_to_emotion_zones(v: float, a: float, expected: str) -> None:
    assert va_to_emotion(v, a) == expected


def test_ema_first_step() -> None:
    assert ema(0.0, 1.0, alpha=0.25) == 0.25


def test_lerp_rgb_endpoints() -> None:
    a = (0, 0, 0)
    b = (100, 200, 255)
    assert lerp_rgb(a, b, 0.0) == a
    assert lerp_rgb(a, b, 1.0) == b
    assert lerp_rgb(a, b, 0.5) == (50, 100, 127)
