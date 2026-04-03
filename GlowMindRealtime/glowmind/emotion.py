"""Valence/arousal → emotion labels, colors, and LED pulse animation."""

from __future__ import annotations

import math
from typing import Tuple

RGB = Tuple[int, int, int]

# (valence, arousal) zones → label; colors are RGB for display/LED math
EMOTIONS: dict[str, RGB] = {
    "sad": (30, 60, 180),
    "neutral": (50, 200, 100),
    "happy": (255, 200, 50),
    "calm": (50, 180, 200),
    "angry": (255, 80, 40),
    "anxious": (180, 60, 220),
    "excited": (255, 100, 150),
    "tired": (80, 80, 120),
}

# Seconds per full breathing cycle per label
EMOTION_PULSE_SPEED: dict[str, float] = {
    "sad": 2.5,
    "neutral": 3.0,
    "happy": 1.2,
    "calm": 3.5,
    "angry": 0.9,
    "anxious": 0.7,
    "excited": 1.0,
    "tired": 4.0,
}

AGITATED_EMOTIONS = frozenset({"anxious", "angry"})


def ema(previous: float, new: float, alpha: float = 0.25) -> float:
    return alpha * new + (1.0 - alpha) * previous


def va_to_emotion(valence: float, arousal: float) -> str:
    """Map continuous valence/arousal to a discrete emotion label."""
    v = max(-1.0, min(1.0, valence))
    a = max(-1.0, min(1.0, arousal))
    if abs(v) < 0.35 and abs(a) < 0.35:
        return "neutral"
    if v >= 0.4 and a >= 0.4:
        return "excited" if a > 0.6 else "happy"
    if v >= 0.4 and a <= -0.3:
        return "calm"
    if v <= -0.4 and a >= 0.4:
        return "anxious" if a > 0.5 else "angry"
    if v <= -0.4 and a <= -0.3:
        return "tired" if a < -0.5 else "sad"
    if v > 0:
        return "happy" if a > 0 else "calm"
    return "angry" if a > 0 else "sad"


def lerp_rgb(rgb1: RGB, rgb2: RGB, t: float) -> RGB:
    t = max(0.0, min(1.0, t))
    return (
        int(rgb1[0] + (rgb2[0] - rgb1[0]) * t),
        int(rgb1[1] + (rgb2[1] - rgb1[1]) * t),
        int(rgb1[2] + (rgb2[2] - rgb1[2]) * t),
    )


def apply_pulse(
    rgb: RGB,
    t: float,
    speed: float,
    min_bright: float = 0.45,
    max_bright: float = 1.0,
) -> RGB:
    phase = (t / speed) * 2 * math.pi
    factor = min_bright + (max_bright - min_bright) * (0.5 + 0.5 * math.sin(phase))
    return (
        int(rgb[0] * factor),
        int(rgb[1] * factor),
        int(rgb[2] * factor),
    )


def apply_agitated_pulse(
    rgb: RGB,
    t: float,
    speed: float,
    min_bright: float = 0.5,
    max_bright: float = 1.0,
) -> RGB:
    phase = (t / speed) * 2 * math.pi
    wave1 = 0.5 + 0.5 * math.sin(phase)
    wave2 = 0.5 + 0.5 * math.sin(phase * 2.3 + 0.7)
    factor = min_bright + (max_bright - min_bright) * (wave1 * 0.6 + wave2 * 0.4)
    return (
        int(rgb[0] * factor),
        int(rgb[1] * factor),
        int(rgb[2] * factor),
    )


def pulse_for_emotion(rgb: RGB, t: float, emotion: str) -> RGB:
    speed = EMOTION_PULSE_SPEED.get(emotion, 2.0)
    if emotion in AGITATED_EMOTIONS:
        return apply_agitated_pulse(rgb, t, speed)
    return apply_pulse(rgb, t, speed)


def pulse_neutral_idle(rgb: RGB, t: float) -> RGB:
    return apply_pulse(rgb, t, 3.0, min_bright=0.25, max_bright=0.6)


def scale_for_led(r: int, g: int, b: int, brightness: float) -> tuple[int, int, int]:
    return int(r * brightness), int(g * brightness), int(b * brightness)
