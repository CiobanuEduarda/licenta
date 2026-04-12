"""Thread-safe live state for API consumers."""

from __future__ import annotations

from glowmind.stream_state import LiveState


def test_live_state_update_and_json() -> None:
    live = LiveState()
    live.update(
        face_active=True,
        emotion="calm",
        valence_smoothed=0.12,
        arousal_smoothed=-0.34,
        valence_display=0.5,
        arousal_display=-0.5,
        led_r=10,
        led_g=20,
        led_b=30,
    )
    d = live.to_json()
    assert d["face_active"] is True
    assert d["emotion"] == "calm"
    assert d["valence_smoothed"] == 0.12
    assert d["arousal_display"] == -0.5
    assert d["led_r"] == 10
    assert "t" in d
