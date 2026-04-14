"""Session timing, discrete emotion buckets, and timeline samples."""

from __future__ import annotations

import time

from glowmind.session_stats import SessionStats


def test_session_no_tick_before_start() -> None:
    s = SessionStats()
    s.tick(face_active=True, emotion="happy")
    out = s.summary()
    assert out["phase"] == "idle"
    assert out["recording"] is False
    assert out["emotion_pct"] == {}
    assert out["timeline"] == []


def test_session_percentages_sum() -> None:
    s = SessionStats()
    s.start_session()
    s.tick(face_active=True, emotion="a")
    time.sleep(0.04)
    s.tick(face_active=False, emotion="neutral")
    time.sleep(0.04)
    s.tick(face_active=False, emotion="neutral")
    out = s.summary()
    assert out["phase"] == "running"
    assert out["recording"] is True
    em = sum(out["emotion_pct"].values())
    assert 99.0 <= em <= 100.01
    assert len(out["timeline"]) >= 1


def test_stop_freezes_and_timeline_preserved() -> None:
    s = SessionStats()
    s.start_session()
    s.tick(face_active=True, emotion="happy")
    time.sleep(0.06)
    s.tick(face_active=True, emotion="happy")
    mid = s.summary()
    assert mid["recording"] is True
    s.stop_session()
    s.tick(face_active=True, emotion="angry")
    after = s.summary()
    assert after["phase"] == "stopped"
    assert after["recording"] is False
    assert "happy" in after["emotion_pct"]
    assert "angry" not in after["emotion_pct"]
    assert len(after["timeline"]) >= len(mid["timeline"])


def test_stop_triggers_on_stop_callback() -> None:
    seen: list[dict] = []

    def on_stop(payload: dict) -> None:
        seen.append(payload)

    s = SessionStats(on_stop=on_stop)
    s.start_session()
    s.tick(face_active=True, emotion="happy")
    time.sleep(0.03)
    s.tick(face_active=True, emotion="happy")
    s.stop_session()

    assert len(seen) == 1
    assert seen[0]["phase"] == "stopped"
