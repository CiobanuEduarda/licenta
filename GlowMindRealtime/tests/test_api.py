"""FastAPI REST and WebSocket surface (in-memory, no camera)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from glowmind.api import create_app
from glowmind.session_stats import SessionStats
from glowmind.stream_state import LiveState


def test_dashboard_root_serves_html() -> None:
    live = LiveState()
    stats = SessionStats()
    app = create_app(live, stats, cors_origins=["*"])
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
    assert "WebSocket" in r.text
    assert 'id="emotion"' in r.text
    assert "Stop session" in r.text


def test_health_and_state_rest() -> None:
    live = LiveState()
    live.update(
        face_active=True,
        emotion="happy",
        valence_smoothed=0.1,
        arousal_smoothed=0.2,
        valence_display=0.3,
        arousal_display=0.4,
        led_r=1,
        led_g=2,
        led_b=3,
    )
    stats = SessionStats()
    app = create_app(live, stats, cors_origins=["*"])
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    body = client.get("/state").json()
    assert body["emotion"] == "happy"
    assert body["face_active"] is True
    assert body["led_b"] == 3


def test_websocket_receives_json() -> None:
    live = LiveState()
    stats = SessionStats()
    app = create_app(live, stats, cors_origins=["*"])
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
    assert "live" in msg and "session" in msg
    assert msg["live"]["emotion"] == "neutral"
    assert "valence_display" in msg["live"]
    assert "emotion_pct" in msg["session"]
    assert "timeline" in msg["session"]
    assert msg["session"]["phase"] == "idle"


def test_session_start_stop_and_stats() -> None:
    live = LiveState()
    stats = SessionStats()
    app = create_app(live, stats, cors_origins=["*"])
    client = TestClient(app)
    assert client.get("/session/stats").json()["phase"] == "idle"
    assert client.post("/session/start").json()["ok"] is True
    stats.tick(face_active=True, emotion="happy")
    time.sleep(0.04)
    stats.tick(face_active=True, emotion="happy")
    time.sleep(0.04)
    stats.tick(face_active=True, emotion="calm")
    summary = client.get("/session/stats").json()
    assert summary["phase"] == "running"
    assert "happy" in summary["emotion_pct"]
    assert "calm" in summary["emotion_pct"]
    stopped = client.post("/session/stop").json()["session"]
    assert stopped["phase"] == "stopped"
    assert stopped["recording"] is False
    assert "happy" in stopped["emotion_pct"]
