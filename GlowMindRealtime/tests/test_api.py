"""FastAPI REST and WebSocket surface (in-memory, no camera)."""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from glowmind.api import create_app
from glowmind.history_store import SessionHistoryStore
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
    started = client.post("/session/start", params={"name": "Morning Run"}).json()
    assert started["ok"] is True
    assert started["session"]["name"] == "Morning Run"
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


def test_session_history_endpoints(tmp_path) -> None:
    live = LiveState()
    store = SessionHistoryStore(str(tmp_path / "history.sqlite3"))
    stats = SessionStats(on_stop=store.save_stopped_session)
    app = create_app(live, stats, cors_origins=["*"], history_store=store)
    client = TestClient(app)

    assert client.get("/session/history").json() == {"items": []}
    client.post("/session/start", params={"name": "Demo Session"})
    stats.tick(face_active=True, emotion="happy")
    time.sleep(0.03)
    stats.tick(face_active=True, emotion="happy")
    stopped = client.post("/session/stop").json()["session"]
    assert stopped["phase"] == "stopped"

    items = client.get("/session/history").json()["items"]
    assert len(items) == 1
    assert items[0]["name"] == "Demo Session"
    session_id = items[0]["id"]
    one = client.get(f"/session/history/{session_id}")
    assert one.status_code == 200
    body = one.json()
    assert body["id"] == session_id
    assert body["name"] == "Demo Session"
    assert "timeline" in body
    assert body["phase"] == "stopped"


def test_session_history_delete_endpoint(tmp_path) -> None:
    live = LiveState()
    store = SessionHistoryStore(str(tmp_path / "history.sqlite3"))
    stats = SessionStats(on_stop=store.save_stopped_session)
    app = create_app(live, stats, cors_origins=["*"], history_store=store)
    client = TestClient(app)

    client.post("/session/start", params={"name": "To Delete"})
    stats.tick(face_active=True, emotion="happy")
    time.sleep(0.03)
    stats.tick(face_active=True, emotion="happy")
    client.post("/session/stop")
    items = client.get("/session/history").json()["items"]
    assert len(items) == 1
    session_id = items[0]["id"]

    deleted = client.delete(f"/session/history/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"ok": True}
    assert client.get("/session/history").json()["items"] == []

    missing = client.delete(f"/session/history/{session_id}")
    assert missing.status_code == 404
