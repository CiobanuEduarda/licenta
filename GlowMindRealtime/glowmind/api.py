"""Local FastAPI server: REST + WebSocket. Browser UI lives in ``dashboard.html`` (same package)."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from glowmind.session_stats import SessionStats
from glowmind.stream_state import LiveState
from glowmind.history_store import SessionHistoryStore

log = logging.getLogger(__name__)

_DASHBOARD_PATH = Path(__file__).resolve().parent / "dashboard.html"


def _load_dashboard_html() -> str:
    return _DASHBOARD_PATH.read_text(encoding="utf-8")


def create_app(
    live: LiveState,
    session_stats: SessionStats,
    *,
    cors_origins: list[str],
    history_store: SessionHistoryStore | None = None,
) -> FastAPI:
    app = FastAPI(title="GlowMind Realtime", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def dashboard() -> str:
        """Serve single-page UI from ``dashboard.html`` (same directory as this module)."""
        return _load_dashboard_html()

    @app.get("/state")
    def state() -> dict:
        return live.to_json()

    @app.get("/session/stats")
    def session_stats_get() -> dict:
        return session_stats.summary()

    @app.post("/session/start")
    def session_start(name: str = "") -> dict:
        session_stats.start_session(name=name)
        return {"ok": True, "session": session_stats.summary()}

    @app.post("/session/stop")
    def session_stop() -> dict:
        session_stats.stop_session()
        return {"ok": True, "session": session_stats.summary()}

    @app.get("/session/history")
    def session_history(limit: int = 20) -> dict:
        if history_store is None:
            return {"items": []}
        return {"items": history_store.list_sessions(limit=limit)}

    @app.get("/session/history/{session_id}")
    def session_history_one(session_id: int) -> dict:
        if history_store is None:
            raise HTTPException(status_code=404, detail="Session history is disabled")
        item = history_store.get_session(session_id)
        if item is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return item

    @app.delete("/session/history/{session_id}")
    def session_history_delete(session_id: int) -> dict:
        if history_store is None:
            raise HTTPException(status_code=404, detail="Session history is disabled")
        deleted = history_store.delete_session(session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"ok": True}

    @app.websocket("/ws")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                await ws.send_json(
                    {"live": live.to_json(), "session": session_stats.summary()}
                )
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            log.debug("WebSocket client disconnected")

    return app


def start_api_server_thread(
    host: str,
    port: int,
    app: FastAPI,
    *,
    log_level: str = "warning",
) -> None:
    """Run uvicorn in a daemon thread so the OpenCV loop can own the main thread."""

    import uvicorn

    def _run() -> None:
        try:
            uvicorn.run(app, host=host, port=port, log_level=log_level, access_log=False)
        except Exception as e:
            log.exception("API server crashed", exc_info=e)

    thread = threading.Thread(target=_run, name="glowmind-uvicorn", daemon=True)
    thread.start()
    log.info(
        "API server thread started at http://%s:%s (/ /state /ws /session/*)",
        host,
        port,
    )
