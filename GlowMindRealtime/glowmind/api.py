"""Local FastAPI server: REST state + WebSocket stream (runs in a background thread)."""

from __future__ import annotations

import asyncio
import logging
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from glowmind.stream_state import LiveState

log = logging.getLogger(__name__)

# Single-page dashboard: uses WebSocket /ws so values update without refreshing the browser.
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GlowMind — live</title>
  <style>
    :root { color-scheme: dark; --bg: #12141a; --card: #1c1f28; --muted: #8b909e; --accent: #5eb8ff; }
    * { box-sizing: border-box; }
    body {
      margin: 0; min-height: 100vh; font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg); color: #e8eaef; display: flex; flex-direction: column; align-items: center;
      justify-content: center; padding: 1.5rem;
    }
    h1 { font-size: 0.85rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
         color: var(--muted); margin: 0 0 1.25rem; }
    .card {
      background: var(--card); border-radius: 16px; padding: 1.75rem 2rem; min-width: min(100%, 320px);
      box-shadow: 0 12px 40px rgba(0,0,0,.35); border: 1px solid rgba(255,255,255,.06);
    }
    .emotion { font-size: 2.25rem; font-weight: 700; line-height: 1.2; margin-bottom: 0.5rem; }
    .badge {
      display: inline-block; font-size: 0.7rem; padding: 0.2rem 0.5rem; border-radius: 6px;
      background: rgba(94,184,255,.15); color: var(--accent); margin-bottom: 1rem;
    }
    .row { display: flex; justify-content: space-between; gap: 1rem; font-size: 0.9rem;
           padding: 0.35rem 0; border-bottom: 1px solid rgba(255,255,255,.06); }
    .row:last-child { border-bottom: none; }
    .label { color: var(--muted); }
    .swatch {
      width: 100%; height: 56px; border-radius: 10px; margin-top: 1rem;
      border: 1px solid rgba(255,255,255,.12);
    }
    .status { margin-top: 1rem; font-size: 0.75rem; color: var(--muted); text-align: center; }
    .err { color: #f87171; }
  </style>
</head>
<body>
  <h1>GlowMind realtime</h1>
  <div class="card">
    <div class="badge" id="face">…</div>
    <div class="emotion" id="emotion">—</div>
    <div class="row"><span class="label">Valence (display)</span><span id="vd">—</span></div>
    <div class="row"><span class="label">Arousal (display)</span><span id="ad">—</span></div>
    <div class="row"><span class="label">Valence (smooth)</span><span id="vs">—</span></div>
    <div class="row"><span class="label">Arousal (smooth)</span><span id="as">—</span></div>
    <div class="row"><span class="label">LED RGB</span><span id="rgb">—</span></div>
    <div class="swatch" id="swatch"></div>
    <p class="status" id="conn">Connecting…</p>
  </div>
  <script>
    const $ = (id) => document.getElementById(id);
    const fmt = (n) => (typeof n === "number" && !Number.isNaN(n)) ? n.toFixed(2) : "—";
    function apply(data) {
      $("emotion").textContent = data.emotion ?? "—";
      $("face").textContent = data.face_active ? "Face detected" : "No face";
      $("vd").textContent = fmt(data.valence_display);
      $("ad").textContent = fmt(data.arousal_display);
      $("vs").textContent = fmt(data.valence_smoothed);
      $("as").textContent = fmt(data.arousal_smoothed);
      const r = data.led_r|0, g = data.led_g|0, b = data.led_b|0;
      $("rgb").textContent = r + ", " + g + ", " + b;
      const mx = Math.max(r,g,b,1);
      const sr = Math.round(r/mx*255), sg = Math.round(g/mx*255), sb = Math.round(b/mx*255);
      $("swatch").style.background = "rgb(" + sr + "," + sg + "," + sb + ")";
    }
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = proto + "//" + location.host + "/ws";
    const ws = new WebSocket(wsUrl);
    ws.onopen = () => { $("conn").textContent = "Live (WebSocket)"; $("conn").classList.remove("err"); };
    ws.onmessage = (ev) => { try { apply(JSON.parse(ev.data)); } catch (e) {} };
    ws.onclose = () => {
      $("conn").textContent = "Disconnected — refresh to reconnect";
      $("conn").classList.add("err");
    };
    ws.onerror = () => {
      $("conn").textContent = "WebSocket error";
      $("conn").classList.add("err");
    };
  </script>
</body>
</html>
"""


def create_app(live: LiveState, *, cors_origins: list[str]) -> FastAPI:
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
        """Browser UI that streams updates via WebSocket (no manual refresh)."""
        return _DASHBOARD_HTML

    @app.get("/state")
    def state() -> dict:
        return live.to_json()

    @app.websocket("/ws")
    async def stream(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                await ws.send_json(live.to_json())
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
        "API server thread started at http://%s:%s (live UI /, REST /state, WS /ws)",
        host,
        port,
    )
