"""Thread-safe runtime metrics for observability (FPS, inference, camera, WebSockets)."""

from __future__ import annotations

import threading
import time
from collections import deque

_FPS_WINDOW_S = 1.0
_MAX_INFER_SAMPLES = 120
_RECENT_FRAME_S = 5.0


class RuntimeMetrics:
    """Written from the camera thread and API/WebSocket; read via :meth:`snapshot`."""

    __slots__ = (
        "_lock",
        "_frame_times",
        "_infer_ms",
        "_dropped_frames",
        "_camera_errors",
        "_ws_clients",
        "_model_ready",
        "_camera_ready",
        "_last_frame_mono",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame_times: deque[float] = deque()
        self._infer_ms: deque[float] = deque(maxlen=_MAX_INFER_SAMPLES)
        self._dropped_frames = 0
        self._camera_errors = 0
        self._ws_clients = 0
        self._model_ready = False
        self._camera_ready = False
        self._last_frame_mono: float | None = None

    def set_model_ready(self, ok: bool) -> None:
        with self._lock:
            self._model_ready = ok

    def set_camera_ready(self, ok: bool) -> None:
        with self._lock:
            self._camera_ready = ok

    def record_frame_processed(self) -> None:
        now = time.monotonic()
        with self._lock:
            self._frame_times.append(now)
            while self._frame_times and self._frame_times[0] < now - _FPS_WINDOW_S:
                self._frame_times.popleft()
            self._last_frame_mono = now

    def record_inference_ms(self, ms: float) -> None:
        with self._lock:
            self._infer_ms.append(max(0.0, float(ms)))

    def record_read_failure(self) -> None:
        """Failed ``cap.read()`` (no frame). Counts as both drop and camera error."""
        with self._lock:
            self._dropped_frames += 1
            self._camera_errors += 1

    def ws_connect(self) -> None:
        with self._lock:
            self._ws_clients += 1

    def ws_disconnect(self) -> None:
        with self._lock:
            self._ws_clients = max(0, self._ws_clients - 1)

    def snapshot(self) -> dict:
        with self._lock:
            now = time.monotonic()
            while self._frame_times and self._frame_times[0] < now - _FPS_WINDOW_S:
                self._frame_times.popleft()
            n_frames = len(self._frame_times)
            if n_frames >= 2:
                span = self._frame_times[-1] - self._frame_times[0]
                fps = (n_frames - 1) / span if span > 0 else float(n_frames)
            elif n_frames == 1:
                fps = 1.0 / _FPS_WINDOW_S
            else:
                fps = 0.0

            infer_list = list(self._infer_ms)
            if infer_list:
                infer_sorted = sorted(infer_list)
                n = len(infer_sorted)
                avg_ms = sum(infer_sorted) / n
                p95_idx = min(n - 1, max(0, int(0.95 * (n - 1))))
                p95_ms = infer_sorted[p95_idx]
            else:
                avg_ms = 0.0
                p95_ms = 0.0

            last = self._last_frame_mono
            recent = last is not None and (now - last) <= _RECENT_FRAME_S
            last_age = round(now - last, 2) if last is not None else None

            checks = {
                "model_loaded": self._model_ready,
                "camera_open": self._camera_ready,
                "recent_frames": recent,
            }
            reasons: list[str] = []
            if not self._model_ready:
                reasons.append("model not loaded")
            if not self._camera_ready:
                reasons.append("camera not open")
            if not recent:
                reasons.append("no frame processed in last 5s")
            ready = self._model_ready and self._camera_ready and recent

            return {
                "fps_estimate": round(fps, 1),
                "fps_window_s": _FPS_WINDOW_S,
                "inference_ms_avg": round(avg_ms, 2),
                "inference_ms_p95": round(p95_ms, 2),
                "inference_samples": len(infer_list),
                "dropped_frames": self._dropped_frames,
                "camera_errors": self._camera_errors,
                "websocket_clients": self._ws_clients,
                "model_ready": self._model_ready,
                "camera_ready": self._camera_ready,
                "last_frame_age_s": last_age,
                "pipeline_recent_frames": recent,
                "ready": ready,
                "ready_checks": checks,
                "ready_reasons": reasons,
            }
