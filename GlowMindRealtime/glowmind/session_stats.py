"""Session-length aggregates: discrete emotion time shares + sampled timeline for charts."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Callable

# One sample every ~0.5s wall time while recording (keeps payloads moderate).
_SAMPLE_INTERVAL_S = 0.5
_MAX_TIMELINE_POINTS = 4000


class SessionStats:
    """Written from the camera thread; read from FastAPI. Percentages are model-facing, not clinical."""

    __slots__ = (
        "_lock",
        "_started_at",
        "_last_mono",
        "_emotion_s",
        "_running",
        "_phase",
        "_samples",
        "_last_sample_mono",
        "_frozen_elapsed",
        "_on_stop",
    )

    def __init__(self, on_stop: Callable[[dict], None] | None = None) -> None:
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._last_mono: float | None = None
        self._emotion_s: dict[str, float] = defaultdict(float)
        self._running = False
        self._phase: str = "idle"
        self._samples: list[dict] = []
        self._last_sample_mono: float | None = None
        self._frozen_elapsed: float | None = None
        self._on_stop = on_stop

    def start_session(self) -> None:
        """Clear state and begin recording (wall-time buckets + timeline samples)."""
        with self._lock:
            self._started_at = time.monotonic()
            self._last_mono = None
            self._last_sample_mono = None
            self._emotion_s.clear()
            self._samples.clear()
            self._running = True
            self._phase = "running"
            self._frozen_elapsed = None

    def stop_session(self) -> None:
        """Freeze totals and timeline; no further accumulation until the next start."""
        payload: dict | None = None
        with self._lock:
            if not self._running or self._started_at is None:
                return
            self._running = False
            self._phase = "stopped"
            self._frozen_elapsed = max(0.0, time.monotonic() - self._started_at)
            payload = self._summary_unlocked()
        if payload is not None and self._on_stop is not None:
            self._on_stop(payload)

    def tick(self, *, face_active: bool, emotion: str) -> None:
        """Accumulate emotion time and occasional timeline points while recording."""
        now = time.monotonic()
        with self._lock:
            if not self._running or self._started_at is None:
                return

            if self._last_mono is None:
                self._last_mono = now
                self._append_sample(now, face_active, emotion)
                return

            dt = max(0.0, now - self._last_mono)
            self._last_mono = now
            if dt <= 0.0:
                return

            if face_active:
                self._emotion_s[emotion] += dt
            else:
                self._emotion_s["no_face"] += dt

            self._append_sample(now, face_active, emotion)

    def _append_sample(self, now: float, face_active: bool, emotion: str) -> None:
        if self._last_sample_mono is None or now - self._last_sample_mono >= _SAMPLE_INTERVAL_S:
            label = "no_face" if not face_active else emotion
            started = self._started_at
            assert started is not None
            t = round(now - started, 2)
            self._samples.append({"t": t, "emotion": label})
            if len(self._samples) > _MAX_TIMELINE_POINTS:
                self._samples.pop(0)
            self._last_sample_mono = now

    def summary(self) -> dict:
        """phase: idle | running | stopped — timeline + emotion_pct for charts."""
        with self._lock:
            return self._summary_unlocked()

    def _summary_unlocked(self) -> dict:
        if self._phase == "idle":
            return {
                "phase": "idle",
                "recording": False,
                "elapsed_s": 0.0,
                "emotion_pct": {},
                "timeline": [],
                "note": "Click “Start session” to record, then “Stop” to freeze results and charts.",
            }

        if self._phase == "running":
            assert self._started_at is not None
            elapsed = max(0.0, time.monotonic() - self._started_at)
            te = sum(self._emotion_s.values()) or 1e-12
            emotion_pct = {k: round(100.0 * v / te, 2) for k, v in sorted(self._emotion_s.items())}
            return {
                "phase": "running",
                "recording": True,
                "elapsed_s": round(elapsed, 2),
                "emotion_pct": emotion_pct,
                "timeline": list(self._samples),
                "note": "Recording… Stop the session to freeze the summary and graphs.",
            }

        # stopped
        elapsed = self._frozen_elapsed if self._frozen_elapsed is not None else 0.0
        te = sum(self._emotion_s.values()) or 1e-12
        emotion_pct = {k: round(100.0 * v / te, 2) for k, v in sorted(self._emotion_s.items())}
        return {
            "phase": "stopped",
            "recording": False,
            "elapsed_s": round(elapsed, 2),
            "emotion_pct": emotion_pct,
            "timeline": list(self._samples),
            "note": "Session ended",
        }
