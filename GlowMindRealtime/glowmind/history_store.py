"""SQLite-backed persistence for completed session summaries."""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class SessionHistoryStore:
    """Persist stopped sessions so users can inspect past runs."""

    def __init__(self, db_path: str) -> None:
        self._db_path = str(Path(db_path))
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phase TEXT NOT NULL,
                    name TEXT NOT NULL DEFAULT '',
                    elapsed_s REAL NOT NULL,
                    emotion_pct_json TEXT NOT NULL,
                    timeline_json TEXT NOT NULL,
                    note TEXT NOT NULL,
                    stopped_at_utc TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )
                """
            )
            cols = {
                r["name"]
                for r in conn.execute("PRAGMA table_info(session_history)").fetchall()
            }
            if "name" not in cols:
                conn.execute(
                    "ALTER TABLE session_history ADD COLUMN name TEXT NOT NULL DEFAULT ''"
                )
            conn.commit()

    def save_stopped_session(self, summary: dict[str, Any]) -> int:
        if summary.get("phase") != "stopped":
            raise ValueError("Only stopped sessions can be persisted")
        emotion_pct = summary.get("emotion_pct") or {}
        timeline = summary.get("timeline") or []
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    """
                    INSERT INTO session_history (phase, name, elapsed_s, emotion_pct_json, timeline_json, note)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "stopped",
                        str(summary.get("name", "")).strip(),
                        float(summary.get("elapsed_s", 0.0)),
                        json.dumps(emotion_pct, separators=(",", ":")),
                        json.dumps(timeline, separators=(",", ":")),
                        str(summary.get("note", "")),
                    ),
                )
                conn.commit()
                return int(cur.lastrowid)

    def list_sessions(self, *, limit: int = 20) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 200))
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, phase, name, elapsed_s, emotion_pct_json, note, stopped_at_utc
                    FROM session_history
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            emotion_pct = json.loads(row["emotion_pct_json"])
            out.append(
                {
                    "id": int(row["id"]),
                    "phase": row["phase"],
                    "name": row["name"],
                    "elapsed_s": round(float(row["elapsed_s"]), 2),
                    "emotion_pct": emotion_pct,
                    "note": row["note"],
                    "stopped_at_utc": row["stopped_at_utc"],
                }
            )
        return out

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT id, phase, name, elapsed_s, emotion_pct_json, timeline_json, note, stopped_at_utc
                    FROM session_history
                    WHERE id = ?
                    """,
                    (int(session_id),),
                ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row["id"]),
            "phase": row["phase"],
            "name": row["name"],
            "elapsed_s": round(float(row["elapsed_s"]), 2),
            "emotion_pct": json.loads(row["emotion_pct_json"]),
            "timeline": json.loads(row["timeline_json"]),
            "note": row["note"],
            "stopped_at_utc": row["stopped_at_utc"],
        }

    def delete_session(self, session_id: int) -> bool:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "DELETE FROM session_history WHERE id = ?",
                    (int(session_id),),
                )
                conn.commit()
                return cur.rowcount > 0
