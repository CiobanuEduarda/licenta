"""Face backend abstraction: Haar (OpenCV) or MediaPipe Face Mesh.

This module keeps MediaPipe optional. If not installed, the backend can fall
back to Haar without breaking the rest of the pipeline.

MediaPipe >= 0.10.14 (approx.) ships only the Tasks API (`FaceLandmarker`), not
`mp.solutions`. We use Tasks when `solutions` is missing and download the
official ``face_landmarker.task`` to the user cache on first use (overridable
via GLOWMIND_FACE_LANDMARKER_MODEL).
"""

from __future__ import annotations

import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Float16 bundle used by MediaPipe Face Landmarker (Tasks).
_FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _face_landmarker_cache_path() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "glowmind" / "face_landmarker.task"


def _resolve_face_landmarker_model_path() -> Path:
    for key in ("GLOWMIND_FACE_LANDMARKER_MODEL", "MEDIAPIPE_FACE_LANDMARKER_MODEL"):
        raw = os.environ.get(key)
        if raw:
            p = Path(raw).expanduser()
            if p.is_file():
                return p.resolve()
            raise FileNotFoundError(
                f"{key} is set to {raw!r} but that file does not exist."
            )
    cache = _face_landmarker_cache_path()
    if cache.is_file():
        return cache.resolve()
    return _download_face_landmarker_task(cache)


def _download_face_landmarker_task(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    log.info("Downloading Face Landmarker model to %s", dest)
    req = urllib.request.Request(
        _FACE_LANDMARKER_MODEL_URL,
        headers={"User-Agent": "GlowMindRealtime/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            with open(tmp, "wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
        tmp.replace(dest)
    except Exception as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise RuntimeError(
            "Could not download the MediaPipe Face Landmarker model bundle.\n"
            f"  URL: {_FACE_LANDMARKER_MODEL_URL}\n"
            "Save the file locally and point GLOWMIND_FACE_LANDMARKER_MODEL to it."
        ) from e
    return dest.resolve()


@dataclass(frozen=True)
class FaceDetection:
    """One face detection result for the runtime loop."""

    x: int
    y: int
    w: int
    h: int
    # Pixel coordinates (x, y). Only populated for MediaPipe Face Mesh.
    landmarks_xy: list[tuple[int, int]] | None = None


class FaceBackend:
    def detect(self, frame_bgr) -> FaceDetection | None:  # pragma: no cover
        raise NotImplementedError


class HaarFaceBackend(FaceBackend):
    def __init__(self, cascade: cv2.CascadeClassifier) -> None:
        self._cascade = cascade

    def detect(self, frame_bgr) -> FaceDetection | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        # Choose primary face: largest area, center tie-break (same as inference.select_primary_face)
        h_frame, w_frame = frame_bgr.shape[:2]
        fw = float(w_frame)
        fh = float(h_frame)
        cx0 = fw / 2.0
        cy0 = fh / 2.0

        def sort_key(row) -> tuple[float, float]:
            x, y, w, h = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            area = w * h
            fcx = x + w / 2.0
            fcy = y + h / 2.0
            dist_sq = (fcx - cx0) ** 2 + (fcy - cy0) ** 2
            return (area, -dist_sq)

        best = max(faces, key=sort_key)
        x, y, w, h = int(best[0]), int(best[1]), int(best[2]), int(best[3])
        return FaceDetection(x=x, y=y, w=w, h=h, landmarks_xy=None)


class MediaPipeFaceMeshBackend(FaceBackend):
    def __init__(self, *, max_num_faces: int = 1) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "MediaPipe is not installed. Install with:\n"
                "  pip install -e '.[vision]'\n"
                "or:\n"
                "  pip install mediapipe\n"
            ) from e

        self._legacy_mesh = None
        self._landmarker = None
        self._last_ts_ms = 0

        if hasattr(mp, "solutions"):
            self._legacy_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=max_num_faces,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            from mediapipe import Image as MPImage  # type: ignore
            from mediapipe import ImageFormat as MPImageFormat  # type: ignore
            from mediapipe.tasks.python import BaseOptions  # type: ignore
            from mediapipe.tasks.python.vision import (  # type: ignore
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )

            self._MPImage = MPImage
            self._MPImageFormat = MPImageFormat
            model_path = _resolve_face_landmarker_model_path()
            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.VIDEO,
                num_faces=max_num_faces,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self._landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr) -> FaceDetection | None:
        if self._legacy_mesh is not None:
            return self._detect_legacy(frame_bgr)
        return self._detect_tasks(frame_bgr)

    def _detect_legacy(self, frame_bgr) -> FaceDetection | None:
        h_frame, w_frame = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self._legacy_mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        face = res.multi_face_landmarks[0]
        xs: list[int] = []
        ys: list[int] = []
        landmarks_xy: list[tuple[int, int]] = []
        for lm in face.landmark:
            x = int(lm.x * w_frame)
            y = int(lm.y * h_frame)
            xs.append(x)
            ys.append(y)
            landmarks_xy.append((x, y))

        x0 = max(0, min(xs))
        y0 = max(0, min(ys))
        x1 = min(w_frame, max(xs))
        y1 = min(h_frame, max(ys))
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        return FaceDetection(x=x0, y=y0, w=w, h=h, landmarks_xy=landmarks_xy)

    def _detect_tasks(self, frame_bgr) -> FaceDetection | None:
        assert self._landmarker is not None
        h_frame, w_frame = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = self._MPImage(image_format=self._MPImageFormat.SRGB, data=rgb)

        now = int(time.monotonic_ns() // 1_000_000)
        if now <= self._last_ts_ms:
            now = self._last_ts_ms + 1
        self._last_ts_ms = now

        res = self._landmarker.detect_for_video(mp_image, now)
        if not res.face_landmarks:
            return None

        face = res.face_landmarks[0]
        xs: list[int] = []
        ys: list[int] = []
        landmarks_xy: list[tuple[int, int]] = []
        for lm in face:
            x = int((lm.x or 0.0) * w_frame)
            y = int((lm.y or 0.0) * h_frame)
            xs.append(x)
            ys.append(y)
            landmarks_xy.append((x, y))

        x0 = max(0, min(xs))
        y0 = max(0, min(ys))
        x1 = min(w_frame, max(xs))
        y1 = min(h_frame, max(ys))
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        return FaceDetection(x=x0, y=y0, w=w, h=h, landmarks_xy=landmarks_xy)


def build_face_backend(backend: str, cascade: cv2.CascadeClassifier | None) -> FaceBackend:
    b = (backend or "").strip().lower()
    if b in ("haar", "opencv", ""):
        if cascade is None:
            raise ValueError("Haar backend requires a loaded CascadeClassifier")
        return HaarFaceBackend(cascade)
    if b in ("mediapipe", "mp", "face_mesh"):
        return MediaPipeFaceMeshBackend()
    raise ValueError(f"Unknown FACE_BACKEND={backend!r} (expected 'haar' or 'mediapipe')")

