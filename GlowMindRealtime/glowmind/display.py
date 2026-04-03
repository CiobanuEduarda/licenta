"""OpenCV preview helpers (camera + on-screen LED swatch)."""

from __future__ import annotations

import cv2


def open_capture(primary_index: int, fallback_index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(primary_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(fallback_index)
    return cap


def draw_face_overlay(
    frame,
    x: int,
    y: int,
    w: int,
    h: int,
    emotion: str,
    v_s: float,
    a_s: float,
) -> None:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"{emotion}  V:{v_s:.2f} A:{a_s:.2f}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )


def draw_led_preview(frame, r: int, g: int, b: int) -> None:
    r, g, b = max(1, r), max(1, g), max(1, b)
    scale = 255 / max(r, g, b)
    preview_rgb = (int(r * scale), int(g * scale), int(b * scale))
    cv2.rectangle(frame, (10, 10), (70, 50), (0, 0, 0), 2)
    cv2.rectangle(
        frame, (12, 12), (68, 48), (preview_rgb[2], preview_rgb[1], preview_rgb[0]), -1
    )
    cv2.putText(frame, "LED", (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
