"""OpenCV preview helpers (camera + on-screen LED swatch)."""

from __future__ import annotations

from collections.abc import Sequence

import cv2

# Romance dashboard palette as BGR (hex: #3b1012, #a93c49, #eaa488, #feeec5, #dfdfdf).
_BGR_BURGUNDY = (18, 16, 59)
_BGR_ROSE = (73, 60, 169)
_BGR_PEACH = (136, 164, 234)
_BGR_CREAM = (197, 238, 254)
_BGR_GREY = (223, 223, 223)
_BGR_WHITE = (255, 255, 255)


def _blend_bgr(
    a: tuple[int, int, int], b: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    """Linear blend a → b as t goes 0 → 1."""
    return tuple(int(a[i] * (1.0 - t) + b[i] * t) for i in range(3))


def _wash_bgr(base: tuple[int, int, int], strength: float) -> tuple[int, int, int]:
    """Push ``base`` toward white for a soft panel tint (strength = amount of base)."""
    return _blend_bgr(_BGR_WHITE, base, strength)


def _va_to_pixel(
    x0: int,
    y0: int,
    inner: int,
    pad: int,
    valence: float,
    arousal: float,
) -> tuple[int, int]:
    """Map valence ∈ [-1,1] (left→right), arousal ∈ [-1,1] (down→up on screen)."""
    v = max(-1.0, min(1.0, valence))
    a = max(-1.0, min(1.0, arousal))
    px = int(x0 + pad + (v + 1.0) * 0.5 * inner)
    py = int(y0 + pad + (1.0 - a) * 0.5 * inner)
    return px, py


def draw_circumplex_mood_ring(
    frame,
    valence: float,
    arousal: float,
    trail: Sequence[tuple[float, float]],
    *,
    active: bool = True,
    panel_size: int = 228,
    margin_right: int = 14,
    margin_top: int = 10,
) -> None:
    """Draw a valence × arousal circumplex (Russell-style) with a dot and optional trail.

    Valence increases to the right; arousal increases upward. Uses the same [-1, 1] space
    as ``va_to_emotion`` after calibration (``v_display`` / ``a_display`` in the main loop).
    """
    h, w = frame.shape[:2]
    pad = 14
    inner = panel_size - 2 * pad
    x0 = max(0, w - margin_right - panel_size)
    y0 = margin_top
    x1 = x0 + panel_size
    y1 = y0 + panel_size
    if y1 > h:
        y0 = max(0, h - panel_size - margin_top)
        y1 = y0 + panel_size

    cx = x0 + pad + inner // 2
    cy = y0 + pad + inner // 2

    overlay = frame.copy()

    # Quadrant tints — soft washes of rose / peach / grey / burgundy (matches web dashboard).
    q_wash = {
        "nw": _wash_bgr(_BGR_ROSE, 0.14),
        "ne": _wash_bgr(_BGR_PEACH, 0.14),
        "sw": _wash_bgr(_BGR_BURGUNDY, 0.07),
        "se": _wash_bgr(_BGR_GREY, 0.12),
    }
    cv2.rectangle(overlay, (x0 + pad, y0 + pad), (cx, cy), q_wash["nw"], -1)
    cv2.rectangle(overlay, (cx, y0 + pad), (x1 - pad, cy), q_wash["ne"], -1)
    cv2.rectangle(overlay, (x0 + pad, cy), (cx, y1 - pad), q_wash["sw"], -1)
    cv2.rectangle(overlay, (cx, cy), (x1 - pad, y1 - pad), q_wash["se"], -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.rectangle(frame, (x0, y0), (x1, y1), _BGR_ROSE, 2)
    axis_color = _blend_bgr(_BGR_GREY, _BGR_BURGUNDY, 0.35)
    cv2.line(frame, (x0 + pad, cy), (x1 - pad, cy), axis_color, 1)
    cv2.line(frame, (cx, y0 + pad), (cx, y1 - pad), axis_color, 1)
    cv2.circle(frame, (cx, cy), 3, _BGR_ROSE, -1)

    label_color = _BGR_BURGUNDY if active else _blend_bgr(_BGR_GREY, _BGR_BURGUNDY, 0.55)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs = 0.38
    t = 1
    cv2.putText(frame, "VA circumplex", (x0 + 6, y0 + 16), font, 0.45, label_color, 1)
    cv2.putText(frame, "V+", (x1 - pad - 18, cy + 4), font, fs, label_color, t)
    cv2.putText(frame, "V-", (x0 + pad + 2, cy + 4), font, fs, label_color, t)
    cv2.putText(frame, "A+", (cx - 8, y0 + pad - 2), font, fs, label_color, t)
    cv2.putText(frame, "A-", (cx - 8, y1 - pad + 10), font, fs, label_color, t)

    if len(trail) >= 2:
        pts = [
            _va_to_pixel(x0, y0, inner, pad, tv, ta) for tv, ta in trail
        ]
        for i in range(1, len(pts)):
            s = i / max(1, len(pts) - 1)
            col = _blend_bgr(_BGR_PEACH, _BGR_ROSE, s * 0.85)
            cv2.line(frame, pts[i - 1], pts[i], col, 2, cv2.LINE_AA)

    if active:
        px, py = _va_to_pixel(x0, y0, inner, pad, valence, arousal)
        cv2.circle(frame, (px, py), 8, _BGR_CREAM, 2, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5, _BGR_ROSE, -1, cv2.LINE_AA)
        cv2.circle(frame, (px, py), 5, _BGR_BURGUNDY, 1, cv2.LINE_AA)
    else:
        px, py = _va_to_pixel(x0, y0, inner, pad, valence, arousal)
        cv2.circle(frame, (px, py), 6, _BGR_GREY, 1, cv2.LINE_AA)


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
    cv2.rectangle(frame, (x, y), (x + w, y + h), _BGR_ROSE, 2)
    cv2.putText(
        frame,
        f"{emotion}  V:{v_s:.2f} A:{a_s:.2f}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        _BGR_BURGUNDY,
        2,
    )


def draw_led_preview(frame, r: int, g: int, b: int) -> None:
    r, g, b = max(1, r), max(1, g), max(1, b)
    scale = 255 / max(r, g, b)
    preview_rgb = (int(r * scale), int(g * scale), int(b * scale))
    cv2.rectangle(frame, (10, 10), (70, 50), _BGR_BURGUNDY, 2)
    cv2.rectangle(
        frame, (12, 12), (68, 48), (preview_rgb[2], preview_rgb[1], preview_rgb[0]), -1
    )
    cv2.putText(frame, "LED", (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, _BGR_CREAM, 1)
