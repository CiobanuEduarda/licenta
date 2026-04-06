"""Realtime valence/arousal from webcam → emotion-colored LED (optional serial)."""

from __future__ import annotations

import time
from collections import deque

import cv2
import torch

from glowmind.config import Settings
from glowmind.display import (
    draw_circumplex_mood_ring,
    draw_face_overlay,
    draw_led_preview,
    open_capture,
)
from glowmind.emotion import (
    EMOTIONS,
    ema,
    lerp_rgb,
    pulse_for_emotion,
    pulse_neutral_idle,
    scale_for_led,
    va_to_emotion,
)
from glowmind.hardware import open_serial, send_rgb
from glowmind.inference import (
    build_va_resnet,
    face_transform,
    load_face_cascade,
    load_model_weights,
    select_primary_face,
)


def _expand_face_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    frame_width: int,
    frame_height: int,
    buffer: float,
) -> tuple[int, int, int, int]:
    """Pad Haar box like AffectNet training (margin around face)."""
    x_min = max(0, int(x - w * buffer))
    y_min = max(0, int(y - h * buffer))
    x_max = min(frame_width, int(x + w * (1.0 + buffer)))
    y_max = min(frame_height, int(y + h * (1.0 + buffer)))
    out_w = max(1, x_max - x_min)
    out_h = max(1, y_max - y_min)
    return x_min, y_min, out_w, out_h


def _forward_va(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    out = model(batch)
    return torch.clamp(out, -1.0, 1.0)


def run(settings: Settings) -> None:
    device = settings.device
    model = build_va_resnet(device)
    load_model_weights(model, settings.model_weights, device)
    print("Model loaded successfully")

    face_cascade = load_face_cascade()
    transform = face_transform()
    ser = open_serial(settings.serial_port, settings.serial_baud)

    cap = open_capture(settings.camera_index, settings.camera_fallback_index)
    if not cap.isOpened():
        print(
            "Could not open camera. Check permissions "
            "(System Settings → Privacy → Camera) or try another app."
        )
        if ser:
            ser.close()
        raise SystemExit(1)
    print("Camera opened")

    v_s = 0.0
    a_s = 0.0
    va_trail: deque[tuple[float, float]] = deque(maxlen=48)
    last_v_plot = 0.0
    last_a_plot = 0.0
    current_rgb = EMOTIONS["neutral"]
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = time.time() - start_time
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                h_frame, w_frame = frame.shape[:2]
                x, y, w, h = select_primary_face(faces, w_frame, h_frame)
                px, py, pw, ph = _expand_face_bbox(
                    x, y, w, h, w_frame, h_frame, settings.face_bbox_buffer
                )
                face = frame[py : py + ph, px : px + pw]
                inp = transform(face).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = _forward_va(model, inp)

                valence = float(output[0, 0].item())
                arousal = float(output[0, 1].item())

                v_s = ema(v_s, valence, settings.ema_alpha)
                a_s = ema(a_s, arousal, settings.ema_alpha)

                v_display = max(
                    -1.0,
                    min(1.0, (v_s + settings.valence_offset) * settings.valence_gain),
                )
                a_display = max(
                    -1.0,
                    min(1.0, (a_s + settings.arousal_offset) * settings.arousal_gain),
                )

                emotion = va_to_emotion(v_display, a_display)
                target_rgb = EMOTIONS[emotion]
                blend = min(1.0, settings.transition_speed * settings.frame_blend)
                current_rgb = lerp_rgb(current_rgb, target_rgb, blend)

                r, g, b = pulse_for_emotion(current_rgb, t, emotion)
                r, g, b = scale_for_led(r, g, b, settings.led_brightness)
                send_rgb(ser, r, g, b)

                draw_face_overlay(frame, x, y, w, h, emotion, v_s, a_s)
                last_v_plot, last_a_plot = v_display, a_display
                va_trail.append((v_display, a_display))
            else:
                target_rgb = EMOTIONS["neutral"]
                current_rgb = lerp_rgb(current_rgb, target_rgb, 0.02)
                r, g, b = pulse_neutral_idle(current_rgb, t)
                r, g, b = scale_for_led(r, g, b, settings.led_brightness)
                send_rgb(ser, r, g, b)

            draw_led_preview(frame, r, g, b)
            draw_circumplex_mood_ring(
                frame,
                last_v_plot,
                last_a_plot,
                va_trail,
                active=len(faces) > 0,
            )

            cv2.imshow("GlowMind", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        if ser:
            ser.close()
        cv2.destroyAllWindows()


def main() -> None:
    run(Settings.from_env())


if __name__ == "__main__":
    main()
