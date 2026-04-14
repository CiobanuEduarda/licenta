"""Main realtime loop: camera → VA model → LED + preview."""

from __future__ import annotations

import logging
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
from glowmind.hardware import LedSink, open_led_sink
from glowmind.inference import (
    build_va_resnet,
    expand_face_bbox,
    face_transform,
    load_face_cascade,
    load_model_weights,
    select_primary_face,
)
from glowmind.runtime_metrics import RuntimeMetrics
from glowmind.session_stats import SessionStats
from glowmind.stream_state import LiveState

log = logging.getLogger(__name__)


class CameraUnavailableError(RuntimeError):
    """Raised when no camera device could be opened."""


def _forward_va(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    out = model(batch)
    return torch.clamp(out, -1.0, 1.0)


def run(
    settings: Settings,
    *,
    led: LedSink | None = None,
    live_state: LiveState | None = None,
    session_stats: SessionStats | None = None,
    metrics: RuntimeMetrics | None = None,
) -> None:
    """Run the realtime pipeline. Always closes the active sink when finished (including on error).

    Pass a custom :class:`LedSink` (e.g. :class:`NullLedSink`) to skip serial or for tests.
    Optional :class:`LiveState` is updated every frame for REST/WebSocket clients.
    Optional :class:`SessionStats` accumulates segment timers when a session is started via the API.
    Optional :class:`RuntimeMetrics` records FPS, inference latency, and camera read failures.
    """
    sink: LedSink = open_led_sink(settings) if led is None else led
    try:
        _run_loop(settings, sink, live_state, session_stats, metrics)
    finally:
        sink.close()


def _run_loop(
    settings: Settings,
    led: LedSink,
    live_state: LiveState | None,
    session_stats: SessionStats | None,
    metrics: RuntimeMetrics | None,
) -> None:
    device = settings.device
    model = build_va_resnet(device)
    load_model_weights(model, settings.model_weights, device)
    log.info("Model loaded successfully")
    if metrics is not None:
        metrics.set_model_ready(True)

    face_cascade = load_face_cascade()
    transform = face_transform()

    cap = open_capture(settings.camera_index, settings.camera_fallback_index)
    if not cap.isOpened():
        if metrics is not None:
            metrics.set_camera_ready(False)
        raise CameraUnavailableError(
            "Could not open camera. Check permissions "
            "(System Settings → Privacy → Camera) or try CAMERA_INDEX / CAMERA_FALLBACK_INDEX."
        )
    log.info("Camera opened (index %s)", settings.camera_index)
    if metrics is not None:
        metrics.set_camera_ready(True)

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
                if metrics is not None:
                    metrics.record_read_failure()
                break

            t = time.time() - start_time
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            has_face = len(faces) > 0

            if has_face:
                h_frame, w_frame = frame.shape[:2]
                x, y, w, h = select_primary_face(faces, w_frame, h_frame)
                px, py, pw, ph = expand_face_bbox(
                    x, y, w, h, w_frame, h_frame, settings.face_bbox_buffer
                )
                face = frame[py : py + ph, px : px + pw]
                inp = transform(face).unsqueeze(0).to(device)

                t_infer0 = time.perf_counter()
                with torch.no_grad():
                    output = _forward_va(model, inp)
                if metrics is not None:
                    metrics.record_inference_ms((time.perf_counter() - t_infer0) * 1000.0)

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
                led.send_rgb(r, g, b)

                draw_face_overlay(frame, x, y, w, h, emotion, v_s, a_s)
                last_v_plot, last_a_plot = v_display, a_display
                va_trail.append((v_display, a_display))
                if live_state is not None:
                    live_state.update(
                        face_active=True,
                        emotion=emotion,
                        valence_smoothed=v_s,
                        arousal_smoothed=a_s,
                        valence_display=v_display,
                        arousal_display=a_display,
                        led_r=r,
                        led_g=g,
                        led_b=b,
                    )
            else:
                target_rgb = EMOTIONS["neutral"]
                current_rgb = lerp_rgb(current_rgb, target_rgb, 0.02)
                r, g, b = pulse_neutral_idle(current_rgb, t)
                r, g, b = scale_for_led(r, g, b, settings.led_brightness)
                led.send_rgb(r, g, b)
                if live_state is not None:
                    live_state.update(
                        face_active=False,
                        emotion="neutral",
                        valence_smoothed=v_s,
                        arousal_smoothed=a_s,
                        valence_display=last_v_plot,
                        arousal_display=last_a_plot,
                        led_r=r,
                        led_g=g,
                        led_b=b,
                    )

            draw_led_preview(frame, r, g, b)
            draw_circumplex_mood_ring(
                frame,
                last_v_plot,
                last_a_plot,
                va_trail,
                active=has_face,
            )

            if session_stats is not None:
                if has_face:
                    session_stats.tick(face_active=True, emotion=emotion)
                else:
                    session_stats.tick(face_active=False, emotion="neutral")

            if metrics is not None:
                metrics.record_frame_processed()

            cv2.imshow("GlowMind", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
