"""CLI entry: logging setup and error handling."""

from __future__ import annotations

import logging
import os
import sys

from glowmind.api import create_app, start_api_server_thread
from glowmind.config import Settings
from glowmind.runner import CameraUnavailableError, run
from glowmind.stream_state import LiveState

log = logging.getLogger(__name__)


def main() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    try:
        settings = Settings.from_env()
    except ValueError as e:
        log.error("Configuration error: %s", e)
        sys.exit(2)

    live: LiveState | None = None
    if settings.api_enabled:
        live = LiveState()
        app = create_app(live, cors_origins=settings.cors_origin_list())
        start_api_server_thread(settings.api_host, settings.api_port, app)

    try:
        run(settings, live_state=live)
    except FileNotFoundError as e:
        log.error("%s", e)
        sys.exit(1)
    except CameraUnavailableError as e:
        log.error("%s", e)
        sys.exit(1)

