# filename: src/utils/logger.py
# purpose:  Centralised logging configuration for PatrolIQ

import logging
import os
from pathlib import Path


def setup_logging() -> None:
    """
    Configure root logger with console + optional file handler.
    Call once from entry points (notebooks, scripts, API startup).
    Level controlled via LOG_LEVEL env var (default: INFO).
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    log_dir = Path("logs")
    try:
        log_dir.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "patroliq.log"))
    except OSError:
        pass  # read-only filesystem (e.g. Streamlit Cloud) — console only

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
