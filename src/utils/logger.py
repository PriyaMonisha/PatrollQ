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
    import sys
    import io

    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"

    # Use UTF-8 for console so Unicode symbols (✓ →) work on Windows cp1252 terminals
    try:
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        console_handler = logging.StreamHandler(stream)
    except AttributeError:
        console_handler = logging.StreamHandler()  # fallback (e.g. captured stdout)

    handlers: list[logging.Handler] = [console_handler]

    log_dir = Path("logs")
    try:
        log_dir.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "patroliq.log", encoding="utf-8"))
    except OSError:
        pass  # read-only filesystem (e.g. Streamlit Cloud) — console only

    logging.basicConfig(level=level, format=fmt, handlers=handlers, force=True)
