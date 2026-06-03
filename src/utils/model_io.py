# filename: src/utils/model_io.py
# purpose:  Shared save/load utilities for sklearn models and JSON artifacts

import json
import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


def save_model(model, directory: Path, filename: str) -> Path:
    """Save a sklearn model artifact using joblib."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    joblib.dump(model, path)
    logger.info(f"Model saved: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


def load_model(path: Path):
    """Load a sklearn model artifact with an actionable error message."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            "Fix: python scripts/run_full_pipeline.py"
        )
    model = joblib.load(path)
    logger.info(f"Model loaded: {path}")
    return model


def save_json(data: dict, directory: Path, filename: str) -> Path:
    """Save a dict as a JSON artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"JSON saved: {path}")
    return path


def load_json(path: Path) -> dict:
    """Load a JSON artifact with an actionable error message."""
    if not path.exists():
        raise FileNotFoundError(
            f"JSON artifact not found: {path}\n"
            "Fix: python scripts/run_full_pipeline.py"
        )
    with open(path) as f:
        return json.load(f)
