# filename: src/utils/helpers.py
# purpose:  Shared utility functions — JSON serialization, file I/O
# version:  1.0

import json
from pathlib import Path

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles numpy scalar and array types.

    np.bool_ is checked FIRST — on numpy < 2.0, np.bool_ is a subclass
    of np.integer, so checking integer first would return int(True) = 1
    instead of the correct bool True.
    """

    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_json(data: dict, path: Path) -> None:
    """Save dict to JSON using NumpyEncoder. Creates parent dirs automatically."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)


def run_row_to_dict(row, experiment: str) -> dict:
    """
    Convert an MLflow search_runs row (pd.Series) to a clean dict for JSON export.

    MLflow returns dotted keys ("metrics.silhouette", "params.k") and stores
    all param values as strings. This function:
      - Strips the "metrics." and "params." prefixes
      - Excludes NaN values
      - Keeps run metadata (run_id, run_name, experiment, status, start_time)
    """
    import pandas as pd

    metrics = {
        k.replace("metrics.", ""): v
        for k, v in row.items()
        if k.startswith("metrics.") and pd.notna(v)
    }
    params = {
        k.replace("params.", ""): v
        for k, v in row.items()
        if k.startswith("params.") and pd.notna(v)
    }
    return {
        "run_id":     row["run_id"],
        "run_name":   row.get("tags.mlflow.runName", "unknown"),
        "experiment": experiment,
        "status":     row["status"],
        "start_time": str(row["start_time"]),
        "metrics":    metrics,
        "params":     params,
    }
