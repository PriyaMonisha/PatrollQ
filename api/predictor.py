# filename: api/predictor.py
# purpose:  Model loading and cluster prediction logic for FastAPI

import json
import math
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))

_TEMPORAL_LABELS: dict[int, tuple[str, str]] = {
    0: ("Late Night / Early Morning", "Low activity; opportunistic crimes peak 1–5 AM"),
    1: ("Morning Commute", "Rising activity 6–10 AM; theft and vehicle crime"),
    2: ("Midday Activity", "Peak non-violent incidents 11 AM–3 PM"),
    3: ("Evening Peak", "Highest overall volume 4–11 PM; assault and battery"),
}


@lru_cache(maxsize=1)
def get_geo_model():
    """Load geographic K-Means model once; cache for process lifetime."""
    path = ARTIFACTS_DIR / "geographic" / "kmeans_model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Geographic model not found at {path}. Run: python scripts/run_full_pipeline.py"
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_cluster_profile() -> dict:
    """Load cluster risk profiles once; cache for process lifetime."""
    path = ARTIFACTS_DIR / "geographic" / "cluster_profile.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Cluster profile not found at {path}. Run: python scripts/run_full_pipeline.py"
        )
    with open(path) as f:
        # JSON keys are strings — convert to int for consistent lookup
        return {int(k): v for k, v in json.load(f).items()}


@lru_cache(maxsize=1)
def get_temporal_model():
    """Load temporal K-Means model once; cache for process lifetime."""
    path = ARTIFACTS_DIR / "temporal" / "kmeans_model.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Temporal model not found at {path}. Run: python scripts/run_full_pipeline.py"
        )
    return joblib.load(path)


def predict_geographic(lat: float, lon: float) -> dict:
    """
    Predict geographic crime cluster for a coordinate using K-Means model.
    Returns cluster risk profile from training-time cluster_profile.json.
    """
    model = get_geo_model()
    profile = get_cluster_profile()

    X = np.array([[lat, lon]], dtype=np.float32)
    cluster_id = int(model.predict(X)[0])

    cluster_info = profile.get(cluster_id, {})
    risk_level = cluster_info.get("risk_level", "MEDIUM")
    dominant_crime = cluster_info.get("dominant_crime", "THEFT")
    avg_severity = cluster_info.get("avg_severity_score")
    arrest_rate = cluster_info.get("arrest_rate")

    return {
        "cluster_id": cluster_id,
        "cluster_label": f"District Cluster {cluster_id}",
        "crime_risk_level": risk_level,
        "dominant_crime_type": dominant_crime,
        "avg_severity_score": avg_severity,
        "arrest_rate": arrest_rate,
        "model_name": "kmeans_geo",
        "model_version": "v1",
    }


def predict_temporal(hour: int, day_of_week: int, month: int, is_weekend: bool | None = None) -> dict:
    """
    Predict temporal cluster using pre-trained K-Means on cyclical features.
    """
    model = get_temporal_model()

    if is_weekend is None:
        is_weekend = day_of_week >= 5

    features = np.array([[
        math.sin(2 * math.pi * hour / 24),
        math.cos(2 * math.pi * hour / 24),
        math.sin(2 * math.pi * day_of_week / 7),
        math.cos(2 * math.pi * day_of_week / 7),
        math.sin(2 * math.pi * month / 12),
        math.cos(2 * math.pi * month / 12),
        float(is_weekend),
    ]], dtype=np.float32)

    cluster_id = int(model.predict(features)[0])
    label, description = _TEMPORAL_LABELS.get(cluster_id, (f"Cluster {cluster_id}", "Pattern data unavailable"))

    return {
        "cluster_id": cluster_id,
        "cluster_label": label,
        "pattern_description": description,
        "model_name": "kmeans_temporal",
        "model_version": "v1",
    }
