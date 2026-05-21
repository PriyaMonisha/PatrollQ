# filename: api/predictor.py
# purpose:  Model loading and cluster prediction logic for FastAPI

import math
import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))

# Cluster risk mapping derived from geo labels analysis (cluster → risk tier)
# Clusters with highest crime density = HIGH; lowest = LOW
_GEO_RISK_MAP: dict[int, str] = {
    0: "HIGH", 1: "MEDIUM", 2: "HIGH", 3: "LOW",
    4: "MEDIUM", 5: "HIGH", 6: "LOW", 7: "MEDIUM",
}

# Dominant crime type per geo cluster (from cluster profiling on training data)
_GEO_DOMINANT_CRIME: dict[int, str] = {
    0: "THEFT", 1: "BATTERY", 2: "THEFT", 3: "CRIMINAL DAMAGE",
    4: "NARCOTICS", 5: "ASSAULT", 6: "BURGLARY", 7: "MOTOR VEHICLE THEFT",
}

_TEMPORAL_LABELS: dict[int, tuple[str, str]] = {
    0: ("Late Night / Early Morning", "Low activity; opportunistic crimes peak 1–5 AM"),
    1: ("Morning Commute", "Rising activity 6–10 AM; theft and vehicle crime"),
    2: ("Midday Activity", "Peak non-violent incidents 11 AM–3 PM"),
    3: ("Evening Peak", "Highest overall volume 4–11 PM; assault and battery"),
}


@lru_cache(maxsize=1)
def get_temporal_model():
    """Load temporal K-Means model once; cache for process lifetime."""
    path = ARTIFACTS_DIR / "temporal" / "kmeans_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Temporal model not found at {path}. Run the training pipeline first.")
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_geo_labels() -> pd.DataFrame:
    """Load geographic cluster labels once; used to find nearest cluster."""
    path = ARTIFACTS_DIR / "geographic" / "kmeans_labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"Geographic labels not found at {path}. Run the training pipeline first.")
    return pd.read_csv(path)


def predict_geographic(lat: float, lon: float) -> dict:
    """
    Find nearest pre-computed geographic cluster for a given coordinate.
    Uses Euclidean distance on lat/lon (valid for small Chicago area).
    """
    labels_df = get_geo_labels()

    # Vectorised nearest-neighbour on lat/lon
    dlat = labels_df["latitude"].values - lat
    dlon = labels_df["longitude"].values - lon
    distances = dlat ** 2 + dlon ** 2
    nearest_idx = int(np.argmin(distances))
    cluster_id = int(labels_df.iloc[nearest_idx]["cluster"])

    return {
        "cluster_id": cluster_id,
        "cluster_label": f"District Cluster {cluster_id}",
        "crime_risk_level": _GEO_RISK_MAP.get(cluster_id, "MEDIUM"),
        "dominant_crime_type": _GEO_DOMINANT_CRIME.get(cluster_id, "THEFT"),
        "model_name": "kmeans_geo",
        "model_version": "v1",
    }


def predict_temporal(hour: int, day_of_week: int, month: int, is_weekend: bool | None = None) -> dict:
    """
    Predict temporal cluster using pre-trained K-Means on cyclical features.
    """
    model = get_temporal_model()

    # Derive is_weekend from day_of_week if not provided
    if is_weekend is None:
        is_weekend = day_of_week >= 5

    # Build cyclical feature vector matching training feature set
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
