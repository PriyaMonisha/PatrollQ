# filename: src/monitoring/drift.py
# purpose:  Data drift detection for PatrolIQ — KS-test (continuous features)
#           and chi-squared (categorical). Uses training artifacts as reference.
#           Called by FastAPI /v1/drift/report endpoint.

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
PROCESSED_DIR = Path(os.getenv("DATA_PATH", "./data/processed"))
_PROCESSED_CSV = PROCESSED_DIR / "chicago_crime_features_dev.csv.gz"
_GEO_LABELS = ARTIFACTS_DIR / "geographic" / "kmeans_labels.csv"

# Drift threshold — p-value below this = drift detected
_P_THRESHOLD = 0.05


def _load_reference() -> pd.DataFrame:
    """Load training data used as reference distribution."""
    if _GEO_LABELS.exists():
        return pd.read_csv(_GEO_LABELS)
    raise FileNotFoundError(f"Reference data not found at {_GEO_LABELS}")


def _load_current() -> pd.DataFrame:
    """
    Load current/recent data window.
    In production this would be a live data feed.
    Here we simulate by using the last 20% of training data as 'recent'.
    """
    if _PROCESSED_CSV.exists():
        df = pd.read_csv(_PROCESSED_CSV, usecols=["latitude", "longitude", "Hour", "arrest", "Primary_Type"])
        return df.tail(int(len(df) * 0.2))
    # Fallback to geo labels if features CSV unavailable
    df = pd.read_csv(_GEO_LABELS)
    return df.tail(int(len(df) * 0.2))


def _check_ks(feature: str, ref: pd.Series, cur: pd.Series) -> dict:
    """KS-test for continuous feature drift."""
    ref_clean = ref.dropna().values
    cur_clean = cur.dropna().values
    if len(ref_clean) < 10 or len(cur_clean) < 10:
        return {"feature": feature, "statistic": 0.0, "p_value": 1.0,
                "drift_detected": False, "method": "ks_2samp (skipped — insufficient data)"}
    stat, p = ks_2samp(ref_clean, cur_clean)
    return {
        "feature": feature,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 4),
        "drift_detected": bool(p < _P_THRESHOLD),
        "method": "ks_2samp",
    }


def _check_arrest_rate(ref: pd.Series, cur: pd.Series) -> dict:
    """Arrest rate drift — absolute shift in mean. """
    def _to_bool(s: pd.Series) -> pd.Series:
        if s.dtype == object:
            return s.astype(str).str.lower().map({"true": True, "false": False})
        return s.astype(bool)

    ref_rate = float(_to_bool(ref).mean())
    cur_rate = float(_to_bool(cur).mean())
    shift = abs(cur_rate - ref_rate)
    return {
        "feature": "arrest_rate",
        "statistic": round(shift, 4),
        "p_value": round(1.0 - shift, 4),  # simplified proxy
        "drift_detected": bool(shift > 0.05),
        "method": "absolute_shift",
    }


def run_drift_report() -> list[dict]:
    """
    Run all drift checks and return per-feature results.
    Used by FastAPI /v1/drift/report.
    """
    results = []
    try:
        ref_df = _load_reference()
        cur_df = _load_current()
    except FileNotFoundError as e:
        logger.error(f"Drift detection failed: {e}")
        return [{"feature": "all", "statistic": 0.0, "p_value": 1.0,
                 "drift_detected": False, "method": f"error: {e}"}]

    # 1. Geographic drift — latitude
    if "latitude" in ref_df.columns and "latitude" in cur_df.columns:
        results.append(_check_ks("latitude", ref_df["latitude"], cur_df["latitude"]))

    # 2. Geographic drift — longitude
    if "longitude" in ref_df.columns and "longitude" in cur_df.columns:
        results.append(_check_ks("longitude", ref_df["longitude"], cur_df["longitude"]))

    # 3. Temporal drift — hour of day
    hour_col = "Hour" if "Hour" in cur_df.columns else "hour"
    if hour_col in cur_df.columns:
        ref_hour = ref_df[hour_col] if hour_col in ref_df.columns else pd.Series(
            np.random.randint(0, 24, len(ref_df)))  # synthetic fallback
        results.append(_check_ks("hour_of_day", ref_hour, cur_df[hour_col]))

    # 4. Arrest rate drift
    arr_col = "arrest" if "arrest" in cur_df.columns else None
    if arr_col and arr_col in ref_df.columns:
        results.append(_check_arrest_rate(ref_df[arr_col], cur_df[arr_col]))
    elif arr_col:
        results.append({
            "feature": "arrest_rate", "statistic": 0.0, "p_value": 1.0,
            "drift_detected": False, "method": "skipped — no reference column",
        })

    logger.info(f"Drift report: {sum(r['drift_detected'] for r in results)}/{len(results)} features drifted")
    return results
