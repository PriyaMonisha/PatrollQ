# filename: src/monitoring/drift.py
# purpose:  Data drift detection for PatrolIQ — KS-test (continuous features)
#           Uses training reference snapshot as reference distribution.
#           Called by FastAPI /v1/drift/report endpoint.

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp  # chi2_contingency removed — never called

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "./artifacts"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
_PROCESSED_CSV = PROCESSED_DIR / "chicago_crime_features_dev.csv.gz"
_GEO_LABELS = ARTIFACTS_DIR / "geographic" / "kmeans_labels.csv"

# Drift threshold — p-value below this = drift detected
_P_THRESHOLD = 0.05


def _load_reference() -> pd.DataFrame:
    """
    Load training reference distribution.
    Priority: parquet snapshot (saved by pipeline) > features CSV fallback.
    Never falls back to synthetic/random data.
    """
    ref_path = ARTIFACTS_DIR / "reference_distribution.parquet"
    if ref_path.exists():
        logger.info(f"Loading reference snapshot from {ref_path}")
        return pd.read_parquet(ref_path)

    # Backward-compat fallback — features CSV (pre-snapshot pipeline runs)
    for candidate in [
        PROCESSED_DIR / "chicago_crime_features_dev.csv.gz",
        PROCESSED_DIR / "chicago_crime_features.csv.gz",
    ]:
        if candidate.exists():
            logger.warning(
                f"No reference snapshot found. Using fallback: {candidate.name}\n"
                "Run training pipeline to generate proper reference: "
                "python scripts/run_full_pipeline.py"
            )
            df = pd.read_csv(candidate)
            df.columns = df.columns.str.lower()
            wanted = ["latitude", "longitude", "hour", "arrest", "primary_type"]
            available = [c for c in wanted if c in df.columns]
            missing = set(wanted) - set(available)
            if missing:
                logger.warning(f"Reference missing columns: {missing}")
            df = df[available].rename(columns={"hour": "Hour"})
            return df

    raise FileNotFoundError(
        "No reference data found.\n"
        "Fix: python scripts/run_full_pipeline.py\n"
        f"Expected: {ref_path}"
    )


def _load_current() -> pd.DataFrame:
    """
    Load current/recent data window.
    Uses last 20% of training features as proxy for 'recent' data.
    In production this would be a live data feed.
    """
    if _PROCESSED_CSV.exists():
        df = pd.read_csv(
            _PROCESSED_CSV,
            usecols=["latitude", "longitude", "Hour", "arrest", "primary_type"]
        )
        # Normalize column names defensively — handles different pipeline version outputs
        df.columns = df.columns.str.lower()
        if "hour" in df.columns:
            df = df.rename(columns={"hour": "Hour"})  # restore expected capitalization
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
    """Arrest rate drift — absolute shift in mean."""
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

    # Use safe column-name lookup — handles capitalization differences
    for feature, col_candidates in [
        ("latitude",    ["latitude", "Latitude"]),
        ("longitude",   ["longitude", "Longitude"]),
        ("hour_of_day", ["Hour", "hour"]),
    ]:
        ref_col = next((c for c in col_candidates if c in ref_df.columns), None)
        cur_col = next((c for c in col_candidates if c in cur_df.columns), None)
        if ref_col and cur_col:
            results.append(_check_ks(feature, ref_df[ref_col], cur_df[cur_col]))
        else:
            logger.warning(f"Skipping {feature} — column not found in reference or current data")

    # Arrest rate drift
    arr_ref = next((c for c in ["arrest", "Arrest"] if c in ref_df.columns), None)
    arr_cur = next((c for c in ["arrest", "Arrest"] if c in cur_df.columns), None)
    if arr_ref and arr_cur:
        results.append(_check_arrest_rate(ref_df[arr_ref], cur_df[arr_cur]))
    else:
        logger.warning("Skipping arrest_rate — column not found")

    drifted = sum(r["drift_detected"] for r in results)
    logger.info(f"Drift report: {drifted}/{len(results)} features drifted")
    return results
