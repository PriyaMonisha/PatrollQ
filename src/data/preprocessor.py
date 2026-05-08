# filename: src/data/preprocessor.py
# purpose:  15-step preprocessing pipeline for Chicago crime data — PatrolIQ
# version:  1.0
#
# PIPELINE:
#   Step 1  : Validate normalized column names
#   Step 2  : Remove duplicate Case Numbers
#   Step 3  : Drop rows with null Latitude or Longitude (CRITICAL — needed for geo clustering)
#   Step 4  : Parse Date column → datetime
#   Step 5  : Extract temporal features from parsed Date (Hour, Day_of_Week, Month, Year)
#   Step 6  : Classify Season from Month
#   Step 7  : Create Is_Weekend boolean flag
#   Step 8  : Impute missing Beat, District, Ward, Community Area with column mode
#   Step 9  : LabelEncode Primary_Type → primary_type_code
#   Step 10 : LabelEncode Location_Description → location_desc_code
#   Step 11 : Cast Arrest and Domestic columns → bool
#   Step 12 : Validate geographic bounds (lat/lon within Chicago range)
#   Step 13 : Cast Latitude, Longitude → float32 (memory optimization)
#   Step 14 : Optionally save LabelEncoder objects for future inference
#   Step 15 : Final null check and summary

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    LABEL_ENCODERS_PATH,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    MODELS_DIR,
)

# ── Logger ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Column sets ─────────────────────────────────────────────
# Administrative boundary columns — mode-imputed when null
ADMIN_COLS = ["beat", "district", "ward", "community_area"]

# Boolean string columns in the raw Chicago CSV
BOOL_COLS = ["arrest", "domestic"]

# Season lookup: month number → season name
_MONTH_TO_SEASON: dict[int, str] = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Fall",   10: "Fall",  11: "Fall",
}


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════

def preprocess_data(
    df: pd.DataFrame,
    save_encoders: bool = True,
) -> pd.DataFrame:
    """
    Run the full 15-step PatrolIQ preprocessing pipeline.

    Input:  Raw Chicago crime DataFrame (from loader.load_raw_csv)
    Output: Clean DataFrame ready for feature engineering

    New columns added (beyond the 22 raw columns):
        Hour            int8   — hour of crime (0–23)
        Day_of_Week     int8   — 0=Monday … 6=Sunday
        Month           int8   — 1–12
        Year            int16  — 4-digit year (re-extracted from Date)
        Season          str    — Winter/Spring/Summer/Fall
        Is_Weekend      bool   — True if Saturday or Sunday
        primary_type_code    int16 — LabelEncoded Primary_Type
        location_desc_code   int16 — LabelEncoded Location_Description

    Args:
        df:            Raw input DataFrame
        save_encoders: Save LabelEncoder objects to models/

    Returns:
        Cleaned and enriched DataFrame
    """
    df = df.copy()
    initial_rows = len(df)

    logger.info("=" * 60)
    logger.info("  PATROLIQ PREPROCESSING PIPELINE v1.0")
    logger.info("=" * 60)
    logger.info(f"  Input : {initial_rows:,} rows × {df.shape[1]} columns")
    logger.info("=" * 60)

    # Step 1: Validate columns
    logger.info("[1/15] Validating column names...")
    df = _validate_columns(df)

    # Step 2: Remove duplicate Case Numbers
    logger.info("[2/15] Removing duplicate Case Numbers...")
    df = _remove_duplicate_cases(df)

    # Step 3: Drop null coordinates (CRITICAL — before any geo work)
    logger.info("[3/15] Dropping rows with null Latitude/Longitude...")
    df = _drop_null_coordinates(df)

    # Step 4: Parse Date column
    logger.info("[4/15] Parsing Date column → datetime...")
    df = _parse_datetime(df)

    # Step 5: Extract temporal features
    logger.info("[5/15] Extracting temporal features (Hour, Day_of_Week, Month, Year)...")
    df = _extract_temporal_features(df)

    # Step 6: Classify Season
    logger.info("[6/15] Classifying Season from Month...")
    df = _create_season(df)

    # Step 7: Create Is_Weekend
    logger.info("[7/15] Creating Is_Weekend flag...")
    df = _create_is_weekend(df)

    # Step 8: Impute administrative boundaries
    logger.info("[8/15] Imputing missing Beat/District/Ward/Community Area...")
    df = _impute_admin_boundaries(df)

    # Step 9: Encode Primary_Type
    logger.info("[9/15] LabelEncoding Primary_Type → primary_type_code...")
    df, le_primary = _encode_column(df, "primary_type", "primary_type_code")

    # Step 10: Encode Location_Description
    logger.info("[10/15] LabelEncoding Location_Description → location_desc_code...")
    df, le_location = _encode_column(df, "location_description", "location_desc_code")

    # Step 11: Cast Arrest and Domestic → bool
    logger.info("[11/15] Casting Arrest and Domestic → bool...")
    df = _cast_boolean_columns(df)

    # Step 12: Validate geographic bounds
    logger.info("[12/15] Validating geographic bounds...")
    df = _validate_geo_bounds(df)

    # Step 13: Memory optimization — lat/lon float64 → float32
    logger.info("[13/15] Casting Latitude/Longitude → float32 (memory optimization)...")
    df = _optimize_coordinates(df)

    # Step 14: Save encoders
    logger.info("[14/15] Saving LabelEncoder objects...")
    if save_encoders:
        _save_encoders(
            {"primary_type": le_primary, "location_description": le_location}
        )
    else:
        logger.info("  save_encoders=False — skipped")

    # Step 15: Final validation
    logger.info("[15/15] Final null check and summary...")
    _validate_final(df)

    # ── Summary ─────────────────────────────────────────────
    dropped = initial_rows - len(df)
    logger.info("=" * 60)
    logger.info("  PREPROCESSING COMPLETE ✓")
    logger.info("=" * 60)
    logger.info(f"  Input rows  : {initial_rows:,}")
    logger.info(f"  Output rows : {len(df):,}")
    logger.info(f"  Dropped     : {dropped:,} ({dropped/initial_rows*100:.1f}%)")
    logger.info(f"  Columns     : {df.shape[1]}")
    logger.info(
        f"  Memory      : "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )
    logger.info("=" * 60)

    return df


# ═══════════════════════════════════════════════════════════
# STEP FUNCTIONS (private)
# ═══════════════════════════════════════════════════════════

def _validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Confirm expected snake_case columns exist. Log any extras."""
    from config import EXPECTED_COLUMNS
    expected_normalized = {
        c.strip().lower().replace(" ", "_") for c in EXPECTED_COLUMNS
    }
    actual = set(df.columns)
    missing = expected_normalized - actual

    if missing:
        raise ValueError(
            f"✗ Missing columns after normalization: {sorted(missing)}\n"
            f"  Run loader.load_raw_csv() before preprocessing."
        )

    extra = actual - expected_normalized
    if extra:
        logger.info(f"  Extra columns present (kept): {sorted(extra)}")

    logger.info(f"  ✓ All 22 expected columns present")
    return df


def _remove_duplicate_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with duplicate case_number (keep first occurrence)."""
    before = len(df)
    df = df.drop_duplicates(subset=["case_number"], keep="first")
    removed = before - len(df)

    if removed > 0:
        logger.info(f"  Removed {removed:,} duplicate Case Numbers ✓")
    else:
        logger.info("  No duplicate Case Numbers found ✓")

    return df.reset_index(drop=True)


def _drop_null_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where Latitude or Longitude is null.

    WHY: Geographic clustering requires valid coordinates.
    These rows cannot be clustered and must be removed, not filled —
    fabricating lat/lon would create false hotspots.
    """
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    removed = before - len(df)

    if removed > 0:
        pct = removed / before * 100
        logger.info(
            f"  Dropped {removed:,} rows with null coordinates "
            f"({pct:.1f}%) ✓"
        )
    else:
        logger.info("  No null coordinates found ✓")

    return df.reset_index(drop=True)


def _parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse 'date' column to pandas datetime.

    Chicago Data Portal format: 'MM/DD/YYYY HH:MM:SS AM/PM'
    format='mixed' handles varying formats gracefully (pandas 2.x).
    Rows with unparseable dates get NaT — temporal steps handle them.
    """
    original_dtype = df["date"].dtype

    if original_dtype == "datetime64[ns]":
        logger.info("  'date' already datetime — skipping parse ✓")
        return df

    df["date"] = pd.to_datetime(
        df["date"], format="mixed", dayfirst=False, errors="coerce"
    )

    null_dates = df["date"].isna().sum()
    if null_dates > 0:
        logger.warning(
            f"  {null_dates:,} rows have unparseable dates → NaT "
            f"(will be dropped in temporal extraction)"
        )
    else:
        logger.info("  ✓ All dates parsed successfully")

    return df


def _extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Hour, Day_of_Week, Month, Year from parsed 'date' column.

    Day_of_Week: 0=Monday, 1=Tuesday, ..., 5=Saturday, 6=Sunday (pandas convention)
    Rows with NaT dates are dropped here (they cannot contribute temporal features).
    """
    # Drop rows where date parsing failed — they have no temporal info
    before = len(df)
    df = df.dropna(subset=["date"])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f"  Dropped {dropped:,} rows with NaT dates")

    df["Hour"] = df["date"].dt.hour.astype("int8")
    df["Day_of_Week"] = df["date"].dt.dayofweek.astype("int8")
    df["Month"] = df["date"].dt.month.astype("int8")
    # Re-extract Year from parsed Date for consistency
    # (Chicago CSV already has 'year' column but we derive ours from the Date)
    df["Year"] = df["date"].dt.year.astype("int16")

    logger.info(
        f"  Hour range   : {df['Hour'].min()}–{df['Hour'].max()} ✓"
    )
    logger.info(
        f"  Year range   : {df['Year'].min()}–{df['Year'].max()} ✓"
    )

    return df.reset_index(drop=True)


def _create_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each crime into a meteorological season.

    Winter = Dec(12), Jan(1), Feb(2)
    Spring = Mar(3),  Apr(4), May(5)
    Summer = Jun(6),  Jul(7), Aug(8)
    Fall   = Sep(9),  Oct(10), Nov(11)
    """
    df["Season"] = df["Month"].map(_MONTH_TO_SEASON)

    dist = df["Season"].value_counts().to_dict()
    logger.info(f"  Season distribution: {dist} ✓")

    return df


def _create_is_weekend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Boolean flag: True if the crime occurred on Saturday (5) or Sunday (6).
    Day_of_Week follows pandas convention: 0=Monday … 6=Sunday.
    """
    df["Is_Weekend"] = df["Day_of_Week"].isin([5, 6])

    weekend_count = df["Is_Weekend"].sum()
    pct = weekend_count / len(df) * 100
    logger.info(
        f"  Weekend crimes: {weekend_count:,} ({pct:.1f}%) ✓"
    )

    return df


def _impute_admin_boundaries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill null values in Beat, District, Ward, Community Area using column mode.

    WHY mode imputation: These are categorical ID codes (not continuous values).
    The most common value is the best substitute when the actual value is unknown.
    """
    for col in ADMIN_COLS:
        if col not in df.columns:
            continue
        null_count = df[col].isnull().sum()
        if null_count == 0:
            logger.info(f"  {col:20}: no nulls ✓")
            continue

        mode_val = df[col].mode().iloc[0]
        df[col] = df[col].fillna(mode_val)
        logger.info(
            f"  {col:20}: {null_count:,} nulls → mode {mode_val} ✓"
        )

    return df


def _encode_column(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    LabelEncode a string column and add the numeric codes as a new column.

    The original string column is preserved for display purposes.
    The encoded column is cast to int16 (max 33 crime types → fits in int8,
    but int16 used for location descriptions which can be 100+).

    Returns:
        (updated DataFrame, fitted LabelEncoder)
    """
    if source_col not in df.columns:
        raise ValueError(
            f"✗ Column '{source_col}' not found for encoding.\n"
            f"  Available: {list(df.columns)}"
        )

    # Fill any nulls with 'UNKNOWN' so the encoder sees a complete series
    null_count = df[source_col].isnull().sum()
    if null_count > 0:
        df[source_col] = df[source_col].fillna("UNKNOWN")
        logger.warning(
            f"  {source_col}: {null_count:,} nulls filled with 'UNKNOWN'"
        )

    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[source_col].astype(str)).astype("int16")

    n_classes = len(le.classes_)
    logger.info(
        f"  {source_col:30} → {target_col} "
        f"({n_classes} unique classes) ✓"
    )

    return df, le


def _cast_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Arrest and Domestic from object/string → bool.

    Chicago CSV exports these as 'true'/'false' strings (lowercase) or
    sometimes True/False Python booleans depending on export method.
    Handle both forms robustly.
    """
    for col in BOOL_COLS:
        if col not in df.columns:
            continue

        if df[col].dtype == bool:
            logger.info(f"  {col}: already bool ✓")
            continue

        # Normalize to string, strip, lowercase → map to bool
        series_str = df[col].astype(str).str.strip().str.lower()
        df[col] = series_str.map({"true": True, "false": False, "1": True, "0": False})

        null_after = df[col].isnull().sum()
        if null_after > 0:
            # Fill remaining nulls with False (most conservative assumption)
            df[col] = df[col].fillna(False)
            logger.warning(
                f"  {col}: {null_after:,} unmapped values → False"
            )

        df[col] = df[col].astype(bool)
        true_pct = df[col].sum() / len(df) * 100
        logger.info(f"  {col:15}: cast → bool | True: {true_pct:.1f}% ✓")

    return df


def _validate_geo_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove records where lat/lon fall outside Chicago's geographic bounds.

    Chicago bounds: Lat 41.6–42.0, Lon -87.9 to -87.5
    Out-of-bounds coordinates indicate data entry errors.
    """
    before = len(df)
    mask_valid = (
        df["latitude"].between(LAT_MIN, LAT_MAX, inclusive="both") &
        df["longitude"].between(LON_MIN, LON_MAX, inclusive="both")
    )
    df = df[mask_valid].reset_index(drop=True)

    removed = before - len(df)
    if removed > 0:
        logger.info(
            f"  Removed {removed:,} out-of-bounds coordinate rows "
            f"(outside Chicago lat {LAT_MIN}–{LAT_MAX}, "
            f"lon {LON_MIN}–{LON_MAX}) ✓"
        )
    else:
        logger.info("  All coordinates within Chicago bounds ✓")

    return df


def _optimize_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast Latitude and Longitude from float64 → float32.

    WHY: Coordinates need ~4 decimal places (~11m precision).
    float32 provides 7 significant digits — more than sufficient.
    Halves memory usage for the two highest-cardinality float columns.
    500K rows × 2 cols × (8→4 bytes) = 4MB saved.
    """
    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    logger.info("  Latitude/Longitude: float64 → float32 (4MB saved) ✓")
    return df


def _save_encoders(encoders: dict[str, LabelEncoder]) -> None:
    """
    Persist fitted LabelEncoder objects to disk.

    Also saves human-readable class mappings to JSON for Streamlit display.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Save encoder objects (for consistent encoding in future runs)
    joblib.dump(encoders, LABEL_ENCODERS_PATH)
    logger.info(f"  ✓ Encoders saved: {LABEL_ENCODERS_PATH}")

    # Save human-readable class→code mapping as JSON
    mapping_path = MODELS_DIR / "label_encoder_mappings.json"
    mappings: dict[str, dict[str, int]] = {}
    for name, le in encoders.items():
        codes = le.transform(le.classes_).tolist()
        mappings[name] = dict(zip(le.classes_.tolist(), codes))

    with open(mapping_path, "w") as f:
        json.dump(mappings, f, indent=2, sort_keys=True)
    logger.info(f"  ✓ Mappings saved: {mapping_path.name}")


def _validate_final(df: pd.DataFrame) -> None:
    """
    Post-processing validation:
    1. Zero nulls in critical columns
    2. New columns are present
    3. Coordinate bounds
    """
    critical_cols = [
        "latitude", "longitude", "date", "Hour",
        "Day_of_Week", "Month", "Year", "Season",
        "Is_Weekend", "primary_type_code", "location_desc_code",
        "Arrest", "Domestic",
    ]

    for col in critical_cols:
        if col not in df.columns:
            logger.warning(f"  ✗ Expected column missing: {col}")
            continue
        null_count = df[col].isnull().sum()
        if null_count > 0:
            logger.warning(f"  ✗ {col}: {null_count:,} nulls remaining")
        else:
            logger.info(f"  ✓ {col}: no nulls")

    # Bounds check on final lat/lon
    lat_ok = df["latitude"].between(LAT_MIN, LAT_MAX).all()
    lon_ok = df["longitude"].between(LON_MIN, LON_MAX).all()
    if lat_ok and lon_ok:
        logger.info("  ✓ All coordinates within Chicago bounds")
    else:
        logger.warning("  ✗ Some coordinates outside Chicago bounds after validation")

    total_nulls = df.isnull().sum().sum()
    if total_nulls == 0:
        logger.info("  ✓ Zero nulls in final DataFrame")
    else:
        cols_with_nulls = df.columns[df.isnull().any()].tolist()
        logger.warning(f"  {total_nulls:,} nulls remaining in: {cols_with_nulls}")
