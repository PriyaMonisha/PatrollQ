# filename: src/data/loader.py
# purpose:  Chicago crime data loading, sampling, validation, and saving for PatrolIQ
# version:  1.0

import gc
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from config import (
    EXPECTED_COLUMNS,
    PROCESSED_CSV,
    SAMPLE_METADATA,
    SAMPLE_SIZE,
)

# ── Logger ──────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────
MIN_EXPECTED_ROWS = 10_000  # fail early if file is truncated

# Normalized column names (snake_case) derived from EXPECTED_COLUMNS
# These are what the loader produces after normalization
NORMALIZED_COLUMNS = [
    c.strip().lower().replace(" ", "_") for c in EXPECTED_COLUMNS
]


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════

def load_raw_csv(
    path: Path,
    validate: bool = True,
    chunksize: int = 200_000,
    n_recent: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load Chicago crime CSV in chunks to avoid OOM on large files.

    If n_recent is set, streams only the N most-recent rows by date
    (peak RAM ≈ 3 × n_recent rows instead of 8.5M rows).

    Steps:
      1. Verify file exists and is .csv or .csv.gz
      2. Log file size
      3. Stream-read in chunks of chunksize rows
         - Normalize column names per chunk
         - Parse 'date' column per chunk (explicit format, 3× faster)
         - If n_recent: prune pool after every chunk to bound memory
      4. Concatenate chunks and optionally take top-N by date
      5. Optionally validate schema and row count
      6. Log load summary

    Args:
        path:      Path to raw CSV or .csv.gz file
        validate:  Run schema and minimum row count checks
        chunksize: Rows per chunk (default 200K ≈ 150 MB/chunk)
        n_recent:  If set, return only the N most-recent rows

    Returns:
        Raw DataFrame with snake_case column names

    Raises:
        FileNotFoundError: File does not exist
        ValueError:        Wrong file type, empty file, or missing columns
    """
    path = Path(path)

    # ── Step 1: Existence & type check ──────────────────────
    if not path.exists():
        raise FileNotFoundError(
            f"✗ File not found: {path}\n"
            "  Download the Chicago crime dataset from:\n"
            "  https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2\n"
            "  → Click Export → CSV → Save to data/raw/"
        )

    suffixes = "".join(path.suffixes).lower()
    if ".csv" not in suffixes:
        raise ValueError(
            f"✗ Expected .csv or .csv.gz file, got: {path.name}"
        )

    # ── Step 2: File size ────────────────────────────────────
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb == 0:
        raise ValueError(f"✗ File is empty (0 bytes): {path}")
    logger.info(f"File: {path.name} | Size: {file_size_mb:.1f} MB")

    # ── Step 3: Chunked streaming load ───────────────────────
    mode = f"n_recent={n_recent:,}" if n_recent else "full load"
    logger.info(f"Loading in chunks (chunksize={chunksize:,}, {mode})...")

    pool: list[pd.DataFrame] = []

    try:
        reader = pd.read_csv(
            path, encoding="utf-8", low_memory=False,
            chunksize=chunksize, on_bad_lines="warn",
        )
    except UnicodeDecodeError:
        logger.warning("UTF-8 failed — retrying with latin-1 encoding")
        reader = pd.read_csv(
            path, encoding="latin-1", low_memory=False,
            chunksize=chunksize, on_bad_lines="warn",
        )

    try:
        for i, chunk in enumerate(reader):
            # Normalize columns immediately (per-chunk, not after full load)
            chunk.columns = (
                chunk.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_", regex=False)
            )
            # Parse date in-place — explicit format is 3× faster than 'mixed'
            # Chicago crime portal format: MM/DD/YYYY HH:MM:SS AM/PM
            chunk["date"] = pd.to_datetime(
                chunk["date"],
                format="%m/%d/%Y %I:%M:%S %p",
                errors="coerce",
            )
            pool.append(chunk)

            # Prune after every chunk when sampling — keeps peak memory bounded
            if n_recent and len(pool) > 1:
                combined = pd.concat(pool, ignore_index=True)
                pool = [combined.nlargest(int(n_recent * 3), "date")]
                del combined

            if (i + 1) % 5 == 0:
                pool_size = sum(len(c) for c in pool)
                logger.info(f"  Chunk {i + 1} | pool: {pool_size:,} rows")
    finally:
        reader.close()  # always release file handle

    # ── Step 4: Assemble final DataFrame ────────────────────
    df = pd.concat(pool, ignore_index=True)
    del pool
    gc.collect()

    if n_recent:
        df = df.nlargest(n_recent, "date").reset_index(drop=True)
        logger.info(f"Final sample: {len(df):,} most-recent rows")

    if df.empty:
        raise ValueError(f"✗ File loaded but contains no rows: {path}")

    # ── Step 5: Optional schema validation ───────────────────
    if validate:
        validate_schema(df)

    # ── Step 6: Summary ──────────────────────────────────────
    _log_load_summary(df, path, file_size_mb)

    return df


def sample_recent_records(
    df: pd.DataFrame,
    n: int = SAMPLE_SIZE,
) -> pd.DataFrame:
    """
    Return the N most recent crime records sorted by Date descending.

    Why most recent? Current patrol allocation needs recent patterns,
    not 2001 crime data.

    Args:
        df: Raw DataFrame with 'date' column (any string format)
        n:  Number of records to sample

    Returns:
        DataFrame with at most n rows, most-recent first, index reset

    Raises:
        ValueError: 'date' column not found
    """
    if "date" not in df.columns:
        raise ValueError(
            f"✗ 'date' column not found — cannot sort by recency.\n"
            f"  Available columns: {list(df.columns[:10])}"
        )

    if len(df) <= n:
        logger.warning(
            f"DataFrame has {len(df):,} rows ≤ requested {n:,}. "
            "Returning all rows."
        )
        return df.copy().reset_index(drop=True)

    logger.info(f"Parsing dates for recency sort ({len(df):,} rows)...")
    # pandas 2.x: format='mixed' handles varying date string formats
    date_parsed = pd.to_datetime(
        df["date"], format="mixed", dayfirst=False, errors="coerce"
    )

    null_dates = date_parsed.isna().sum()
    if null_dates > 0:
        logger.warning(
            f"{null_dates:,} dates failed to parse — placed last in sort"
        )

    sampled = (
        df.assign(_sort_date=date_parsed)
        .sort_values("_sort_date", ascending=False, na_position="last")
        .head(n)
        .drop(columns=["_sort_date"])
        .reset_index(drop=True)
    )

    logger.info(
        f"Sampled {len(sampled):,} most-recent records "
        f"(from {len(df):,} total)"
    )
    return sampled


def validate_schema(df: pd.DataFrame) -> None:
    """
    Assert all 22 expected Chicago columns are present (after normalization).

    Args:
        df: DataFrame with normalized snake_case column names

    Raises:
        ValueError: One or more expected columns are missing
    """
    actual = set(df.columns)
    expected = set(NORMALIZED_COLUMNS)
    missing = expected - actual

    if missing:
        raise ValueError(
            f"✗ Missing {len(missing)} expected column(s): {sorted(missing)}\n"
            f"  Found columns: {sorted(actual)[:10]}..."
        )

    if len(df) < MIN_EXPECTED_ROWS:
        raise ValueError(
            f"✗ Too few rows: {len(df):,}\n"
            f"  Expected ≥ {MIN_EXPECTED_ROWS:,} — check downloaded file."
        )

    logger.info(f"✓ Schema valid | {len(df):,} rows | {df.shape[1]} columns")


def save_processed(
    df: pd.DataFrame,
    path: Optional[Path] = None,
) -> Path:
    """
    Save processed DataFrame as gzipped CSV.

    pd.read_csv('file.csv.gz') decompresses automatically.
    Gzip reduces a 200MB CSV to ~25-35MB.

    Args:
        df:   Processed DataFrame to save
        path: Target path (default: config.PROCESSED_CSV)

    Returns:
        Absolute path to the saved file
    """
    if path is None:
        path = PROCESSED_CSV

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(path, index=False, compression="gzip")

    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info(
        f"✓ Saved: {path.name} | {len(df):,} rows | {size_mb:.1f} MB (gzipped)"
    )

    _save_sample_metadata(df)

    return path.resolve()


# ═══════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════

def _save_sample_metadata(df: pd.DataFrame) -> None:
    """Persist sample metadata JSON for downstream consumers (e.g., Streamlit sidebar)."""
    date_min = date_max = "unknown"
    if "date" in df.columns:
        try:
            dates = pd.to_datetime(
                df["date"], format="mixed", dayfirst=False, errors="coerce"
            )
            date_min = str(dates.min().date()) if not dates.isna().all() else "unknown"
            date_max = str(dates.max().date()) if not dates.isna().all() else "unknown"
        except Exception:
            pass

    metadata = {
        "n_records": int(len(df)),
        "n_columns": int(df.shape[1]),
        "date_min": date_min,
        "date_max": date_max,
        "crime_types": (
            int(df["primary_type"].nunique())
            if "primary_type" in df.columns else 0
        ),
        "sampled_at": datetime.now().isoformat(),
    }

    SAMPLE_METADATA.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLE_METADATA, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Metadata saved: {SAMPLE_METADATA.name}")


def _log_load_summary(
    df: pd.DataFrame,
    path: Path,
    file_size_mb: float,
) -> None:
    """Log a clean load summary. Uses logger (no print()) per PatrolIQ code standards."""
    null_counts = df.isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    logger.info("=" * 56)
    logger.info("  DATA LOAD SUMMARY")
    logger.info("=" * 56)
    logger.info(f"  File    : {path.name}")
    logger.info(f"  Size    : {file_size_mb:.1f} MB")
    logger.info(f"  Rows    : {df.shape[0]:,}")
    logger.info(f"  Columns : {df.shape[1]}")
    logger.info(
        f"  Memory  : "
        f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
    )

    if cols_with_nulls.empty:
        logger.info("  Nulls   : None")
    else:
        for col, cnt in cols_with_nulls.items():
            pct = cnt / len(df) * 100
            logger.info(f"  Null    : {col:30} {cnt:>8,} ({pct:.1f}%)")

    for dtype, count in df.dtypes.value_counts().items():
        logger.info(f"  Dtype   : {str(dtype):15} {count} column(s)")

    logger.info("=" * 56)
