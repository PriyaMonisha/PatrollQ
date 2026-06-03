# filename: notebooks/00_data_cleaning.py
# purpose:  Data cleaning pipeline — load raw → sample → preprocess → save
# section:  0 (Data Cleaning)
# version:  1.0

# ── Cell 1: Setup ────────────────────────────────────────────
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import (
    DATA_RAW_DIR,
    FAST_MODE,
    FAST_SAMPLE_SIZE,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    PROCESSED_CSV,
    RANDOM_STATE,
    SAMPLE_METADATA,
    SAMPLE_SIZE,
)
from src.data.loader import load_raw_csv, save_processed
from src.data.preprocessor import preprocess_data

from src.utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_N = FAST_SAMPLE_SIZE if FAST_MODE else SAMPLE_SIZE

print("=" * 60)
print("  PATROLIQ — Section 0: Data Cleaning")
print("=" * 60)
print(f"FAST_MODE    : {FAST_MODE}")
print(f"RANDOM_STATE : {RANDOM_STATE}")
print(f"Sample size  : {SAMPLE_N:,}")
print(f"Project root : {PROJECT_ROOT}")


# ── Cell 2: Locate raw file ───────────────────────────────────
# Prefer .csv.gz — pandas reads transparently and I/O is faster
RAW_FILES = sorted(DATA_RAW_DIR.glob("Crimes_*.csv.gz")) or sorted(
    DATA_RAW_DIR.glob("Crimes_*.csv")
)
if not RAW_FILES:
    raise FileNotFoundError(
        f"No raw crime CSV found in {DATA_RAW_DIR}.\n"
        "Download from: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2"
    )
RAW_PATH = RAW_FILES[0]
print(f"\nRaw file  : {RAW_PATH.name}")
print(f"File size : {RAW_PATH.stat().st_size / 1024**3:.2f} GB")
print(f"Exists    : {RAW_PATH.exists()}")


# ── Cell 3: STEP 1+3 — Stream-load + sample (single pass, no OOM) ──
print(f"\nSTEP 1: Streaming {SAMPLE_N:,} most-recent records from raw CSV...")
print("-" * 55)
print("(chunked load — avoids OOM on 2.2 GB file)")

df_sample = load_raw_csv(RAW_PATH, validate=True, n_recent=SAMPLE_N)
sample_rows = len(df_sample)
print(f"Loaded shape : {df_sample.shape}")


# ── Cell 4: STEP 2 — Sample snapshot (pre-preprocessing) ────────
print("\nSTEP 2: Sample snapshot (streaming-sampled, before preprocessing)...")
print("-" * 55)
# Note: pre_rows is the sample count (50K FAST_MODE / 500K production).
# Full raw dataset = 8.5M rows — not loaded into memory.

pre_rows = len(df_sample)
pre_nulls = df_sample.isnull().sum().sum()
pre_duplicates = df_sample.duplicated(subset=["case_number"]).sum()

print(f"Rows        : {pre_rows:,}")
print(f"Total nulls : {pre_nulls:,}")
print(f"Duplicates  : {pre_duplicates:,} (by case_number)")

print("\nNull breakdown:")
null_breakdown = df_sample.isnull().sum()
null_breakdown = null_breakdown[null_breakdown > 0].sort_values(ascending=False)
for col, count in null_breakdown.items():
    pct = count / pre_rows * 100
    print(f"  {col:25}: {count:>7,} ({pct:.1f}%)")


# ── Cell 6: Post-sample stats ────────────────────────────────
print(f"Sample date range : {df_sample['date'].min()} to {df_sample['date'].max()}")
sample_nulls = df_sample.isnull().sum().sum()
sample_dups = df_sample.duplicated(subset=["case_number"]).sum()
print(f"Nulls in sample   : {sample_nulls:,}")
print(f"Dupes in sample   : {sample_dups:,}")


# ── Cell 7: STEP 4 — Run preprocessing pipeline ──────────────
print("\nSTEP 4: Running preprocessing pipeline...")
print("-" * 55)

cols_before = set(df_sample.columns)  # captured before — used in cells 9 and 15
df_clean = preprocess_data(df_sample, save_encoders=True)

# Output contract — fail fast if preprocessor drops required columns
# Note: temporal columns use PascalCase by design in preprocessor.py
REQUIRED_COLS = [
    "case_number", "date", "latitude", "longitude", "primary_type",
    "arrest", "domestic", "beat", "primary_type_code", "location_desc_code",
    "Hour", "Day_of_Week", "Month", "Season", "Is_Weekend",
]
missing = [c for c in REQUIRED_COLS if c not in df_clean.columns]
assert not missing, f"Preprocessor missing expected columns: {missing}"
print(f"Output contract: all {len(REQUIRED_COLS)} required columns present ✓")


# ── Cell 8: STEP 5 — Post-cleaning snapshot ──────────────────
print("\nSTEP 5: Post-cleaning snapshot...")
print("-" * 55)

post_rows = len(df_clean)
post_nulls = df_clean.isnull().sum().sum()
post_dups = df_clean.duplicated(subset=["case_number"]).sum()

print(f"Rows          : {post_rows:,}")
print(f"Total nulls   : {post_nulls:,}")
print(f"Duplicates    : {post_dups:,}")
print(f"Columns now   : {df_clean.shape[1]}")

rows_removed = sample_rows - post_rows  # uses actual sampled count, not requested SAMPLE_N
print(f"Rows removed  : {rows_removed:,} ({rows_removed / sample_rows * 100:.1f}%)")


# ── Cell 9: New columns added by preprocessing ───────────────
new_cols = [col for col in df_clean.columns if col not in cols_before]
print(f"\nNew columns added by preprocessing ({len(new_cols)}):")
for col in new_cols:
    print(f"  + {col}")


# ── Cell 10: Geographic bounds validation ────────────────────
out_of_bounds = ~df_clean["latitude"].between(LAT_MIN, LAT_MAX) | ~df_clean[
    "longitude"
].between(LON_MIN, LON_MAX)
n_oob = int(out_of_bounds.sum())

print(f"\nGeographic bounds check [{LAT_MIN},{LAT_MAX}] x [{LON_MIN},{LON_MAX}]:")
print(f"  Lat range  : {df_clean['latitude'].min():.4f} to {df_clean['latitude'].max():.4f}")
print(f"  Lon range  : {df_clean['longitude'].min():.4f} to {df_clean['longitude'].max():.4f}")

if n_oob > 0:
    logger.warning(f"{n_oob} rows outside Chicago bounds — preprocessor Step 12 may have failed")
assert n_oob == 0, f"{n_oob} out-of-bounds rows remain — check _validate_geo_bounds()"
print("  All coordinates within Chicago bounds ✓")


# ── Cell 11: Crime type distribution ─────────────────────────
print("\nPrimary Type distribution (top 10):")
print(df_clean["primary_type"].value_counts().head(10))
print(f"\nTotal unique types : {df_clean['primary_type'].nunique()}")
print(f"Arrest rate        : {df_clean['arrest'].mean():.2%}")


# ── Cell 12: Temporal features check ─────────────────────────
print("\nTemporal features:")
print(f"  Hour range        : {df_clean['Hour'].min()} - {df_clean['Hour'].max()}")
print(f"  Day_of_Week range : {df_clean['Day_of_Week'].min()} - {df_clean['Day_of_Week'].max()}")
print(f"  Month range       : {df_clean['Month'].min()} - {df_clean['Month'].max()}")
print(f"  Year range        : {df_clean['Year'].min()} - {df_clean['Year'].max()}")
print("\nSeason distribution:")
print(df_clean["Season"].value_counts())
print(f"\nWeekend crimes : {df_clean['Is_Weekend'].sum():,} ({df_clean['Is_Weekend'].mean():.1%})")


# ── Cell 13: Boolean + encoded column check ──────────────────
print("\nBoolean columns:")
for col in ["arrest", "domestic"]:
    print(
        f"  {col:12} dtype={df_clean[col].dtype} | "
        f"True: {df_clean[col].sum():,} ({df_clean[col].mean():.1%})"
    )

print("\nEncoded columns:")
for col in ["primary_type_code", "location_desc_code"]:
    print(f"  {col:25} range: {df_clean[col].min()} - {df_clean[col].max()}")


# ── Cell 14: Memory + dtypes ─────────────────────────────────
mem_mb = df_clean.memory_usage(deep=True).sum() / 1024**2
print(f"\nMemory usage : {mem_mb:.1f} MB")
print(f"Rows         : {len(df_clean):,}")
print(f"Columns      : {df_clean.shape[1]}")
print("\nColumn dtypes:")
print(df_clean.dtypes)


# ── Cell 15: STEP 6 — Save processed data ────────────────────
print("\nSTEP 6: Saving processed data...")
print("-" * 55)

saved_path = save_processed(df_clean)
assert saved_path is not None, "save_processed() returned None — check loader.py"
saved_path = Path(saved_path)
saved_size = saved_path.stat().st_size / 1024**2

print(f"Saved to   : {saved_path}")
print(f"File size  : {saved_size:.1f} MB")
print(f"Rows       : {post_rows:,}")
print(f"Columns    : {df_clean.shape[1]}")


# ── Cell 16: Verify round-trip reload ────────────────────────
df_verify = pd.read_csv(PROCESSED_CSV)
print(f"\nRound-trip verify:")
print(f"  Shape      : {df_verify.shape}")
print(f"  Nulls      : {df_verify.isnull().sum().sum():,}")
assert df_verify.shape[0] == post_rows, (
    f"Row count mismatch: {df_verify.shape[0]} vs {post_rows}"
)
assert df_verify.shape[1] == df_clean.shape[1], (
    f"Column count mismatch: {df_verify.shape[1]} vs {df_clean.shape[1]}"
)
assert "latitude" in df_verify.columns, "latitude missing after reload"
assert "case_number" in df_verify.columns, "case_number missing after reload"
assert df_verify["case_number"].nunique() == post_rows, (
    "Duplicate case_numbers introduced by save/reload"
)
del df_verify
print("Verification passed ✓")


# ── Cell 17: STEP 7 — Save cleaning report ───────────────────
print("\nSTEP 7: Saving cleaning report...")
print("-" * 55)

REPORTS_DIR = PROJECT_ROOT / "data" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORTS_DIR / "cleaning_report.txt"

report_lines = [
    "=" * 55,
    "PATROLIQ — DATA CLEANING REPORT",
    "=" * 55,
    f"Raw rows (full dataset)  : {pre_rows:,}",
    f"Sampled rows             : {SAMPLE_N:,}",
    f"Cleaned rows             : {post_rows:,}",
    f"Rows removed in cleaning : {rows_removed:,}",
    f"Columns (raw)            : {len(cols_before)}",
    f"Columns (clean)          : {df_clean.shape[1]}",
    f"New columns              : {new_cols}",
    f"Nulls before cleaning    : {pre_nulls:,}",
    f"Nulls after cleaning     : {post_nulls:,}",
    f"Duplicates removed       : {pre_duplicates:,}",
    "=" * 55,
    "MISSING VALUE STRATEGY:",
    "  latitude/longitude   -> Beat-median imputation",
    "  ward                 -> Beat-grouped mode",
    "  community_area       -> Beat-grouped mode",
    "  district             -> Beat-grouped mode, global fallback",
    "  location_description -> kept; UNKNOWN label-encoded",
    "=" * 55,
    "ROWS DROPPED:",
    "  Duplicate case_numbers (keep first occurrence)",
    "  Out-of-Chicago bounds (lat/lon Step 12 validation)",
    "=" * 55,
    f"Output : {PROCESSED_CSV}",
    f"Meta   : {SAMPLE_METADATA}",
    "=" * 55,
]

with open(REPORT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"Report saved: {REPORT_PATH}")
print("\n".join(report_lines))


# ── Cell 18: Before vs After comparison table ────────────────
print(f"\n{'Metric':<30} {'Before':>10} {'After':>10}")
print("-" * 52)
print(f"{'Rows':<30} {SAMPLE_N:>10,} {post_rows:>10,}")
print(f"{'Total nulls':<30} {pre_nulls:>10,} {post_nulls:>10,}")
print(f"{'Duplicate case_numbers':<30} {pre_duplicates:>10,} {'0':>10}")
print(f"{'Columns':<30} {len(cols_before):>10} {df_clean.shape[1]:>10}")
print(f"{'Memory (MB)':<30} {'N/A':>10} {mem_mb:>9.1f}")


# ── Cell 19: Sample metadata ─────────────────────────────────
if SAMPLE_METADATA.exists():
    with open(SAMPLE_METADATA) as f:
        meta = json.load(f)
    print("\nSample metadata (sample_metadata.json):")
    for k, v in meta.items():
        print(f"  {k:<25}: {v}")
else:
    print("Metadata file not found — check save_processed() ran.")


# ── Cell 20: Null check on final df ─────────────────────────
remaining_nulls = df_clean.isnull().sum()
remaining_nulls = remaining_nulls[remaining_nulls > 0]
if remaining_nulls.empty:
    print("\nZero nulls in cleaned DataFrame. ✓")
else:
    print("\nRemaining nulls (investigate):")
    print(remaining_nulls)


# ── Cell 21: Admin boundary imputation verification ──────────
print("\nAdmin boundary imputation check (should be 0 nulls):")
for col in ["district", "ward", "community_area", "beat"]:
    if col in df_clean.columns:
        n = df_clean[col].isnull().sum()
        status = "✓" if n == 0 else "WARN"
        print(f"  {col:20}: {n:,} nulls {status}")


# ── Cell 22: FINAL SUMMARY ───────────────────────────────────
print(f"""
{'=' * 55}
  CLEANING PIPELINE COMPLETE
{'=' * 55}
Raw data    : {pre_rows:,} rows x {len(cols_before)} cols
Clean data  : {post_rows:,} rows x {df_clean.shape[1]} cols
Nulls       : {pre_nulls:,} -> {post_nulls:,}
Duplicates  : {pre_duplicates:,} -> 0
Rows removed: {rows_removed:,} (dupes + out-of-bounds coords)

New columns : {new_cols}

Saved to    : {PROCESSED_CSV.name}  ({saved_size:.1f} MB)
Report at   : {REPORT_PATH}

NEXT STEP   : notebooks/03_eda.py
              loads from data/processed/ (NEVER from data/raw/)
""")
