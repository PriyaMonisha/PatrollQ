# filename: notebooks/01_data_acquisition.py
# purpose:  Section 2 — Data acquisition, sampling, and preprocessing for PatrolIQ
# version:  1.0

# ── Cell 1: Setup ────────────────────────────────────────────
import logging
import sys
from pathlib import Path

# Ensure project root is on path (needed when running from notebooks/)
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from config import (
    DATA_RAW_DIR,
    FAST_MODE,
    FAST_SAMPLE_SIZE,
    PROCESSED_CSV,
    RANDOM_STATE,
    SAMPLE_SIZE,
)
from src.data.loader import load_raw_csv, sample_recent_records, save_processed
from src.data.preprocessor import preprocess_data

from src.utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

print("=" * 60)
print("  PatrolIQ — Section 2: Data Acquisition")
print("=" * 60)
print(f"  FAST_MODE    : {FAST_MODE}")
print(f"  RANDOM_STATE : {RANDOM_STATE}")
sample_n = FAST_SAMPLE_SIZE if FAST_MODE else SAMPLE_SIZE
print(f"  Sample size  : {sample_n:,} records")
print("=" * 60)


# ── Cell 2: Find raw data file ───────────────────────────────
# The Chicago crime CSV must be downloaded manually from:
# https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2
# → Export → CSV → Save to data/raw/

raw_files = list(DATA_RAW_DIR.glob("*.csv")) + list(DATA_RAW_DIR.glob("*.csv.gz"))

if not raw_files:
    print("\n✗ No CSV file found in data/raw/")
    print("  Please download the Chicago crime dataset:")
    print("  1. Go to: https://data.cityofchicago.org/Public-Safety/")
    print("             Crimes-2001-to-Present/ijzp-q8t2")
    print("  2. Click Export → CSV")
    print("  3. Save to: data/raw/chicago_crimes.csv")
    print("\n  File size: ~1.7 GB | ~7.8M records | 22 columns")
    sys.exit(1)

RAW_CSV = raw_files[0]
print(f"\nUsing raw file: {RAW_CSV.name}")


# ── Cell 3: Load raw CSV ─────────────────────────────────────
print("\n--- Loading raw CSV ---")
df_raw = load_raw_csv(RAW_CSV, validate=True)

print(f"\nRaw data shape : {df_raw.shape}")
print(f"Columns        : {list(df_raw.columns)}")
print(f"Memory usage   : {df_raw.memory_usage(deep=True).sum() / 1024**2:.1f} MB")


# ── Cell 4: Sample most recent records ──────────────────────
print(f"\n--- Sampling {sample_n:,} most recent records ---")
df_sampled = sample_recent_records(df_raw, n=sample_n)

# Free raw DataFrame memory — no longer needed
del df_raw
import gc
gc.collect()

print(f"Sampled shape  : {df_sampled.shape}")
print(f"Memory usage   : {df_sampled.memory_usage(deep=True).sum() / 1024**2:.1f} MB")


# ── Cell 5: Quick EDA before preprocessing ──────────────────
print("\n--- Pre-processing quick look ---")
print(f"\nNull counts (top 10):")
null_counts = df_sampled.isnull().sum().sort_values(ascending=False)
print(null_counts[null_counts > 0].head(10).to_string())

print(f"\nData types:")
print(df_sampled.dtypes.value_counts().to_string())

if "primary_type" in df_sampled.columns:
    print(f"\nTop 10 crime types:")
    print(df_sampled["primary_type"].value_counts().head(10).to_string())

if "date" in df_sampled.columns:
    # Show raw date sample to confirm format
    print(f"\nDate column sample (first 3):")
    print(df_sampled["date"].head(3).to_string())


# ── Cell 6: Run preprocessing pipeline ──────────────────────
print("\n--- Running preprocessing pipeline ---")
df_processed = preprocess_data(df_sampled, save_encoders=True)

# Free sampled raw data
del df_sampled
gc.collect()


# ── Cell 7: Validate preprocessing output ───────────────────
print("\n--- Preprocessing output validation ---")
print(f"\nProcessed shape : {df_processed.shape}")
print(f"Memory usage    : {df_processed.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

print(f"\nNew columns added:")
new_cols = [
    "Hour", "Day_of_Week", "Month", "Year", "Season",
    "Is_Weekend", "primary_type_code", "location_desc_code",
]
for col in new_cols:
    if col in df_processed.columns:
        sample_vals = df_processed[col].value_counts().head(3).to_dict()
        print(f"  {col:25}: {sample_vals}")
    else:
        print(f"  {col:25}: ✗ MISSING")

print(f"\nNull counts after preprocessing:")
final_nulls = df_processed.isnull().sum()
null_remaining = final_nulls[final_nulls > 0]
if null_remaining.empty:
    print("  ✓ Zero nulls")
else:
    print(null_remaining.to_string())

print(f"\nData types after preprocessing:")
print(df_processed.dtypes.value_counts().to_string())

# Spot-check coordinate columns
print(f"\nCoordinate ranges:")
print(f"  latitude  : {df_processed['latitude'].min():.4f} – {df_processed['latitude'].max():.4f}")
print(f"  longitude : {df_processed['longitude'].min():.4f} – {df_processed['longitude'].max():.4f}")

# Arrest and Domestic dtypes
print(f"\nBoolean columns:")
for col in ["Arrest", "Domestic"]:
    if col in df_processed.columns:
        print(f"  {col}: dtype={df_processed[col].dtype} | True={df_processed[col].sum():,}")


# ── Cell 8: Save processed data ─────────────────────────────
print("\n--- Saving processed data ---")
if FAST_MODE:
    # In FAST_MODE, save to a separate dev file to avoid overwriting production data
    save_path = PROCESSED_CSV.parent / "chicago_crime_dev.csv.gz"
    print(f"FAST_MODE=True → saving to: {save_path.name} (not overwriting production)")
else:
    save_path = PROCESSED_CSV
    print(f"FAST_MODE=False → saving to: {save_path.name} (production file)")

saved_path = save_processed(df_processed, path=save_path)
print(f"\n✓ Saved: {saved_path}")


# ── Cell 9: Section summary ──────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION 2 SUMMARY")
print("=" * 60)
print(f"  Records processed : {len(df_processed):,}")
print(f"  Output columns    : {df_processed.shape[1]}")
print(f"  Output file       : {save_path.name}")
print(f"  Encoders saved    : models/label_encoders.pkl")
print(f"  Metadata saved    : data/processed/sample_metadata.json")
print("=" * 60)

print("\nKEY FINDINGS:")
if "Year" in df_processed.columns:
    print(f"  Date range   : {df_processed['Year'].min()} – {df_processed['Year'].max()}")
if "primary_type" in df_processed.columns:
    print(f"  Crime types  : {df_processed['primary_type'].nunique()} distinct types")
if "Hour" in df_processed.columns:
    peak_hour = df_processed["Hour"].value_counts().idxmax()
    print(f"  Peak hour    : {peak_hour:02d}:00")
if "Season" in df_processed.columns:
    peak_season = df_processed["Season"].value_counts().idxmax()
    print(f"  Peak season  : {peak_season}")
if "Is_Weekend" in df_processed.columns:
    weekend_pct = df_processed["Is_Weekend"].mean() * 100
    print(f"  Weekend crimes: {weekend_pct:.1f}%")
if "Arrest" in df_processed.columns:
    arrest_pct = df_processed["Arrest"].mean() * 100
    print(f"  Arrest rate  : {arrest_pct:.1f}%")

print("\nNEXT STEPS:")
print("  → Section 3: EDA (notebooks/02_eda.py)")
print("  → Section 4: Feature Engineering (notebooks/03_feature_engineering.py)")
print("=" * 60)
