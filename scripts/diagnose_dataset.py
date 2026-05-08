# filename: scripts/diagnose_dataset.py
# purpose:  Inspect the raw Chicago crime CSV before writing any preprocessing code
# version:  1.0
#
# HOW TO RUN:
#   python scripts/diagnose_dataset.py
#
# Share the full output with Claude before proceeding to preprocessing.
# This tells us: column names, dtypes, null counts, date format, and
# what Arrest/Domestic/Primary_Type values actually look like.

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Find the first CSV in data/raw/
csv_files = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("*.csv.gz"))
if not csv_files:
    print("No CSV file found in data/raw/")
    print("Download from: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2")
    sys.exit(1)

RAW_CSV = csv_files[0]
print(f"File: {RAW_CSV.name}")
print(f"Size: {RAW_CSV.stat().st_size / (1024*1024):.1f} MB")

# Load only first 5000 rows for fast diagnosis
print("\nLoading first 5000 rows...")
df = pd.read_csv(RAW_CSV, nrows=5000, low_memory=False)

print("\n" + "=" * 60)
print("1. EXACT COLUMN NAMES (copy these exactly)")
print("=" * 60)
for i, col in enumerate(df.columns):
    print(f"  [{i:02d}] '{col}'")

print("\n" + "=" * 60)
print("2. SHAPE")
print("=" * 60)
print(f"  Rows (sample): {df.shape[0]}")
print(f"  Columns      : {df.shape[1]}")

print("\n" + "=" * 60)
print("3. DTYPES")
print("=" * 60)
for col, dtype in df.dtypes.items():
    print(f"  {col:35}: {dtype}")

print("\n" + "=" * 60)
print("4. NULL COUNTS (in 5000 rows)")
print("=" * 60)
null_counts = df.isnull().sum()
for col, cnt in null_counts.items():
    marker = "  <<< HAS NULLS" if cnt > 0 else ""
    print(f"  {col:35}: {cnt:5,}{marker}")

print("\n" + "=" * 60)
print("5. DATE COLUMN — first 5 raw values")
print("=" * 60)
date_col = None
for col in df.columns:
    if 'date' in col.lower():
        date_col = col
        break
if date_col:
    print(f"  Column name: '{date_col}'")
    for val in df[date_col].head(5):
        print(f"  '{val}'")
else:
    print("  No date column found!")

print("\n" + "=" * 60)
print("6. ARREST COLUMN — unique values")
print("=" * 60)
arrest_col = None
for col in df.columns:
    if 'arrest' in col.lower():
        arrest_col = col
        break
if arrest_col:
    print(f"  Column name : '{arrest_col}'")
    print(f"  dtype       : {df[arrest_col].dtype}")
    print(f"  Unique vals : {df[arrest_col].unique().tolist()}")
    print(f"  Value counts:\n{df[arrest_col].value_counts().to_string()}")
else:
    print("  No Arrest column found!")

print("\n" + "=" * 60)
print("7. DOMESTIC COLUMN — unique values")
print("=" * 60)
dom_col = None
for col in df.columns:
    if 'domestic' in col.lower():
        dom_col = col
        break
if dom_col:
    print(f"  Column name : '{dom_col}'")
    print(f"  dtype       : {df[dom_col].dtype}")
    print(f"  Unique vals : {df[dom_col].unique().tolist()}")
else:
    print("  No Domestic column found!")

print("\n" + "=" * 60)
print("8. PRIMARY TYPE — all unique values")
print("=" * 60)
for col in df.columns:
    if 'primary' in col.lower() or 'type' in col.lower():
        print(f"  Column name : '{col}'")
        print(f"  dtype       : {df[col].dtype}")
        print(f"  Unique count: {df[col].nunique()}")
        print("  Values:")
        for v in sorted(df[col].dropna().unique()):
            print(f"    '{v}'")
        break

print("\n" + "=" * 60)
print("9. LATITUDE / LONGITUDE — sample values and range")
print("=" * 60)
for col in df.columns:
    if 'latitude' in col.lower() or 'longitude' in col.lower():
        print(f"  '{col}': dtype={df[col].dtype}, "
              f"min={df[col].min()}, max={df[col].max()}, "
              f"nulls={df[col].isnull().sum()}")

print("\n" + "=" * 60)
print("10. FIRST 3 ROWS (full)")
print("=" * 60)
print(df.head(3).to_string())

print("\n" + "=" * 60)
print("DONE — share this full output with Claude")
print("=" * 60)
