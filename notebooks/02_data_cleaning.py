# filename: notebooks/02_data_cleaning.py
# purpose:  Section 2b — Data cleaning for PatrolIQ Chicago crime dataset
# version:  1.0
#
# WHAT THIS NOTEBOOK DOES:
#   Loads the raw sampled data, inspects it, cleans it step by step,
#   and saves a clean version ready for EDA and feature engineering.
#
# STATUS: SKELETON — implementation pending after diagnostic output
#   Run scripts/diagnose_dataset.py on the actual raw CSV first,
#   share the output, then fill in each cell below.
#
# INPUT : data/processed/chicago_crime_dev.csv.gz   (FAST_MODE)
#         data/processed/chicago_crime_500k.csv.gz  (production)
#
# OUTPUT: data/processed/chicago_crime_clean.csv.gz
# ================================================================


# ── Cell 1: Setup ────────────────────────────────────────────
# TODO: imports, logging, paths, FAST_MODE flag


# ── Cell 2: Load raw sampled data ────────────────────────────
# TODO: load chicago_crime_dev.csv.gz or chicago_crime_500k.csv.gz
#       print shape, dtypes, memory usage


# ── Cell 3: Inspect before cleaning ─────────────────────────
# TODO: df.info(), df.head(5)
#       df.isnull().sum() — which columns have nulls and how many
#       df.dtypes — confirm actual types (string vs bool vs int etc.)
#       df['Date'].head(5) — see actual date format
#       df['Arrest'].unique() — see actual values (True/False vs string)
#       df['Primary Type'].value_counts() — all 33 crime types
#       df['Latitude'].describe() — check coordinate range


# ── Cell 4: Drop exact duplicate rows ────────────────────────
# TODO: drop duplicates on Case Number (unique crime identifier)
#       log how many removed


# ── Cell 5: Handle null Latitude / Longitude ─────────────────
# TODO: drop rows where Latitude or Longitude is null
#       WHY: cannot be clustered geographically without coordinates
#       log how many rows dropped and what % of total


# ── Cell 6: Parse Date column ────────────────────────────────
# TODO: convert Date string → datetime
#       format depends on what Cell 3 shows
#       handle any rows where date fails to parse
#       log null count after parsing


# ── Cell 7: Extract temporal features ───────────────────────
# TODO: from the parsed Date column, extract:
#       Hour (0–23), Day_of_Week (0=Mon…6=Sun),
#       Month (1–12), Year (4-digit)
#       verify ranges after extraction


# ── Cell 8: Classify Season ─────────────────────────────────
# TODO: map Month → Season (Winter/Spring/Summer/Fall)
#       verify all 4 seasons appear in result


# ── Cell 9: Create Is_Weekend flag ──────────────────────────
# TODO: Is_Weekend = Day_of_Week in {5, 6}
#       log weekend vs weekday split


# ── Cell 10: Handle null Beat / District / Ward ──────────────
# TODO: impute with column mode (most common value)
#       depends on actual null counts seen in Cell 3
#       log how many nulls were filled per column


# ── Cell 11: Encode Primary_Type ─────────────────────────────
# TODO: LabelEncode Primary_Type → primary_type_code
#       save label encoder to models/label_encoders.pkl
#       keep original Primary_Type column for display


# ── Cell 12: Encode Location_Description ────────────────────
# TODO: LabelEncode Location_Description → location_desc_code
#       add to same label_encoders.pkl


# ── Cell 13: Fix Arrest and Domestic dtypes ──────────────────
# TODO: convert to bool — actual conversion depends on Cell 3
#       if strings like "true"/"false": map to True/False
#       if already bool: just confirm dtype
#       if "Y"/"N" or "1"/"0": handle accordingly


# ── Cell 14: Validate geographic bounds ─────────────────────
# TODO: drop rows outside Chicago bounds
#       lat: 41.6 – 42.0
#       lon: -87.9 – -87.5
#       log how many removed


# ── Cell 15: Memory optimization ────────────────────────────
# TODO: Latitude, Longitude → float32 (halves coordinate memory)
#       log memory before and after


# ── Cell 16: Final null check ────────────────────────────────
# TODO: df.isnull().sum().sum() — should be 0 for critical columns
#       print final shape and memory
#       print column list before saving


# ── Cell 17: Save cleaned data ───────────────────────────────
# TODO: save to data/processed/chicago_crime_clean.csv.gz
#       print file size
#       print: how many rows in vs out, % retained


# ── Cell 18: Section summary ─────────────────────────────────
# TODO: print summary of all cleaning steps and what changed
#       list all new columns added
#       print path to saved file
#       next step: notebooks/03_eda.py
