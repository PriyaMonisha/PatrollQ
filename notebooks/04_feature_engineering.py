# filename: notebooks/03_feature_engineering.py
# purpose:  Section 4 — Feature engineering for PatrolIQ crime clustering
# version:  1.0

# ── Cell 1: Setup ────────────────────────────────────────────
import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd

from config import (
    DATA_PROCESSED_DIR,
    DOCS_FIGURES_DIR,
    FAST_MODE,
    FEATURE_ENGINEER_PATH,
    FEATURES_CSV,
    GEO_FEATURES,
    MODELS_DIR,
    PROCESSED_CSV,
    RANDOM_STATE,
    TEMPORAL_FEATURES,
    FULL_FEATURES,
)
from src.features.engineer import CrimeFeatureEngineer, NEW_FEATURES

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PatrolIQ — Section 4: Feature Engineering")
print("=" * 60)
print(f"  FAST_MODE    : {FAST_MODE}")
print(f"  RANDOM_STATE : {RANDOM_STATE}")
print(f"  New features : {len(NEW_FEATURES)}")
print("=" * 60)


# ── Cell 2: Load preprocessed data ──────────────────────────
# Load from dev file in FAST_MODE, production file otherwise
if FAST_MODE:
    dev_csv = DATA_PROCESSED_DIR / "chicago_crime_dev.csv.gz"
    csv_path = dev_csv if dev_csv.exists() else PROCESSED_CSV
else:
    csv_path = PROCESSED_CSV

if not csv_path.exists():
    print(f"\n✗ Preprocessed data not found: {csv_path}")
    print("  Run notebooks/01_data_acquisition.py first.")
    sys.exit(1)

print(f"\nLoading preprocessed data: {csv_path.name}")
df = pd.read_csv(csv_path)

# Restore bool dtypes (CSV round-trip may convert bool → object)
for col in ['arrest', 'domestic', 'Is_Weekend']:
    if col in df.columns and df[col].dtype != bool:
        df[col] = df[col].astype(str).str.lower().map(
            {'true': True, 'false': False, '1': True, '0': False}
        ).fillna(False).astype(bool)

total = len(df)
print(f"✓ Loaded: {total:,} rows × {df.shape[1]} columns")
print(f"  Memory : {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")


# ── Cell 3: Quick pre-engineering check ─────────────────────
print("\n--- Required columns check ---")
required_for_fe = [
    'Hour', 'Day_of_Week', 'Month', 'Season',
    'latitude', 'longitude', 'primary_type',
    'Is_Weekend', 'primary_type_code', 'location_desc_code',
    'arrest', 'domestic'
]
for col in required_for_fe:
    status = '✓' if col in df.columns else '✗ MISSING'
    print(f"  {col:30}: {status}")


# ── Cell 4: Run feature engineering ─────────────────────────
print("\n--- Running CrimeFeatureEngineer ---")
fe = CrimeFeatureEngineer()
df_features = fe.fit_transform(df)

print(f"\nShape before : {df.shape}")
print(f"Shape after  : {df_features.shape}")
print(f"New columns  : {df_features.shape[1] - df.shape[1]}")


# ── Cell 5: Validate each new feature group ──────────────────
print("\n--- Feature Validation ---")

# Group 1: Cyclical encoding validation
print("\n[Group 1] Cyclical temporal encoding:")
cyclical_pairs = [
    ('Hour',        'hour_sin',  'hour_cos',  24),
    ('Day_of_Week', 'day_sin',   'day_cos',    7),
    ('Month',       'month_sin', 'month_cos', 12),
]
for raw_col, sin_col, cos_col, period in cyclical_pairs:
    # Verify: sin²+cos² = 1 for all rows (Pythagoras identity)
    identity_ok = np.allclose(
        df_features[sin_col]**2 + df_features[cos_col]**2,
        1.0, atol=1e-5
    )
    sin_range = f"[{df_features[sin_col].min():.3f}, {df_features[sin_col].max():.3f}]"
    cos_range = f"[{df_features[cos_col].min():.3f}, {df_features[cos_col].max():.3f}]"
    print(
        f"  {sin_col:12} {sin_range}  |  "
        f"{cos_col:12} {cos_range}  |  "
        f"sin²+cos²≡1: {'✓' if identity_ok else '✗ FAILED'}"
    )

# Check that Hour 0 and Hour 23 are close in cyclical space
h0   = df_features[df_features['Hour'] == 0][['hour_sin', 'hour_cos']].mean()
h23  = df_features[df_features['Hour'] == 23][['hour_sin', 'hour_cos']].mean()
dist = np.sqrt((h0['hour_sin'] - h23['hour_sin'])**2 + (h0['hour_cos'] - h23['hour_cos'])**2)
print(f"\n  Midnight continuity check (Hour 0 ↔ Hour 23):")
print(f"  Euclidean distance in sin/cos space: {dist:.4f}")
print(f"  Raw integer distance would be:       23")
print(f"  ✓ Cyclical encoding correctly wraps midnight" if dist < 1.0 else "  ✗ Check failed")

# Group 2: Geographic normalization validation
print("\n[Group 2] Geographic normalization:")
params = fe.normalization_params
print(f"  Fitted lat range: [{params['lat_min']:.4f}, {params['lat_max']:.4f}]")
print(f"  Fitted lon range: [{params['lon_min']:.4f}, {params['lon_max']:.4f}]")

lat_norm_ok = df_features['lat_norm'].between(0, 1).all()
lon_norm_ok = df_features['lon_norm'].between(0, 1).all()
print(f"  lat_norm in [0,1]: {'✓' if lat_norm_ok else '✗ FAILED'}")
print(f"  lon_norm in [0,1]: {'✓' if lon_norm_ok else '✗ FAILED'}")
print(f"  lat_norm range:    [{df_features['lat_norm'].min():.4f}, {df_features['lat_norm'].max():.4f}]")
print(f"  lon_norm range:    [{df_features['lon_norm'].min():.4f}, {df_features['lon_norm'].max():.4f}]")

# Group 3: Crime characteristics validation
print("\n[Group 3] Crime characteristics:")
sev_range = f"[{df_features['Crime_Severity_Score'].min()}, {df_features['Crime_Severity_Score'].max()}]"
sev_dist  = df_features['Crime_Severity_Score'].value_counts().sort_index().to_dict()
print(f"  Crime_Severity_Score range: {sev_range}  (expected [1,10])")
print(f"  Score distribution: {sev_dist}")

# Null check on all new features
print("\n[Null check] New features:")
any_null = False
for col in NEW_FEATURES:
    if col not in df_features.columns:
        print(f"  ✗ MISSING: {col}")
        any_null = True
        continue
    null_count = df_features[col].isnull().sum()
    if null_count > 0:
        print(f"  ✗ {col}: {null_count:,} nulls")
        any_null = True
    else:
        print(f"  ✓ {col}: no nulls")
if not any_null:
    print("  All new features: zero nulls ✓")


# ── Cell 6: Validate FULL_FEATURES set ──────────────────────
print("\n--- FULL_FEATURES validation ---")
print(f"GEO_FEATURES     : {GEO_FEATURES}")
print(f"TEMPORAL_FEATURES: {TEMPORAL_FEATURES}")
print(f"\nAll FULL_FEATURES present in df_features:")
all_present = True
for col in FULL_FEATURES:
    present = col in df_features.columns
    if not present:
        print(f"  ✗ MISSING: {col}")
        all_present = False
if all_present:
    print(f"  ✓ All {len(FULL_FEATURES)} features present")


# ── Cell 7: Feature correlation heatmap (multicollinearity check) ─
print("\n--- Generating feature correlation heatmap ---")

numeric_features = [f for f in FULL_FEATURES if f in df_features.columns]
# Cast bool columns to int for correlation
corr_df = df_features[numeric_features].copy()
for col in ['arrest', 'domestic', 'Is_Weekend']:
    if col in corr_df.columns:
        corr_df[col] = corr_df[col].astype(int)

corr_matrix = corr_df.corr()

fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(
    corr_matrix, annot=True, fmt='.2f',
    annot_kws={'size': 7}, cmap='coolwarm',
    center=0, vmin=-1, vmax=1,
    square=True,
    linewidths=0.3, linecolor='#ebebeb',
    cbar_kws={'shrink': 0.7, 'label': 'Pearson r'},
    ax=ax
)
ax.set_title(
    'FULL_FEATURES Correlation Matrix\n'
    '(Multicollinearity check before clustering)',
    fontsize=12, pad=12
)
ax.tick_params(axis='x', rotation=45, labelsize=8)
ax.tick_params(axis='y', rotation=0,  labelsize=8)
plt.tight_layout()
fig.savefig(DOCS_FIGURES_DIR / '03_feature_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved: 03_feature_correlation.png")

# Report highly correlated pairs (|r| > 0.85) — potential multicollinearity
print("\n  High-correlation pairs (|r| > 0.85):")
found_any = False
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.85:
            print(
                f"  {corr_matrix.columns[i]:25} ↔ "
                f"{corr_matrix.columns[j]:25}: r={r:.3f}"
            )
            found_any = True
if not found_any:
    print("  None — no multicollinearity issues ✓")


# ── Cell 8: Save fitted FeatureEngineer ─────────────────────
print("\n--- Saving CrimeFeatureEngineer object ---")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(fe, FEATURE_ENGINEER_PATH)
print(f"  ✓ Saved: {FEATURE_ENGINEER_PATH.name}")

# Save normalization params as JSON (human-readable)
params_path = MODELS_DIR / "feature_engineer_params.json"
with open(params_path, "w") as f:
    json.dump(fe.normalization_params, f, indent=2)
print(f"  ✓ Saved: {params_path.name}")

# Verify round-trip: reload and re-transform, check shapes match
fe_reloaded = joblib.load(FEATURE_ENGINEER_PATH)
df_check    = fe_reloaded.transform(df.head(100))
assert df_check.shape[1] == df_features.shape[1], "Round-trip shape mismatch!"
print("  ✓ Reload + transform round-trip: OK")


# ── Cell 9: Save feature-engineered dataset ─────────────────
print("\n--- Saving feature-engineered dataset ---")
if FAST_MODE:
    save_path = DATA_PROCESSED_DIR / "chicago_crime_features_dev.csv.gz"
    print(f"  FAST_MODE=True → saving to: {save_path.name}")
else:
    save_path = FEATURES_CSV
    print(f"  FAST_MODE=False → saving to: {save_path.name} (production)")

save_path.parent.mkdir(parents=True, exist_ok=True)
df_features.to_csv(save_path, index=False, compression='gzip')
size_mb = save_path.stat().st_size / (1024 * 1024)
print(f"  ✓ Saved: {save_path.name} | {len(df_features):,} rows | {size_mb:.1f} MB")


# ── Cell 10: Section summary ─────────────────────────────────
print("\n" + "=" * 60)
print("  SECTION 4 SUMMARY")
print("=" * 60)
print(f"  Input rows       : {total:,}")
print(f"  Input columns    : {df.shape[1]}")
print(f"  Output columns   : {df_features.shape[1]}")
print(f"  New features     : {df_features.shape[1] - df.shape[1]}")
print(f"  FULL_FEATURES    : {len(FULL_FEATURES)} features ready for clustering")
print(f"  Engineer saved   : {FEATURE_ENGINEER_PATH.name}")
print(f"  Features CSV     : {save_path.name}")
print("=" * 60)

print("\nNEW FEATURES SUMMARY:")
print(f"\n  Group 1 — Cyclical Temporal (6 features):")
for col in ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']:
    print(f"    {col:15}: float32, range [{df_features[col].min():.3f}, {df_features[col].max():.3f}]")

print(f"\n  Group 2 — Geographic Normalization (2 features):")
for col in ['lat_norm', 'lon_norm']:
    print(f"    {col:15}: float32, range [{df_features[col].min():.4f}, {df_features[col].max():.4f}]")

print(f"\n  Group 3 — Crime Characteristics (1 feature):")
print(f"    Crime_Severity_Score : int8, "
      f"mean={df_features['Crime_Severity_Score'].mean():.2f}, "
      f"range [{df_features['Crime_Severity_Score'].min()}, {df_features['Crime_Severity_Score'].max()}]")

print(f"\nFEATURE SETS READY:")
print(f"  GEO_FEATURES      = {GEO_FEATURES}")
print(f"  TEMPORAL_FEATURES = {TEMPORAL_FEATURES}")
print(f"  FULL_FEATURES     = {len(FULL_FEATURES)} columns")

print("\nNEXT STEPS:")
print("  → Section 5: Geographic Clustering (notebooks/04_geographic_clustering.py)")
print("  → Section 6: Temporal Clustering  (notebooks/05_temporal_clustering.py)")
print("=" * 60)
