# filename: config.py
# purpose:  Central configuration — all hyperparameters, paths, constants for PatrolIQ
# version:  1.0

from pathlib import Path

# ── Execution Mode ───────────────────────────────────────────
# FAST_MODE = True  → dev runs: smaller subsamples, skip t-SNE, K elbow range 2-5 only
# FAST_MODE = False → production: full 500K pipeline, complete elbow sweep, full t-SNE
# Rule: always develop with FAST_MODE=True; flip to False only for final artifact generation
# Lesson learned from EMI project: define this BEFORE writing any training script (TR-02)
FAST_MODE = True

# ── Reproducibility ──────────────────────────────────────────
RANDOM_STATE = 42

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"
DOCS_FIGURES_DIR = BASE_DIR / "docs" / "figures"

PROCESSED_CSV = DATA_PROCESSED_DIR / "chicago_crime_500k.csv.gz"
FEATURES_CSV = DATA_PROCESSED_DIR / "chicago_crime_features.csv.gz"
SAMPLE_METADATA = DATA_PROCESSED_DIR / "sample_metadata.json"
LABEL_ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"
FEATURE_ENGINEER_PATH = MODELS_DIR / "feature_engineer.pkl"

# ── Data ─────────────────────────────────────────────────────
SAMPLE_SIZE = 500_000
FAST_SAMPLE_SIZE = 50_000   # for FAST_MODE=True development runs

# Expected columns from Chicago Data Portal CSV
EXPECTED_COLUMNS = [
    "ID", "Case Number", "Date", "Block", "IUCR", "Primary Type",
    "Description", "Location Description", "Arrest", "Domestic",
    "Beat", "District", "Ward", "Community Area", "FBI Code",
    "X Coordinate", "Y Coordinate", "Year", "Updated On",
    "Latitude", "Longitude", "Location"
]

# Chicago geographic bounds (for validation)
LAT_MIN, LAT_MAX = 41.6, 42.0
LON_MIN, LON_MAX = -87.9, -87.5

# ── Feature Engineering ──────────────────────────────────────
GEO_FEATURES = ["lat_norm", "lon_norm"]

TEMPORAL_FEATURES = [
    "hour_sin", "hour_cos",
    "day_sin", "day_cos",
    "month_sin", "month_cos",
    "Is_Weekend",
]

FULL_FEATURES = GEO_FEATURES + TEMPORAL_FEATURES + [
    "Crime_Severity_Score",
    "primary_type_code",
    "location_desc_code",
    "arrest",    # lowercase — normalized by loader, cast to bool in preprocessor
    "domestic",  # lowercase — normalized by loader, cast to bool in preprocessor
]

# Crime severity scores (higher = more severe)
SEVERITY_SCORES: dict[str, int] = {
    "HOMICIDE": 10,
    "KIDNAPPING": 9,
    "CRIM SEXUAL ASSAULT": 9,
    "ARSON": 8,
    "ROBBERY": 8,
    "WEAPONS VIOLATION": 7,
    "BATTERY": 7,
    "ASSAULT": 6,
    "BURGLARY": 6,
    "MOTOR VEHICLE THEFT": 5,
    "STALKING": 5,
    "INTIMIDATION": 5,
    "THEFT": 4,
    "NARCOTICS": 4,
    "DECEPTIVE PRACTICE": 4,
    "CRIMINAL DAMAGE": 3,
    "CRIMINAL TRESPASS": 3,
    "OFFENSE INVOLVING CHILDREN": 7,
    "SEX OFFENSE": 8,
    "PUBLIC PEACE VIOLATION": 3,
    "LIQUOR LAW VIOLATION": 2,
    "GAMBLING": 2,
    "OBSCENITY": 2,
}
SEVERITY_DEFAULT = 2  # for crime types not in lookup

# ── Geographic Clustering ────────────────────────────────────
KMEANS_GEO_K_RANGE = range(2, 16)     # test K = 2 to 15 in elbow method
KMEANS_GEO_N_CLUSTERS = 8             # override after running elbow method
DBSCAN_EPS = 0.008                     # ~800m at Chicago latitude ~42°N
# Derivation: 1 degree lat ≈ 111km at equator; at 42°N, 1° ≈ 82km; 0.008° ≈ 656m
DBSCAN_MIN_SAMPLES = 100              # appropriate density for 500K records
HIERARCHICAL_N_CLUSTERS = 8           # same K as K-Means for fair comparison
HIERARCHICAL_SUBSAMPLE = 10_000       # Ward linkage on 500K needs ~200GB RAM

# ── Temporal Clustering ──────────────────────────────────────
KMEANS_TEMPORAL_K_RANGE = range(2, 9)  # test K = 2 to 8
KMEANS_TEMPORAL_N_CLUSTERS = 4        # override after running

# ── Dimensionality Reduction ─────────────────────────────────
PCA_N_COMPONENTS = 3
PCA_VARIANCE_TARGET = 0.70             # target: explain ≥70% variance
PCA_SCATTER_SAMPLE = 50_000           # sample for 2D scatter plot

TSNE_SUBSAMPLE = 50_000               # t-SNE is O(n²) — must subsample
TSNE_PCA_PREPROCESS_N = 50            # PCA → 50 components before t-SNE
TSNE_N_COMPONENTS = 2
TSNE_PERPLEXITY = 30
TSNE_N_ITER = 1000

# ── MLflow ───────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
MLFLOW_EXPERIMENT_GEO = "patroliq_geographic_clustering"
MLFLOW_EXPERIMENT_TEMPORAL = "patroliq_temporal_clustering"
MLFLOW_EXPERIMENT_DIM = "patroliq_dimensionality_reduction"
