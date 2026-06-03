# filename: scripts/run_full_pipeline.py
# purpose:  Run the full PatrolIQ training pipeline in order
# version:  1.0

import os
import subprocess
import sys
import time
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

# ── Pipeline definition ───────────────────────────────────────
# Skips: 01_data_acquisition.py (manual download step)
#        03_eda.py              (analysis notebook — not part of training)
PIPELINE = [
    "notebooks/00_data_cleaning.py",         # load → sample → preprocess → save
    "notebooks/04_feature_engineering.py",   # engineer features → save feature CSV
    "notebooks/05_geographic_clustering.py", # K-Means, DBSCAN, Hierarchical + MLflow
    "notebooks/06_temporal_clustering.py",   # Temporal K-Means k=4 + MLflow
    "notebooks/07_dimensionality_reduction.py",  # PCA + t-SNE + MLflow
    "notebooks/08_mlflow_experiments.py",    # export runs to JSON + register model
]


def save_reference_snapshot() -> None:
    """
    Save a sample of the features CSV as the drift monitoring reference.
    Called after feature engineering (step 2) to capture the training distribution.
    The reference is loaded by src/monitoring/drift.py at inference time.
    """
    import pandas as pd
    from config import ARTIFACTS_DIR, DATA_PROCESSED_DIR, RANDOM_STATE

    candidates = [
        DATA_PROCESSED_DIR / "chicago_crime_features_dev.csv.gz",
        DATA_PROCESSED_DIR / "chicago_crime_features.csv.gz",
    ]
    src = next((p for p in candidates if p.exists()), None)
    if src is None:
        logger.warning("Reference snapshot skipped — features CSV not found yet")
        return

    ref_cols = ["latitude", "longitude", "Hour", "arrest", "primary_type"]
    df = pd.read_csv(src)
    df.columns = df.columns.str.lower()
    available = [c for c in [col.lower() for col in ref_cols] if c in df.columns]
    if len(available) < 3:
        logger.warning(f"Only {len(available)} columns available — snapshot may be limited")

    sample = df[available].sample(n=min(10_000, len(df)), random_state=RANDOM_STATE)
    # Restore Hour capitalization expected by drift.py
    if "hour" in sample.columns:
        sample = sample.rename(columns={"hour": "Hour"})

    ref_path = ARTIFACTS_DIR / "reference_distribution.parquet"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(ref_path, index=False)
    logger.info(
        f"Reference snapshot saved: {len(sample)} rows, "
        f"{len(available)} features → {ref_path}"
    )


def run_pipeline() -> None:
    print("=" * 60)
    print("  PATROLIQ — Full Training Pipeline")
    print("=" * 60)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Steps        : {len(PIPELINE)}")
    print()

    total_start = time.time()

    for i, script in enumerate(PIPELINE, 1):
        script_path = PROJECT_ROOT / script
        if not script_path.exists():
            print(f"[{i}/{len(PIPELINE)}] SKIP: {script} (file not found)")
            continue

        print(f"[{i}/{len(PIPELINE)}] RUNNING: {script}")
        print("-" * 60)
        step_start = time.time()

        try:
            subprocess.run(
                [sys.executable, str(script_path)],  # str() — Windows Path safety
                check=True,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            elapsed = time.time() - step_start
            print(f"[{i}/{len(PIPELINE)}] DONE: {script} ({elapsed:.1f}s)")
        except subprocess.CalledProcessError as e:
            print(f"\n[{i}/{len(PIPELINE)}] FAILED: {script} (exit code {e.returncode})")
            print(f"  Fix {script} and re-run the pipeline.")
            sys.exit(1)

        # After feature engineering, save reference snapshot for drift monitoring
        if "04_feature_engineering" in script:
            logger.info("Saving drift monitoring reference snapshot...")
            save_reference_snapshot()

        print()

    total_elapsed = time.time() - total_start
    print("=" * 60)
    print(f"  PIPELINE COMPLETE ({total_elapsed:.1f}s total)")
    print("=" * 60)
    print("\nOutputs:")
    print("  data/processed/chicago_crime_500k.csv.gz")
    print("  data/processed/chicago_crime_features_dev.csv.gz")
    print("  artifacts/geographic/  (labels + metrics)")
    print("  artifacts/temporal/    (labels + metrics + model)")
    print("  artifacts/dimensionality/ (PCA + t-SNE)")
    print("  artifacts/mlflow_exports/ (all_runs.json + best_models.json)")
    print("  mlruns/mlflow.db        (MLflow tracking)")


if __name__ == "__main__":
    run_pipeline()
