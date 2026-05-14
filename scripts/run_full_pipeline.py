# filename: scripts/run_full_pipeline.py
# purpose:  Run the full PatrolIQ training pipeline in order
# version:  1.0

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

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
            )
            elapsed = time.time() - step_start
            print(f"[{i}/{len(PIPELINE)}] DONE: {script} ({elapsed:.1f}s)")
        except subprocess.CalledProcessError as e:
            print(f"\n[{i}/{len(PIPELINE)}] FAILED: {script} (exit code {e.returncode})")
            print(f"  Fix {script} and re-run the pipeline.")
            sys.exit(1)

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
