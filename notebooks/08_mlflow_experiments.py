# filename: notebooks/08_mlflow_experiments.py
# purpose:  Section 8 — Export all MLflow runs to JSON + register temporal model
# version:  1.0

# ── Section 1: Setup ─────────────────────────────────────────
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import pandas as pd

from config import (
    MLFLOW_EXPERIMENT_DIM,
    MLFLOW_EXPERIMENT_GEO,
    MLFLOW_EXPERIMENT_TEMPORAL,
    MLFLOW_EXPORTS_DIR,
    MLFLOW_TRACKING_URI,
    PCA_N_COMPONENTS,
)
from src.utils.helpers import NumpyEncoder, run_row_to_dict, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

MLFLOW_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("=" * 60)
print("  PATROLIQ - Section 8: MLflow Experiments Export")
print("=" * 60)
print(f"Tracking URI  : {MLFLOW_TRACKING_URI}")
print(f"Exports dir   : {MLFLOW_EXPORTS_DIR}")


# ── Section 2: Safe query helper + collect all runs ───────────
def get_runs(exp_name: str) -> pd.DataFrame:
    """Get all runs for an experiment. Uses experiment_ids — safe across MLflow versions."""
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp is None:
        raise ValueError(
            f"Experiment not found: {exp_name}\n"
            "Run the corresponding training notebook first."
        )
    return mlflow.search_runs(experiment_ids=[exp.experiment_id])


print("\nQuerying all experiments...")
geo_runs  = get_runs(MLFLOW_EXPERIMENT_GEO)
temp_runs = get_runs(MLFLOW_EXPERIMENT_TEMPORAL)
dim_runs  = get_runs(MLFLOW_EXPERIMENT_DIM)

total_runs = len(geo_runs) + len(temp_runs) + len(dim_runs)
print(f"  Geographic    : {len(geo_runs)} runs")
print(f"  Temporal      : {len(temp_runs)} runs")
print(f"  Dimensionality: {len(dim_runs)} runs")
print(f"  Total         : {total_runs} runs (GUVI target >= 6: {'PASS' if total_runs >= 6 else 'FAIL'})")

# Build all_runs list with clean keys
all_runs = []
for _, row in geo_runs.iterrows():
    all_runs.append(run_row_to_dict(row, MLFLOW_EXPERIMENT_GEO))
for _, row in temp_runs.iterrows():
    all_runs.append(run_row_to_dict(row, MLFLOW_EXPERIMENT_TEMPORAL))
for _, row in dim_runs.iterrows():
    all_runs.append(run_row_to_dict(row, MLFLOW_EXPERIMENT_DIM))

# Sort for consistent ordering
all_runs.sort(key=lambda r: (r["experiment"], r["run_name"]))

save_json(all_runs, MLFLOW_EXPORTS_DIR / "all_runs.json")
print(f"  Saved: all_runs.json ({len(all_runs)} entries)")


# ── Section 3: Best run per experiment ───────────────────────
print("\nSelecting best run per experiment...")

# Geo — filter elbow runs, guard all-NaN silhouette, sort by date
geo_final = geo_runs[
    geo_runs["tags.mlflow.runName"].isin(
        ["geo_kmeans", "geo_dbscan", "geo_hierarchical"]
    )
]
valid_geo = geo_final.dropna(subset=["metrics.silhouette"])
if valid_geo.empty:
    raise ValueError(
        "No geo runs with 'metrics.silhouette' found. "
        "Check 05_geographic_clustering.py logged this metric."
    )
best_geo = valid_geo.loc[valid_geo["metrics.silhouette"].idxmax()]
print(f"  Best geo      : {best_geo.get('tags.mlflow.runName')} "
      f"(sil={float(best_geo['metrics.silhouette']):.4f})")

# Temporal — filter, guard, sort by date
temp_final = temp_runs[temp_runs["tags.mlflow.runName"] == "temporal_kmeans"]
if temp_final.empty:
    raise ValueError("No temporal_kmeans run. Run 06_temporal_clustering.py first.")
best_temp = temp_final.sort_values("start_time", ascending=False).iloc[0]
print(f"  Best temporal : temporal_kmeans "
      f"(sil={float(best_temp['metrics.silhouette']):.4f})")

# PCA — filter, guard, sort by date
pca_runs = dim_runs[dim_runs["tags.mlflow.runName"] == "pca"]
if pca_runs.empty:
    raise ValueError("No pca run. Run 07_dimensionality_reduction.py first.")
best_pca = pca_runs.sort_values("start_time", ascending=False).iloc[0]
print(f"  Best dim      : pca "
      f"(cum_var={float(best_pca['metrics.cumulative_variance']):.4f})")

# Build flat best_models.json (params cast from strings)
best_models = {
    "geo_kmeans": {
        "run_id":     best_geo["run_id"],
        "silhouette": float(best_geo["metrics.silhouette"]),
        "n_clusters": int(best_geo["params.k"]),   # geo_kmeans logged "k" as param
        "experiment": MLFLOW_EXPERIMENT_GEO,
    },
    "temporal_kmeans": {
        "run_id":     best_temp["run_id"],
        "silhouette": float(best_temp["metrics.silhouette"]),
        "n_clusters": int(best_temp["params.k"]),  # temporal logged "k" as param
        "experiment": MLFLOW_EXPERIMENT_TEMPORAL,
    },
    "pca": {
        "run_id":              best_pca["run_id"],
        "cumulative_variance": float(best_pca["metrics.cumulative_variance"]),
        "n_components":        PCA_N_COMPONENTS,   # not a MLflow param — use config constant
        "experiment":          MLFLOW_EXPERIMENT_DIM,
    },
}
save_json(best_models, MLFLOW_EXPORTS_DIR / "best_models.json")
print(f"  Saved: best_models.json (3 entries)")


# ── Section 4: Comparison table ──────────────────────────────
print("\nModel comparison table:")

KEY_METRIC_MAP = {
    "geo_elbow":        "best_k_silhouette",      # geo logged this name (verified)
    "geo_kmeans":       "silhouette",
    "geo_dbscan":       "silhouette",
    "geo_hierarchical": "silhouette",
    "temporal_elbow":   "best_k_by_silhouette",   # temporal logged this name (verified)
    "temporal_kmeans":  "silhouette",
    "pca":              "cumulative_variance",
    "tsne":             "kl_divergence",
}

print(f"\n{'Run Name':<22} {'Experiment':<30} {'Metric':<25} {'Value':>10}")
print("-" * 90)

# Deduplicate: sort by start_time desc, keep most recent per (experiment, run_name)
seen = set()
for run in sorted(all_runs, key=lambda r: r.get("start_time", ""), reverse=True):
    key = (run["experiment"], run["run_name"])
    if key in seen:
        continue
    seen.add(key)
    run_name   = run["run_name"]
    metric_key = KEY_METRIC_MAP.get(run_name)
    value      = run["metrics"].get(metric_key, "N/A") if metric_key else "N/A"
    value_str  = f"{value:.4f}" if isinstance(value, float) else str(value)
    exp_short  = run["experiment"].replace("patroliq_", "")
    print(f"{run_name:<22} {exp_short:<30} {metric_key or 'N/A':<25} {value_str:>10}")


# ── Section 5: Register temporal_kmeans model ─────────────────
print("\nRegistering PatrolIQ_TemporalClustering model...")

km_runs = temp_runs[temp_runs["tags.mlflow.runName"] == "temporal_kmeans"]
if km_runs.empty:
    raise ValueError("No temporal_kmeans run. Run 06_temporal_clustering.py first.")

km_run    = km_runs.sort_values("start_time", ascending=False).iloc[0]
model_uri = f"runs:/{km_run['run_id']}/temporal_kmeans_model"
logger.info(f"Model URI: {model_uri}")

reg_version = None
try:
    reg_result = mlflow.register_model(model_uri, "PatrolIQ_TemporalClustering")
    reg_version = reg_result.version
    print(f"  Registered: PatrolIQ_TemporalClustering v{reg_version}")
except mlflow.exceptions.MlflowException as e:
    logger.warning(f"Model registration skipped: {e}")
    print(f"  Registration skipped (see warning above)")


# ── Section 6: Final summary ──────────────────────────────────
print(f"\n{'='*55}")
print(f"  SECTION 8: MLFLOW EXPERIMENTS EXPORT COMPLETE")
print(f"{'='*55}")
print(f"  Runs exported    : {len(all_runs)} / {total_runs} expected")
print(f"  GUVI target >=6  : {'PASS' if total_runs >= 6 else 'FAIL'}")
print(f"  Models registered: PatrolIQ_TemporalClustering v{reg_version or 'N/A'}")
print(f"  Exports saved    : {MLFLOW_EXPORTS_DIR}")
print(f"  Files:")
print(f"    all_runs.json    ({len(all_runs)} runs)")
print(f"    best_models.json (3 experiments)")
print(f"\n  NEXT: Section 9 - Streamlit App (streamlit_app.py)")
print(f"{'='*55}")
