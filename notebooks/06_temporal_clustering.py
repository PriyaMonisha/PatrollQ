# filename: notebooks/06_temporal_clustering.py
# purpose:  Section 6 — Temporal crime pattern clustering (K-Means on cyclical features)
# version:  1.0

# ── Section 1: Setup ─────────────────────────────────────────
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns

from config import (
    ARTIFACTS_DIR,
    DOCS_FIGURES_DIR,
    FAST_MODE,
    FEATURES_CSV,
    KMEANS_TEMPORAL_N_CLUSTERS,
    MLFLOW_EXPERIMENT_TEMPORAL,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    TEMPORAL_FEATURES,
)
from src.models.temporal_clustering import (
    TEMPORAL_DIR,
    run_elbow_temporal,
    run_kmeans_temporal,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# FAST_MODE: use dev feature CSV if it exists, else fall back to production
_dev_csv = Path(str(FEATURES_CSV).replace(".csv.gz", "_dev.csv.gz"))
FEATURES_CSV_ACTIVE = _dev_csv if (FAST_MODE and _dev_csv.exists()) else FEATURES_CSV

print("=" * 60)
print("  PATROLIQ - Section 6: Temporal Clustering")
print("=" * 60)
print(f"FAST_MODE    : {FAST_MODE}")
print(f"RANDOM_STATE : {RANDOM_STATE}")
print(f"K (chosen)   : {KMEANS_TEMPORAL_N_CLUSTERS}")
print(f"Loading from : {FEATURES_CSV_ACTIVE.name}")


# ── Section 2: Load data ─────────────────────────────────────
print("\nLoading feature-engineered data...")
assert FEATURES_CSV_ACTIVE.exists(), (
    f"Feature CSV not found: {FEATURES_CSV_ACTIVE}\n"
    "Run notebooks/04_feature_engineering.py first."
)
df = pd.read_csv(FEATURES_CSV_ACTIVE)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Pre-flight: assert all temporal features + Hour column present
missing = [f for f in TEMPORAL_FEATURES if f not in df.columns]
assert not missing, f"Missing temporal features in CSV: {missing}"
assert "Hour" in df.columns, "Hour column missing — needed for heatmap"
assert "case_number" in df.columns, "case_number missing"

# Filter to only needed columns, then define feature matrix X
df = df[["case_number", "Hour"] + TEMPORAL_FEATURES]
X = df[TEMPORAL_FEATURES].copy()  # feature matrix used in all model calls

print(f"Temporal features : {TEMPORAL_FEATURES}")
print(f"Feature matrix    : {X.shape}")


# ── Section 3: MLflow setup ───────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_TEMPORAL)
print(f"\nMLflow tracking  : {MLFLOW_TRACKING_URI}")
print(f"Experiment       : {MLFLOW_EXPERIMENT_TEMPORAL}")


# ── Section 4: Elbow method ───────────────────────────────────
print("\n[1/2] Running temporal elbow method...")

with mlflow.start_run(run_name="temporal_elbow"):
    mlflow.log_param("fast_mode", FAST_MODE)
    mlflow.log_param("n_samples", len(X))
    mlflow.log_param("n_features", X.shape[1])

    elbow = run_elbow_temporal(X, chosen_k=KMEANS_TEMPORAL_N_CLUSTERS)

    for k, sil in zip(elbow["k_values"], elbow["silhouettes"]):
        mlflow.log_metric(f"silhouette_k{k}", round(sil, 6))
    for k, inertia in zip(elbow["k_values"], elbow["inertias"]):
        mlflow.log_metric(f"inertia_k{k}", round(inertia, 2))
    mlflow.log_metric("best_k_by_silhouette", elbow["best_k_by_silhouette"])
    mlflow.log_artifact(str(TEMPORAL_DIR / "elbow_results.json"))

    # Elbow chart with best-K and chosen-K annotations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(elbow["k_values"], elbow["inertias"], "bo-")
    ax1.set_xlabel("K"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Method")

    ax2.plot(elbow["k_values"], elbow["silhouettes"], "ro-")
    ax2.axvline(elbow["best_k_by_silhouette"], color="orange", linestyle="--",
                label=f"Best K={elbow['best_k_by_silhouette']} (silhouette)")
    ax2.axvline(elbow["chosen_k"], color="red", linestyle="--",
                label=f"Chosen K={elbow['chosen_k']} (domain)")
    ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette"); ax2.set_title("Silhouette Score")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig_path = DOCS_FIGURES_DIR / "06_temporal_elbow.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(fig_path))
    print(f"  Best K (silhouette) : {elbow['best_k_by_silhouette']}")
    print(f"  Chosen K (domain)   : {elbow['chosen_k']}")

    if elbow["best_k_by_silhouette"] != elbow["chosen_k"]:
        print(
            f"  NOTE: Using K={elbow['chosen_k']} (domain-driven, ~4 daily activity windows) "
            f"vs algorithmic best K={elbow['best_k_by_silhouette']}"
        )


# ── Section 5: K-Means temporal clustering ───────────────────
print("\n[2/2] Running K-Means temporal clustering...")

with mlflow.start_run(run_name="temporal_kmeans"):
    mlflow.log_params({
        "k": KMEANS_TEMPORAL_N_CLUSTERS,
        "n_init": 10,
        "random_state": RANDOM_STATE,
        "n_samples": len(X),
        "features": str(TEMPORAL_FEATURES),
    })

    result = run_kmeans_temporal(X, k=KMEANS_TEMPORAL_N_CLUSTERS)
    # Filter to numeric values only — MLflow rejects string fields
    numeric_metrics = {k: v for k, v in result["metrics"].items() if isinstance(v, (int, float))}
    mlflow.log_metrics(numeric_metrics)
    mlflow.sklearn.log_model(result["model"], "temporal_kmeans_model")
    mlflow.log_artifact(str(TEMPORAL_DIR / "kmeans_metrics.json"))

    print(f"  Silhouette    : {result['metrics']['silhouette']:.4f}")
    print(f"  Davies-Bouldin: {result['metrics']['davies_bouldin']:.4f}")

# ── Section 5b: Label quality check ──────────────────────────
labels = result["labels"]
assert len(labels) == len(df), "Label count mismatch"
assert pd.Series(labels).nunique() == KMEANS_TEMPORAL_N_CLUSTERS, "Wrong cluster count"
assert pd.Series(labels).isna().sum() == 0, "Null labels found"

label_counts = pd.Series(labels).value_counts().sort_index()
assert label_counts.min() >= 10, f"Degenerate cluster: {label_counts.min()} rows"

print(f"\nCluster distribution:")
for cluster, count in label_counts.items():
    pct = count / len(labels) * 100
    print(f"  Cluster {cluster}: {count:,} crimes ({pct:.1f}%)")

# ── Section 5c: Merge labels back (REQUIRED before heatmap) ──
df["temporal_cluster"] = labels


# ── Section 6: Temporal pattern heatmap ──────────────────────
print("\nGenerating temporal heatmap...")

heatmap_data = (
    df[["temporal_cluster", "Hour"]]
    .groupby(["temporal_cluster", "Hour"])
    .size()
    .unstack(fill_value=0)
)
# Normalize by cluster size — shows WHEN each cluster is active, not cluster dominance
heatmap_norm = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)

fig, axes = plt.subplots(1, 2, figsize=(18, 5))
fig.suptitle("Temporal Crime Cluster Patterns", y=1.01)

# Raw counts heatmap
sns.heatmap(
    heatmap_data, cmap="YlOrRd", ax=axes[0],
    cbar_kws={"label": "Crime Count"},
    linewidths=0.3, linecolor="#f0f0f0",
)
axes[0].set_title("Crime Count per Cluster x Hour")
axes[0].set_xlabel("Hour of Day")
axes[0].set_ylabel("Temporal Cluster")

# Normalized heatmap
sns.heatmap(
    heatmap_norm, cmap="YlOrRd", ax=axes[1],
    cbar_kws={"label": "Fraction of Cluster Crimes"},
    linewidths=0.3, linecolor="#f0f0f0",
    vmin=0,
)
axes[1].set_title("Normalized Activity Pattern\n(fraction of cluster's crimes by hour)")
axes[1].set_xlabel("Hour of Day")
axes[1].set_ylabel("Temporal Cluster")

plt.tight_layout()
fig_path = DOCS_FIGURES_DIR / "06_temporal_heatmap.png"
fig_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Heatmap saved: {fig_path.name}")


# ── Section 7: Results table + final summary ─────────────────
metrics = result["metrics"]

print(f"\n{'=' * 50}")
print(f"  SECTION 6: TEMPORAL CLUSTERING COMPLETE")
print(f"{'=' * 50}")
print(f"\n{'Metric':<28} {'Value':>12}")
print(f"{'-' * 42}")
print(f"{'K (chosen)':<28} {KMEANS_TEMPORAL_N_CLUSTERS:>12}")
print(f"{'Best K (silhouette)':<28} {elbow['best_k_by_silhouette']:>12}")
print(f"{'Silhouette Score':<28} {metrics['silhouette']:>12.4f}")
print(f"{'Davies-Bouldin Score':<28} {metrics['davies_bouldin']:>12.4f}")
print(f"{'Inertia':<28} {metrics['inertia']:>12.0f}")
print(f"{'Rows Clustered':<28} {len(labels):>12,}")
print(f"\nArtifacts saved to : {TEMPORAL_DIR}")
print(f"MLflow experiment  : {MLFLOW_EXPERIMENT_TEMPORAL}")
print(f"Figures saved to   : {DOCS_FIGURES_DIR}")
print(f"""
ARTIFACTS:
  {TEMPORAL_DIR}/kmeans_labels.csv
  {TEMPORAL_DIR}/kmeans_metrics.json
  {TEMPORAL_DIR}/kmeans_model.pkl
  {TEMPORAL_DIR}/elbow_results.json
  docs/figures/06_temporal_elbow.png
  docs/figures/06_temporal_heatmap.png

NEXT: notebooks/07_dimensionality_reduction.py
""")
