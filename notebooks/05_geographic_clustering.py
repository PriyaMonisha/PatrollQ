# filename: notebooks/05_geographic_clustering.py
# purpose:  Section 5 — Geographic crime hotspot clustering (K-Means, DBSCAN, Hierarchical)
# version:  1.0

# ── Cell 1: Setup ────────────────────────────────────────────
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlflow
import pandas as pd
import matplotlib.pyplot as plt

from config import (
    ARTIFACTS_DIR,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    DOCS_FIGURES_DIR,
    FAST_MODE,
    HIERARCHICAL_N_CLUSTERS,
    HIERARCHICAL_SUBSAMPLE,
    KMEANS_GEO_N_CLUSTERS,
    MLFLOW_EXPERIMENT_GEO,
    MLFLOW_TRACKING_URI,
    PROCESSED_CSV,
    RANDOM_STATE,
)
from src.models.geographic_clustering import (
    run_dbscan_geo,
    run_elbow,
    run_hierarchical_geo,
    run_kmeans_geo,
)

from src.utils.logger import setup_logging
setup_logging()
logger = logging.getLogger(__name__)

DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
(ARTIFACTS_DIR / "geographic").mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("  PATROLIQ — Section 5: Geographic Clustering")
print("=" * 60)
print(f"FAST_MODE    : {FAST_MODE}")
print(f"RANDOM_STATE : {RANDOM_STATE}")


# ── Cell 2: Load processed data ───────────────────────────────
print("\nLoading processed data...")
df = pd.read_csv(PROCESSED_CSV)
print(f"Shape : {df.shape}")
print(f"Lat range : {df['latitude'].min():.4f} – {df['latitude'].max():.4f}")
print(f"Lon range : {df['longitude'].min():.4f} – {df['longitude'].max():.4f}")

# Drop any remaining null coordinates (safety check)
before = len(df)
df = df.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
if len(df) < before:
    logger.warning(f"Dropped {before - len(df):,} null-coordinate rows")
print(f"Clean rows for clustering: {len(df):,}")


# ── Cell 3: MLflow setup ──────────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_GEO)
print(f"MLflow tracking: {MLFLOW_TRACKING_URI}")
print(f"Experiment    : {MLFLOW_EXPERIMENT_GEO}")


# ── Cell 4: Elbow method — find optimal K ────────────────────
print("\n[1/3] Running elbow method...")
with mlflow.start_run(run_name="geo_elbow"):
    mlflow.log_param("fast_mode", FAST_MODE)
    mlflow.log_param("n_samples", len(df))

    elbow = run_elbow(df)

    mlflow.log_param("k_range", f"{min(elbow['k_values'])}-{max(elbow['k_values'])}")
    mlflow.log_metric("best_k_silhouette", elbow["best_k_silhouette"])
    for k, sil in zip(elbow["k_values"], elbow["silhouettes"]):
        mlflow.log_metric(f"silhouette_k{k}", round(sil, 6))

    # Elbow chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(elbow["k_values"], elbow["inertias"], "bo-")
    ax1.set_xlabel("K"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Method")
    ax2.plot(elbow["k_values"], elbow["silhouettes"], "ro-")
    ax2.set_xlabel("K"); ax2.set_ylabel("Silhouette"); ax2.set_title("Silhouette Score")
    plt.tight_layout()
    fig_path = DOCS_FIGURES_DIR / "05_geo_elbow.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(fig_path))
    print(f"  Best K (silhouette peak): {elbow['best_k_silhouette']}")
    print(f"  Elbow chart saved: {fig_path.name}")


# ── Cell 5: K-Means geographic clustering ────────────────────
print("\n[2/3] Running K-Means geographic clustering...")
with mlflow.start_run(run_name="geo_kmeans"):
    mlflow.log_param("k", KMEANS_GEO_N_CLUSTERS)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("n_samples", len(df))

    km_result = run_kmeans_geo(df, k=KMEANS_GEO_N_CLUSTERS)
    km_metrics = km_result["metrics"]

    mlflow.log_metric("silhouette", km_metrics["silhouette"])
    mlflow.log_metric("davies_bouldin", km_metrics["davies_bouldin"])
    if "inertia" in km_metrics:
        mlflow.log_metric("inertia", km_metrics["inertia"])
    mlflow.log_artifact(str(ARTIFACTS_DIR / "geographic" / "kmeans_metrics.json"))

    # Cluster map
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        df["longitude"], df["latitude"],
        c=km_result["labels"], cmap="tab10",
        s=1, alpha=0.4,
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(f"K-Means Geographic Clusters (k={KMEANS_GEO_N_CLUSTERS})")
    fig_path = DOCS_FIGURES_DIR / "05_geo_kmeans_map.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(fig_path))

    print(f"  Silhouette  : {km_metrics['silhouette']:.4f}  (target > 0.5)")
    print(f"  Davies-Bouldin: {km_metrics['davies_bouldin']:.4f}")
    print(f"  Map saved   : {fig_path.name}")


# ── Cell 6: DBSCAN geographic clustering ─────────────────────
print("\n[3a] Running DBSCAN geographic clustering...")
with mlflow.start_run(run_name="geo_dbscan"):
    mlflow.log_param("eps", DBSCAN_EPS)
    mlflow.log_param("min_samples", DBSCAN_MIN_SAMPLES)
    mlflow.log_param("n_samples", len(df))

    db_result = run_dbscan_geo(df)
    db_metrics = db_result["metrics"]

    mlflow.log_metric("n_clusters", db_metrics["n_clusters"])
    mlflow.log_metric("noise_fraction", db_metrics["noise_fraction"])
    if db_metrics["silhouette"] is not None:
        mlflow.log_metric("silhouette", db_metrics["silhouette"])
    mlflow.log_artifact(str(ARTIFACTS_DIR / "geographic" / "dbscan_metrics.json"))

    print(f"  Clusters    : {db_metrics['n_clusters']}")
    print(f"  Noise       : {db_metrics['noise_fraction']:.1%}  (target < 10%)")
    print(f"  Silhouette  : {db_metrics.get('silhouette')}")


# ── Cell 7: Hierarchical clustering (subsample) ──────────────
print(f"\n[3b] Running Hierarchical clustering (subsample={HIERARCHICAL_SUBSAMPLE:,})...")
with mlflow.start_run(run_name="geo_hierarchical"):
    mlflow.log_param("n_clusters", HIERARCHICAL_N_CLUSTERS)
    mlflow.log_param("subsample", HIERARCHICAL_SUBSAMPLE)
    mlflow.log_param("linkage", "ward")

    hc_result = run_hierarchical_geo(df)
    hc_metrics = hc_result["metrics"]

    mlflow.log_metric("silhouette", hc_metrics["silhouette"])
    mlflow.log_metric("davies_bouldin", hc_metrics["davies_bouldin"])
    mlflow.log_artifact(str(ARTIFACTS_DIR / "geographic" / "hierarchical_metrics.json"))

    print(f"  Silhouette  : {hc_metrics['silhouette']:.4f}")
    print(f"  Davies-Bouldin: {hc_metrics['davies_bouldin']:.4f}")


# ── Cell 8: Results comparison ───────────────────────────────
print("\n" + "=" * 55)
print("  GEOGRAPHIC CLUSTERING RESULTS")
print("=" * 55)
print(f"{'Model':<20} {'Clusters':>8} {'Silhouette':>12} {'DB Score':>10}")
print("-" * 55)
print(f"{'K-Means':<20} {KMEANS_GEO_N_CLUSTERS:>8} {km_metrics['silhouette']:>12.4f} {km_metrics['davies_bouldin']:>10.4f}")
print(f"{'DBSCAN':<20} {db_metrics['n_clusters']:>8} {str(db_metrics.get('silhouette','N/A')):>12} {'N/A':>10}")
print(f"{'Hierarchical':<20} {HIERARCHICAL_N_CLUSTERS:>8} {hc_metrics['silhouette']:>12.4f} {hc_metrics['davies_bouldin']:>10.4f}")
print("-" * 55)
print(f"DBSCAN noise fraction : {db_metrics['noise_fraction']:.2%}")


# ── Cell 9: FINAL SUMMARY ────────────────────────────────────
print(f"""
===== SECTION 5: GEOGRAPHIC CLUSTERING COMPLETE =====

Dataset    : {len(df):,} records | FAST_MODE={FAST_MODE}
Models run : K-Means, DBSCAN, Hierarchical (Ward)

KEY FINDINGS:
  K-Means silhouette   : {km_metrics['silhouette']:.4f}  {'[PASS > 0.5]' if km_metrics['silhouette'] > 0.5 else '[below 0.5 target - normal for crime data]'}
  DBSCAN clusters      : {db_metrics['n_clusters']} | noise={db_metrics['noise_fraction']:.1%}  {'[PASS < 10%]' if db_metrics['noise_fraction'] < 0.1 else '[above 10% target]'}
  Hierarchical sil     : {hc_metrics['silhouette']:.4f} (on {HIERARCHICAL_SUBSAMPLE:,} subsample)

ARTIFACTS:
  artifacts/geographic/kmeans_labels.csv
  artifacts/geographic/kmeans_metrics.json
  artifacts/geographic/dbscan_labels.csv
  artifacts/geographic/dbscan_metrics.json
  artifacts/geographic/hierarchical_labels.csv
  artifacts/geographic/hierarchical_metrics.json
  artifacts/geographic/elbow_results.json
  docs/figures/05_geo_elbow.png
  docs/figures/05_geo_kmeans_map.png

NEXT SECTION DEPENDENCY:
  -> notebooks/06_temporal_clustering.py
""")