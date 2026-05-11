# filename: notebooks/07_dimensionality_reduction.py
# purpose:  Section 7 — Dimensionality reduction (PCA + t-SNE) on FULL_FEATURES
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
    MLFLOW_EXPERIMENT_DIM,
    MLFLOW_TRACKING_URI,
    PCA_VARIANCE_TARGET,
    RANDOM_STATE,
    REDUCTION_FEATURES,
    TSNE_N_ITER,
    TSNE_PERPLEXITY,
    TSNE_SUBSAMPLE,
)
from src.models.dimensionality_reduction import DIM_DIR, run_pca, run_tsne

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DOCS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

_dev_csv = Path(str(FEATURES_CSV).replace(".csv.gz", "_dev.csv.gz"))
FEATURES_CSV_ACTIVE = _dev_csv if (FAST_MODE and _dev_csv.exists()) else FEATURES_CSV

print("=" * 60)
print("  PATROLIQ - Section 7: Dimensionality Reduction")
print("=" * 60)
print(f"FAST_MODE    : {FAST_MODE}")
print(f"RANDOM_STATE : {RANDOM_STATE}")
print(f"Loading from : {FEATURES_CSV_ACTIVE.name}")


# ── Section 2: Load data ─────────────────────────────────────
print("\nLoading feature-engineered data...")
assert FEATURES_CSV_ACTIVE.exists(), (
    f"Feature CSV not found: {FEATURES_CSV_ACTIVE}\n"
    "Run notebooks/04_feature_engineering.py first."
)
df = pd.read_csv(FEATURES_CSV_ACTIVE)
print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

# Pre-flight: assert all reduction features present
missing = [f for f in REDUCTION_FEATURES if f not in df.columns]
assert not missing, f"Missing features: {missing}"
print(f"Features for reduction ({len(REDUCTION_FEATURES)}): {REDUCTION_FEATURES}")

# Also load cluster labels for coloring scatter plots
_geo_labels = ARTIFACTS_DIR / "geographic" / "kmeans_labels.csv"
_temp_labels = ARTIFACTS_DIR / "temporal" / "kmeans_labels.csv"

geo_labels = pd.read_csv(_geo_labels)["cluster"].values if _geo_labels.exists() else None
temp_labels = pd.read_csv(_temp_labels)["cluster"].values if _temp_labels.exists() else None

X = df[REDUCTION_FEATURES].copy()


# ── Section 3: MLflow setup ───────────────────────────────────
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_DIM)
print(f"\nMLflow experiment : {MLFLOW_EXPERIMENT_DIM}")


# ── Section 4: PCA ───────────────────────────────────────────
print("\n[1/2] Running PCA...")

with mlflow.start_run(run_name="pca"):
    mlflow.log_params({
        "fast_mode": FAST_MODE,
        "n_samples": len(X),
        "n_features": len(REDUCTION_FEATURES),
        "variance_target": PCA_VARIANCE_TARGET,
    })

    pca_result = run_pca(X)
    pca_metrics = pca_result["metrics"]

    mlflow.log_metric("cumulative_variance", pca_metrics["cumulative_variance"])
    for i, v in enumerate(pca_metrics["explained_variance_ratio"]):
        mlflow.log_metric(f"explained_variance_pc{i+1}", v)
    mlflow.log_metric("variance_target_met", int(pca_metrics["variance_target_met"]))
    mlflow.log_artifact(str(DIM_DIR / "pca_metrics.json"))

    print(f"  Explained variance: "
          + " | ".join(f"PC{i+1}={v:.1%}"
                       for i, v in enumerate(pca_metrics["explained_variance_ratio"]))
          + f" | Total={pca_metrics['cumulative_variance']:.1%}")
    target_status = "PASS" if pca_metrics["variance_target_met"] else f"below {PCA_VARIANCE_TARGET:.0%} target"
    print(f"  Variance target ({PCA_VARIANCE_TARGET:.0%}): {target_status}")

    # PCA scatter plot — PC1 vs PC2, colored by geographic cluster
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("PCA — 2D Projection of Crime Feature Space", y=1.01)

    pca_2d = pca_result["pca_2d"]

    # Panel 1: color by geo cluster (if available, else plain)
    if geo_labels is not None and len(geo_labels) >= len(pca_2d):
        sc = axes[0].scatter(
            pca_2d["PC1"], pca_2d["PC2"],
            c=geo_labels[:len(pca_2d)], cmap="tab10",
            s=2, alpha=0.4,
        )
        plt.colorbar(sc, ax=axes[0], label="Geo Cluster")
        axes[0].set_title("PC1 vs PC2 — colored by Geographic Cluster")
    else:
        axes[0].scatter(pca_2d["PC1"], pca_2d["PC2"], s=2, alpha=0.3, color="steelblue")
        axes[0].set_title("PC1 vs PC2")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    # Panel 2: Scree plot
    evr = pca_metrics["explained_variance_ratio"]
    cum_var = list(np.cumsum(evr))
    pc_labels = [f"PC{i+1}" for i in range(len(evr))]
    axes[1].bar(pc_labels, evr, color="steelblue", alpha=0.8, label="Individual")
    axes[1].plot(pc_labels, cum_var, "ro-", linewidth=2, label="Cumulative")
    axes[1].axhline(PCA_VARIANCE_TARGET, color="gray", linestyle="--",
                    label=f"Target {PCA_VARIANCE_TARGET:.0%}")
    axes[1].set_title("Scree Plot — Explained Variance per Component")
    axes[1].set_xlabel("Principal Component")
    axes[1].set_ylabel("Explained Variance Ratio")
    axes[1].legend()

    plt.tight_layout()
    fig_path = DOCS_FIGURES_DIR / "07_pca_scatter.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(fig_path))
    print(f"  PCA chart saved: {fig_path.name}")


# ── Section 5: t-SNE ─────────────────────────────────────────
print("\n[2/2] Running t-SNE (this takes a few minutes)...")

with mlflow.start_run(run_name="tsne"):
    n_sub = min(5_000 if FAST_MODE else TSNE_SUBSAMPLE, len(X))
    mlflow.log_params({
        "perplexity": TSNE_PERPLEXITY,
        "n_iter": TSNE_N_ITER,
        "n_subsample": n_sub,
        "fast_mode": FAST_MODE,
    })

    tsne_result = run_tsne(X, pca_result["scaler"], pca_result["pca"])
    tsne_metrics = tsne_result["metrics"]

    mlflow.log_metrics({k: v for k, v in tsne_metrics.items() if isinstance(v, (int, float))})
    mlflow.log_artifact(str(DIM_DIR / "tsne_metrics.json"))

    print(f"  KL divergence: {tsne_metrics['kl_divergence']:.4f}")
    print(f"  Subsample    : {tsne_metrics['n_subsample']:,} rows")

    # t-SNE scatter — colored by geo cluster
    tsne_2d = tsne_result["tsne_2d"]
    subsample_idx = tsne_result["subsample_idx"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("t-SNE — 2D Embedding of Crime Feature Space", y=1.01)

    # Panel 1: colored by geo cluster
    if geo_labels is not None and len(geo_labels) > max(subsample_idx):
        sc = axes[0].scatter(
            tsne_2d["tSNE1"], tsne_2d["tSNE2"],
            c=geo_labels[subsample_idx], cmap="tab10",
            s=3, alpha=0.5,
        )
        plt.colorbar(sc, ax=axes[0], label="Geo Cluster")
        axes[0].set_title("t-SNE colored by Geographic Cluster")
    else:
        axes[0].scatter(tsne_2d["tSNE1"], tsne_2d["tSNE2"],
                        s=3, alpha=0.4, color="steelblue")
        axes[0].set_title("t-SNE Embedding")
    axes[0].set_xlabel("tSNE1"); axes[0].set_ylabel("tSNE2")

    # Panel 2: colored by temporal cluster
    if temp_labels is not None and len(temp_labels) > max(subsample_idx):
        sc2 = axes[1].scatter(
            tsne_2d["tSNE1"], tsne_2d["tSNE2"],
            c=temp_labels[subsample_idx], cmap="Set1",
            s=3, alpha=0.5,
        )
        plt.colorbar(sc2, ax=axes[1], label="Temporal Cluster")
        axes[1].set_title("t-SNE colored by Temporal Cluster")
    else:
        axes[1].scatter(tsne_2d["tSNE1"], tsne_2d["tSNE2"],
                        s=3, alpha=0.4, color="coral")
        axes[1].set_title("t-SNE Embedding (temporal)")
    axes[1].set_xlabel("tSNE1"); axes[1].set_ylabel("tSNE2")

    plt.tight_layout()
    fig_path = DOCS_FIGURES_DIR / "07_tsne_scatter.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    mlflow.log_artifact(str(fig_path))
    print(f"  t-SNE chart saved: {fig_path.name}")


# ── Section 6: Results + summary ─────────────────────────────
pca_m = pca_result["metrics"]
tsne_m = tsne_result["metrics"]

print(f"\n{'=' * 55}")
print(f"  SECTION 7: DIMENSIONALITY REDUCTION COMPLETE")
print(f"{'=' * 55}")
print(f"\n--- PCA Results ---")
print(f"{'Components':<30} {pca_m['n_components']:>10}")
for i, v in enumerate(pca_m["explained_variance_ratio"]):
    print(f"{'  PC' + str(i+1) + ' explained variance':<30} {v:>10.1%}")
print(f"{'Cumulative variance':<30} {pca_m['cumulative_variance']:>10.1%}")
print(f"{'Target met (>= ' + str(int(PCA_VARIANCE_TARGET*100)) + '%)':<30} {'YES' if pca_m['variance_target_met'] else 'NO':>10}")

print(f"\n--- t-SNE Results ---")
print(f"{'Subsample size':<30} {tsne_m['n_subsample']:>10,}")
print(f"{'Perplexity':<30} {tsne_m['perplexity']:>10}")
print(f"{'KL divergence':<30} {tsne_m['kl_divergence']:>10.4f}")

print(f"""
ARTIFACTS:
  {DIM_DIR}/pca_2d.csv
  {DIM_DIR}/pca_3d.csv
  {DIM_DIR}/pca_metrics.json
  {DIM_DIR}/tsne_2d.csv
  {DIM_DIR}/tsne_metrics.json
  docs/figures/07_pca_scatter.png
  docs/figures/07_tsne_scatter.png

NEXT: notebooks/08_mlflow_experiments.py
""")
