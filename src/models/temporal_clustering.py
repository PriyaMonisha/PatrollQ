# filename: src/models/temporal_clustering.py
# purpose:  Temporal crime pattern clustering — K-Means on cyclical time features
# version:  1.0

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from config import (
    ARTIFACTS_DIR,
    FAST_MODE,
    KMEANS_TEMPORAL_K_RANGE,
    KMEANS_TEMPORAL_N_CLUSTERS,
    RANDOM_STATE,
    TEMPORAL_FEATURES,
)

logger = logging.getLogger(__name__)

# ── Artifact path (Path only — no mkdir at module level) ─────
TEMPORAL_DIR = ARTIFACTS_DIR / "temporal"

# ── Feature set used for temporal clustering ─────────────────
# K=4 pre-chosen: crime patterns segment into ~4 activity windows:
#   Cluster 0: Late-night (0h-5h)      Cluster 1: Morning rush (6h-10h)
#   Cluster 2: Afternoon peak (11h-17h) Cluster 3: Evening (18h-23h)
# Validated against elbow + silhouette analysis via run_elbow_temporal()


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════

def run_elbow_temporal(
    X: pd.DataFrame,
    chosen_k: int = KMEANS_TEMPORAL_N_CLUSTERS,
) -> dict:
    """
    Fit K-Means for each K in KMEANS_TEMPORAL_K_RANGE and collect metrics.

    Args:
        X:        Feature DataFrame (TEMPORAL_FEATURES columns only)
        chosen_k: The K that will be used for the final model (stored in output
                  alongside best_k_by_silhouette for comparison)

    Returns:
        dict with keys: k_values, inertias, silhouettes,
                        best_k_by_silhouette, chosen_k
    """
    k_range = range(2, 5) if FAST_MODE else KMEANS_TEMPORAL_K_RANGE
    X_arr = X.to_numpy(dtype=np.float32)

    inertias, silhouettes = [], []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X_arr)
        inertias.append(float(model.inertia_))

        sil = float(silhouette_score(
            X_arr, labels,
            sample_size=min(5_000, len(X_arr)),
            random_state=RANDOM_STATE,
        ))
        silhouettes.append(sil)
        logger.info(f"  K={k} | inertia={model.inertia_:.1f} | silhouette={sil:.4f}")

    best_k = int(list(k_range)[silhouettes.index(max(silhouettes))])

    result = {
        "k_values": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k_by_silhouette": best_k,
        "chosen_k": chosen_k,
    }
    _save_json(result, "elbow_results.json")
    logger.info(f"[Temporal Elbow] Best K by silhouette: {best_k} | Chosen K: {chosen_k}")
    return result


def run_kmeans_temporal(
    X: pd.DataFrame,
    k: int = KMEANS_TEMPORAL_N_CLUSTERS,
) -> dict:
    """
    Fit K-Means on temporal features.

    Args:
        X: Feature DataFrame (TEMPORAL_FEATURES columns only)
        k: Number of clusters (default: KMEANS_TEMPORAL_N_CLUSTERS = 4)

    Returns:
        {
            'labels':  np.ndarray of shape (n_samples,) — cluster assignments
            'model':   fitted sklearn KMeans object
            'metrics': {silhouette, davies_bouldin, inertia, n_clusters}
        }
    """
    X_arr = X.to_numpy(dtype=np.float32)

    logger.info(f"[K-Means Temporal] Fitting k={k} on {len(X_arr):,} points...")
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_arr)

    sil = float(silhouette_score(
        X_arr, labels,
        sample_size=min(5_000, len(X_arr)),
        random_state=RANDOM_STATE,
    ))
    db = float(davies_bouldin_score(X_arr, labels))

    metrics = {
        "model": "kmeans_temporal",
        "n_clusters": k,
        "silhouette": round(sil, 6),
        "davies_bouldin": round(db, 6),
        "inertia": round(float(model.inertia_), 2),
        "n_samples": len(X_arr),
    }

    _save_labels(labels, X, "kmeans")
    _save_metrics(metrics, "kmeans")
    _save_model(model, "kmeans_model.pkl")

    logger.info(
        f"[K-Means Temporal] silhouette={sil:.4f} | "
        f"davies_bouldin={db:.4f} | inertia={model.inertia_:.1f}"
    )
    return {"labels": labels, "model": model, "metrics": metrics}


# ═══════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════

def _save_labels(labels: np.ndarray, X: pd.DataFrame, model_name: str) -> None:
    TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"row_idx": range(len(labels)), "cluster": labels})
    path = TEMPORAL_DIR / f"{model_name}_labels.csv"
    out.to_csv(path, index=False)
    logger.info(f"  Labels saved: {path.name}")


def _save_metrics(metrics: dict, model_name: str) -> None:
    TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMPORAL_DIR / f"{model_name}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics saved: {path.name}")


def _save_model(model, filename: str) -> None:
    TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMPORAL_DIR / filename
    joblib.dump(model, path)
    logger.info(f"  Model saved: {path.name}")


def _save_json(data: dict, filename: str) -> None:
    TEMPORAL_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMPORAL_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"  Saved: {path.name}")
