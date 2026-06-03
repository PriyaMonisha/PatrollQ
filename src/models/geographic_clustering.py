# filename: src/models/geographic_clustering.py
# purpose:  Geographic crime hotspot clustering — K-Means, DBSCAN, Hierarchical
# version:  1.0

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from config import (
    ARTIFACTS_DIR,
    DBSCAN_EPS,
    DBSCAN_MIN_SAMPLES,
    FAST_MODE,
    HIERARCHICAL_N_CLUSTERS,
    HIERARCHICAL_SUBSAMPLE,
    KMEANS_GEO_K_RANGE,
    KMEANS_GEO_N_CLUSTERS,
    RANDOM_STATE,
)
from src.utils.model_io import save_model as _save_model_artifact

logger = logging.getLogger(__name__)

# ── Artifact paths ───────────────────────────────────────────
GEO_DIR = ARTIFACTS_DIR / "geographic"

# ── Feature columns used for geographic clustering ───────────
GEO_FEATURES = ["latitude", "longitude"]


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════

def run_kmeans_geo(
    df: pd.DataFrame,
    k: Optional[int] = None,
) -> dict:
    """
    Fit K-Means on lat/lon coordinates.

    Args:
        df: Full featured DataFrame (needs primary_type, arrest, Crime_Severity_Score
            for cluster profiling in addition to latitude/longitude)
        k:  Number of clusters (default: KMEANS_GEO_N_CLUSTERS from config)

    Returns:
        dict with keys: labels (np.ndarray), metrics (dict), model (KMeans)
    """
    k = k or KMEANS_GEO_N_CLUSTERS

    # Extract and align coords — dropna may reduce row count
    coords = df[GEO_FEATURES].dropna()
    X = coords.to_numpy(dtype=np.float32)

    logger.info(f"[K-Means Geo] Fitting k={k} on {len(X):,} points...")
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X)

    metrics = _compute_metrics(X, labels, model_name="kmeans_geo", k=k)
    metrics["inertia"] = round(float(model.inertia_), 2)
    _save_labels(labels, df, "kmeans")
    _save_metrics(metrics, "kmeans")

    # Save model for O(1) API inference (replaces O(n) nearest-neighbor scan)
    _save_model_artifact(model, GEO_DIR, "kmeans_model.pkl")

    # Generate cluster profile — pass df.loc[coords.index] so indices align with labels
    profile = _compute_cluster_profile(df.loc[coords.index], labels, k)
    _save_json(profile, "cluster_profile.json")

    logger.info(
        f"[K-Means Geo] silhouette={metrics['silhouette']:.4f} | "
        f"inertia={metrics['inertia']:.1f}"
    )
    return {"labels": labels, "metrics": metrics, "model": model}


def run_elbow(df: pd.DataFrame) -> dict:
    """
    Run K-Means for K in KMEANS_GEO_K_RANGE and return inertia + silhouette per K.
    Used to pick the optimal K visually (elbow) and by silhouette peak.

    Returns:
        dict with keys: k_values, inertias, silhouettes
    """
    k_range = range(2, 6) if FAST_MODE else KMEANS_GEO_K_RANGE
    X = _geo_features(df)

    inertias, silhouettes = [], []

    for k in k_range:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = model.fit_predict(X)
        inertias.append(float(model.inertia_))
        sil = float(silhouette_score(X, labels, sample_size=5_000, random_state=RANDOM_STATE))
        silhouettes.append(sil)
        logger.info(f"  K={k} | inertia={model.inertia_:.1f} | silhouette={sil:.4f}")

    result = {
        "k_values": list(k_range),
        "inertias": inertias,
        "silhouettes": silhouettes,
        "best_k_silhouette": int(list(k_range)[silhouettes.index(max(silhouettes))]),
    }
    _save_json(result, "elbow_results.json")
    logger.info(f"[Elbow] Best K by silhouette: {result['best_k_silhouette']}")
    return result


def run_dbscan_geo(df: pd.DataFrame) -> dict:
    """
    Fit DBSCAN on lat/lon coordinates.

    eps is in DEGREES (not km). At Chicago lat 42°N:
        0.008° ≈ 656m  (DBSCAN_EPS from config)

    Returns:
        dict with keys: labels, metrics, noise_fraction
    """
    X = _geo_features(df)

    logger.info(
        f"[DBSCAN Geo] eps={DBSCAN_EPS}° (~{DBSCAN_EPS * 82:.0f}m at 42°N) | "
        f"min_samples={DBSCAN_MIN_SAMPLES}"
    )
    model = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, n_jobs=-1)
    labels = model.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int((labels == -1).sum())
    noise_fraction = noise_count / len(labels)

    metrics = {
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_fraction": round(noise_fraction, 6),
        "eps": DBSCAN_EPS,
        "min_samples": DBSCAN_MIN_SAMPLES,
    }

    # Silhouette only if enough clusters and non-noise points
    non_noise = labels != -1
    if n_clusters >= 2 and non_noise.sum() > 100:
        sil = float(silhouette_score(
            X[non_noise], labels[non_noise],
            sample_size=min(5_000, non_noise.sum()),
            random_state=RANDOM_STATE,
        ))
        metrics["silhouette"] = round(sil, 6)
    else:
        metrics["silhouette"] = None

    _save_labels(labels, df, "dbscan")
    _save_metrics(metrics, "dbscan")

    logger.info(
        f"[DBSCAN Geo] clusters={n_clusters} | "
        f"noise={noise_fraction:.1%} | "
        f"silhouette={metrics.get('silhouette')}"
    )
    return {"labels": labels, "metrics": metrics}


def run_hierarchical_geo(df: pd.DataFrame) -> dict:
    """
    Fit Agglomerative (Ward linkage) clustering on a subsample.

    WHY subsample: Ward linkage requires O(n²) memory.
    500K rows → ~200GB RAM. We subsample HIERARCHICAL_SUBSAMPLE rows.

    Returns:
        dict with keys: labels (on subsample), metrics, subsample_idx
    """
    X_full = _geo_features(df)
    n_sub = min(HIERARCHICAL_SUBSAMPLE, len(X_full))

    rng = np.random.default_rng(RANDOM_STATE)
    subsample_idx = rng.choice(len(X_full), size=n_sub, replace=False)
    X_sub = X_full[subsample_idx]

    logger.info(
        f"[Hierarchical Geo] Ward linkage | k={HIERARCHICAL_N_CLUSTERS} | "
        f"subsample={n_sub:,} rows"
    )
    model = AgglomerativeClustering(
        n_clusters=HIERARCHICAL_N_CLUSTERS, linkage="ward"
    )
    labels = model.fit_predict(X_sub)

    sil = float(silhouette_score(X_sub, labels, random_state=RANDOM_STATE))
    db = float(davies_bouldin_score(X_sub, labels))

    metrics = {
        "n_clusters": HIERARCHICAL_N_CLUSTERS,
        "subsample_size": n_sub,
        "silhouette": round(sil, 6),
        "davies_bouldin": round(db, 6),
        "linkage": "ward",
    }

    # Save subsample labels only (not interpolated to full dataset)
    labels_df = df.iloc[subsample_idx][["case_number", "latitude", "longitude"]].copy()
    labels_df["cluster"] = labels
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(GEO_DIR / "hierarchical_labels.csv", index=False)

    _save_metrics(metrics, "hierarchical")

    logger.info(
        f"[Hierarchical Geo] silhouette={sil:.4f} | davies_bouldin={db:.4f}"
    )
    return {"labels": labels, "metrics": metrics, "subsample_idx": subsample_idx}


# ═══════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════

def _geo_features(df: pd.DataFrame) -> np.ndarray:
    """Extract and validate lat/lon as numpy array. Drops any remaining nulls."""
    missing = [c for c in GEO_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing geo columns: {missing}")

    coords = df[GEO_FEATURES].dropna()
    if len(coords) < len(df):
        logger.warning(f"Dropped {len(df) - len(coords):,} rows with null coordinates")

    return coords.to_numpy(dtype=np.float32)


def _compute_metrics(
    X: np.ndarray,
    labels: np.ndarray,
    model_name: str,
    k: int,
) -> dict:
    """Compute silhouette, davies-bouldin, calinski-harabasz metrics."""
    sil = float(silhouette_score(
        X, labels,
        sample_size=min(5_000, len(X)),
        random_state=RANDOM_STATE,
    ))
    db = float(davies_bouldin_score(X, labels))
    ch = float(calinski_harabasz_score(X, labels))

    metrics: dict = {
        "model": model_name,
        "n_clusters": k,
        "silhouette": round(sil, 6),
        "davies_bouldin": round(db, 6),
        "calinski_harabasz": round(ch, 2),
        "n_samples": len(X),
    }
    return metrics


def _save_labels(labels: np.ndarray, df: pd.DataFrame, model_name: str) -> None:
    """Save cluster labels aligned with case_number, lat, lon."""
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    coords = df[["case_number", "latitude", "longitude"]].dropna().copy()
    coords["cluster"] = labels
    out_path = GEO_DIR / f"{model_name}_labels.csv"
    coords.to_csv(out_path, index=False)
    logger.info(f"  Labels saved: {out_path.name}")


def _save_metrics(metrics: dict, model_name: str) -> None:
    """Save metrics JSON to artifacts/geographic/."""
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GEO_DIR / f"{model_name}_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  Metrics saved: {out_path.name}")


def _save_json(data: dict, filename: str) -> None:
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GEO_DIR / filename
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"  Saved: {out_path.name}")


def _compute_cluster_profile(df: pd.DataFrame, labels: np.ndarray, k: int) -> dict:
    """
    Build a risk profile per cluster using crime severity, not cluster size.

    Risk based on average Crime_Severity_Score (1–10):
      HIGH   >= 7.0  (homicide, assault, weapons)
      MEDIUM >= 4.0  (theft, narcotics, deception)
      LOW    <  4.0  (criminal damage, trespass, liquor)

    Size-based risk gives backwards results — a large cluster of minor crimes
    scores HIGH while a small cluster of homicides scores LOW.

    Requires df to have: Crime_Severity_Score, primary_type, arrest columns.
    df must be df.loc[coords.index] (index-aligned with labels from dropna()).
    """
    df = df.copy()
    df["_cluster"] = labels
    profile: dict = {}

    for cluster_id in range(k):
        cluster_df = df[df["_cluster"] == cluster_id]

        if len(cluster_df) == 0:
            logger.warning(f"Cluster {cluster_id} is empty — K may be too large")
            # Include empty cluster so profile has all k keys (prevents KeyError in predictor)
            profile[int(cluster_id)] = {
                "dominant_crime": "UNKNOWN",
                "risk_level": "LOW",
                "avg_severity_score": 0.0,
                "crime_count": 0,
                "arrest_rate": None,
                "warning": "Empty cluster",
            }
            continue

        if "Crime_Severity_Score" in cluster_df.columns:
            avg_sev = float(cluster_df["Crime_Severity_Score"].mean())
        elif "primary_type" in cluster_df.columns:
            # Compute severity from primary_type when feature column absent
            from config import SEVERITY_SCORES, SEVERITY_DEFAULT
            avg_sev = float(
                cluster_df["primary_type"].map(SEVERITY_SCORES).fillna(SEVERITY_DEFAULT).mean()
            )
        else:
            avg_sev = 5.0  # neutral default — no crime type info available

        risk = "HIGH" if avg_sev >= 7.0 else "MEDIUM" if avg_sev >= 4.0 else "LOW"

        dominant = (
            cluster_df["primary_type"].value_counts().index[0]
            if "primary_type" in cluster_df.columns else "UNKNOWN"
        )
        arrest_rate = (
            round(float(cluster_df["arrest"].mean()), 3)
            if "arrest" in cluster_df.columns else None
        )

        profile[int(cluster_id)] = {
            "dominant_crime": dominant,
            "risk_level": risk,
            "avg_severity_score": round(avg_sev, 3),
            "crime_count": int(len(cluster_df)),
            "arrest_rate": arrest_rate,
        }

    logger.info(f"  Cluster profile generated: {k} clusters")
    return profile
