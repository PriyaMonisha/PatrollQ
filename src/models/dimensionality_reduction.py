# filename: src/models/dimensionality_reduction.py
# purpose:  Dimensionality reduction — PCA and t-SNE on FULL_FEATURES
# version:  1.0

import json
import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import (
    ARTIFACTS_DIR,
    FAST_MODE,
    PCA_N_COMPONENTS,
    PCA_SCATTER_SAMPLE,
    PCA_VARIANCE_TARGET,
    RANDOM_STATE,
    TSNE_N_COMPONENTS,
    TSNE_N_ITER,
    TSNE_PCA_PREPROCESS_N,
    TSNE_PERPLEXITY,
    TSNE_SUBSAMPLE,
)

logger = logging.getLogger(__name__)

# ── Artifact path (Path only — no mkdir at module level) ─────
DIM_DIR = ARTIFACTS_DIR / "dimensionality"


# ═══════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════

def run_pca(X: pd.DataFrame) -> dict:
    """
    Fit PCA on REDUCTION_FEATURES and return 2D + 3D projections.

    Steps:
      1. StandardScaler (PCA is sensitive to feature scale)
      2. Fit PCA with PCA_N_COMPONENTS components
      3. Save 2D scatter sample + explained variance metadata

    Args:
        X: Feature DataFrame (REDUCTION_FEATURES columns)

    Returns:
        {
            'pca_2d':       pd.DataFrame with cols [PC1, PC2] (scatter sample)
            'pca_3d':       pd.DataFrame with cols [PC1, PC2, PC3]
            'scaler':       fitted StandardScaler
            'pca':          fitted PCA object
            'metrics': {
                'explained_variance_ratio': list,
                'cumulative_variance':      list,
                'n_components':             int,
                'variance_target_met':      bool,
            }
        }
    """
    X_arr = X.to_numpy(dtype=np.float64)

    logger.info(f"[PCA] Fitting {PCA_N_COMPONENTS} components on {len(X_arr):,} rows...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    pca = PCA(n_components=PCA_N_COMPONENTS, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_scaled)

    evr = pca.explained_variance_ratio_.tolist()
    cumulative = float(np.cumsum(evr)[-1])
    target_met = cumulative >= PCA_VARIANCE_TARGET

    logger.info(
        "[PCA] Explained variance: "
        + " | ".join(f"PC{i+1}={v:.1%}" for i, v in enumerate(evr))
        + f" | Total={cumulative:.1%} "
        + ("(PASS)" if target_met else f"(below {PCA_VARIANCE_TARGET:.0%} target)")
    )

    # 2D scatter: subsample for readability
    n_scatter = min(PCA_SCATTER_SAMPLE, len(X_arr))
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_arr), size=n_scatter, replace=False)

    pca_2d = pd.DataFrame(components[idx, :2], columns=["PC1", "PC2"])
    pca_3d = pd.DataFrame(components[:, :3], columns=["PC1", "PC2", "PC3"])

    metrics = {
        "explained_variance_ratio": [round(v, 6) for v in evr],
        "cumulative_variance": round(cumulative, 6),
        "n_components": PCA_N_COMPONENTS,
        "variance_target": PCA_VARIANCE_TARGET,
        "variance_target_met": target_met,
        "n_samples": len(X_arr),
        "scatter_sample_size": n_scatter,
    }

    _save_csv(pca_2d, "pca_2d.csv")
    _save_csv(pca_3d, "pca_3d.csv")
    _save_json(metrics, "pca_metrics.json")

    return {
        "pca_2d": pca_2d,
        "pca_3d": pca_3d,
        "scaler": scaler,
        "pca": pca,
        "metrics": metrics,
    }


def run_tsne(X: pd.DataFrame, pca_scaler: StandardScaler, fitted_pca: PCA) -> dict:
    """
    Run t-SNE on a subsample, pre-processed with PCA to 50 components.

    Why PCA first: t-SNE is O(n²) and O(d²). Reducing to 50 components
    via PCA before t-SNE dramatically cuts compute without losing structure.

    Args:
        X:           Feature DataFrame (REDUCTION_FEATURES columns)
        pca_scaler:  Fitted StandardScaler from run_pca()
        fitted_pca:  Fitted PCA from run_pca() — reused for preprocessing

    Returns:
        {
            'tsne_2d':      pd.DataFrame with cols [tSNE1, tSNE2]
            'metrics':      {n_subsample, perplexity, n_iter}
            'subsample_idx': indices used from X
        }
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        raise ImportError("scikit-learn is required for t-SNE")

    n_sub = min(TSNE_SUBSAMPLE, len(X))
    # FAST_MODE: use smaller subsample to keep runtime <30s
    if FAST_MODE:
        n_sub = min(5_000, len(X))

    rng = np.random.default_rng(RANDOM_STATE)
    subsample_idx = rng.choice(len(X), size=n_sub, replace=False)
    X_sub = X.iloc[subsample_idx].to_numpy(dtype=np.float64)

    logger.info(
        f"[t-SNE] Subsample: {n_sub:,} | "
        f"perplexity={TSNE_PERPLEXITY} | n_iter={TSNE_N_ITER}"
    )

    # Step 1: Scale
    X_scaled = pca_scaler.transform(X_sub)

    # Step 2: PCA → TSNE_PCA_PREPROCESS_N components
    n_pca_pre = min(TSNE_PCA_PREPROCESS_N, X_scaled.shape[1])
    pca_pre = PCA(n_components=n_pca_pre, random_state=RANDOM_STATE)
    X_pca = pca_pre.fit_transform(X_scaled)
    logger.info(
        f"[t-SNE] PCA preprocess: {X_scaled.shape[1]} → {n_pca_pre} components "
        f"({pca_pre.explained_variance_ratio_.sum():.1%} variance retained)"
    )

    # Step 3: t-SNE
    tsne = TSNE(
        n_components=TSNE_N_COMPONENTS,
        perplexity=TSNE_PERPLEXITY,
        n_iter=TSNE_N_ITER,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    embedding = tsne.fit_transform(X_pca)
    logger.info(f"[t-SNE] Done. KL divergence: {tsne.kl_divergence_:.4f}")

    tsne_2d = pd.DataFrame(embedding, columns=["tSNE1", "tSNE2"])

    metrics = {
        "n_subsample": n_sub,
        "perplexity": TSNE_PERPLEXITY,
        "n_iter": TSNE_N_ITER,
        "kl_divergence": round(float(tsne.kl_divergence_), 6),
        "pca_preprocess_n": n_pca_pre,
        "pca_variance_retained": round(float(pca_pre.explained_variance_ratio_.sum()), 6),
    }

    _save_csv(tsne_2d, "tsne_2d.csv")
    _save_json(metrics, "tsne_metrics.json")

    return {"tsne_2d": tsne_2d, "metrics": metrics, "subsample_idx": subsample_idx}


# ═══════════════════════════════════════════════════════════
# PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════

def _save_csv(df: pd.DataFrame, filename: str) -> None:
    DIM_DIR.mkdir(parents=True, exist_ok=True)
    path = DIM_DIR / filename
    df.to_csv(path, index=False)
    logger.info(f"  Saved: {path.name}")


def _save_json(data: dict, filename: str) -> None:
    DIM_DIR.mkdir(parents=True, exist_ok=True)
    path = DIM_DIR / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"  Saved: {path.name}")
