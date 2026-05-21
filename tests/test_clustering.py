# filename: tests/test_clustering.py
# purpose:  Unit tests for geographic and temporal clustering modules

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))


def _geo_df(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "case_number": [f"JK{i}" for i in range(n)],
        "latitude":    np.random.uniform(41.65, 41.99, n),
        "longitude":   np.random.uniform(-87.89, -87.53, n),
    })


def _temporal_features(n: int = 200) -> pd.DataFrame:
    np.random.seed(42)
    hours = np.random.randint(0, 24, n)
    import math
    return pd.DataFrame({
        "hour_sin":   [math.sin(2 * math.pi * h / 24) for h in hours],
        "hour_cos":   [math.cos(2 * math.pi * h / 24) for h in hours],
        "day_sin":    np.random.uniform(-1, 1, n),
        "day_cos":    np.random.uniform(-1, 1, n),
        "month_sin":  np.random.uniform(-1, 1, n),
        "month_cos":  np.random.uniform(-1, 1, n),
        "Is_Weekend": np.random.choice([True, False], n),
    })


_SAVE_PATCH = patch.multiple(
    "src.models.geographic_clustering",
    _save_labels=lambda *a, **kw: None,
    _save_metrics=lambda *a, **kw: None,
)
_TEMP_SAVE_PATCH = patch.multiple(
    "src.models.temporal_clustering",
    _save_labels=lambda *a, **kw: None,
    _save_metrics=lambda *a, **kw: None,
    _save_model=lambda *a, **kw: None,
)


class TestGeographicClustering:

    def test_kmeans_returns_correct_label_count(self):
        from src.models.geographic_clustering import run_kmeans_geo
        df = _geo_df(300)
        with _SAVE_PATCH:
            result = run_kmeans_geo(df, k=3)
        assert "labels" in result
        assert len(result["labels"]) == len(df)

    def test_kmeans_label_values_in_range(self):
        from src.models.geographic_clustering import run_kmeans_geo
        df = _geo_df(300)
        k = 4
        with _SAVE_PATCH:
            result = run_kmeans_geo(df, k=k)
        unique = set(result["labels"])
        assert unique <= set(range(k)), f"Label values {unique} out of range for k={k}"

    def test_kmeans_metrics_dict_has_required_keys(self):
        from src.models.geographic_clustering import run_kmeans_geo
        df = _geo_df(300)
        with _SAVE_PATCH:
            result = run_kmeans_geo(df, k=3)
        for key in ("silhouette", "davies_bouldin", "calinski_harabasz", "n_clusters"):
            assert key in result["metrics"], f"Missing metric: {key}"

    def test_silhouette_score_range(self):
        from src.models.geographic_clustering import run_kmeans_geo
        df = _geo_df(300)
        with _SAVE_PATCH:
            result = run_kmeans_geo(df, k=3)
        sil = result["metrics"]["silhouette"]
        assert -1.0 <= sil <= 1.0, f"Silhouette {sil} out of valid range [-1, 1]"

    def test_calinski_harabasz_positive(self):
        from src.models.geographic_clustering import run_kmeans_geo
        df = _geo_df(300)
        with _SAVE_PATCH:
            result = run_kmeans_geo(df, k=3)
        ch = result["metrics"]["calinski_harabasz"]
        assert ch > 0, f"Calinski-Harabasz should be positive, got {ch}"


class TestTemporalClustering:

    def test_temporal_kmeans_returns_correct_label_count(self):
        from src.models.temporal_clustering import run_kmeans_temporal
        X = _temporal_features(300)
        with _TEMP_SAVE_PATCH:
            result = run_kmeans_temporal(X, k=3)
        assert len(result["labels"]) == len(X)

    def test_temporal_metrics_present(self):
        from src.models.temporal_clustering import run_kmeans_temporal
        X = _temporal_features(300)
        with _TEMP_SAVE_PATCH:
            result = run_kmeans_temporal(X, k=3)
        for key in ("silhouette", "davies_bouldin", "calinski_harabasz", "inertia"):
            assert key in result["metrics"], f"Missing metric: {key}"
