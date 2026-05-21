# filename: tests/test_pipeline_integration.py
# purpose:  Integration smoke tests — full mini-pipeline on 100 rows

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Prevent tests from overwriting production artifacts
_NO_SAVE = patch.multiple(
    "src.models.geographic_clustering",
    _save_labels=lambda *a, **kw: None,
    _save_metrics=lambda *a, **kw: None,
)
_NO_SAVE_TEMP = patch.multiple(
    "src.models.temporal_clustering",
    _save_labels=lambda *a, **kw: None,
    _save_metrics=lambda *a, **kw: None,
    _save_model=lambda *a, **kw: None,
)


def _minimal_raw(n: int = 120) -> pd.DataFrame:
    """Loader-normalised schema (snake_case). Preprocessor expects these column names."""
    np.random.seed(42)
    return pd.DataFrame({
        "id":                   range(n),
        "case_number":          [f"JK{i:06d}" for i in range(n)],
        "date":                 [f"{(i % 12) + 1:02d}/{(i % 28) + 1:02d}/2024 {i % 24:02d}:30:00 {'AM' if i % 24 < 12 else 'PM'}" for i in range(n)],
        "block":                ["001XX N STATE ST"] * n,
        "iucr":                 ["0560"] * n,
        "primary_type":         np.random.choice(["THEFT", "BATTERY", "ASSAULT", "NARCOTICS"], n),
        "description":          ["SIMPLE"] * n,
        "location_description": ["STREET"] * n,
        "arrest":               np.random.choice([True, False], n),
        "domestic":             np.random.choice([True, False], n),
        "beat":                 np.random.randint(100, 2400, n),
        "district":             np.random.randint(1, 25, n),
        "ward":                 np.random.randint(1, 50, n),
        "community_area":       np.random.randint(1, 77, n),
        "fbi_code":             ["06"] * n,
        "x_coordinate":         np.random.randint(1100000, 1200000, n),
        "y_coordinate":         np.random.randint(1800000, 1900000, n),
        "year":                 [2024] * n,
        "updated_on":           ["05/16/2024 03:45:00 PM"] * n,
        "latitude":             np.random.uniform(41.65, 41.99, n),
        "longitude":            np.random.uniform(-87.89, -87.53, n),
        "location":             ["(41.85, -87.65)"] * n,
    })


class TestPipelineIntegration:

    def test_preprocess_does_not_crash(self):
        from src.data.preprocessor import preprocess_data
        df = _minimal_raw()
        result = preprocess_data(df, save_encoders=False)
        assert len(result) > 0

    def test_feature_engineering_after_preprocess(self):
        from src.data.preprocessor import preprocess_data
        from src.features.engineer import CrimeFeatureEngineer
        df = _minimal_raw()
        clean = preprocess_data(df, save_encoders=False)
        eng = CrimeFeatureEngineer()
        eng.fit(clean)
        features = eng.transform(clean)
        assert len(features) == len(clean)
        assert "hour_sin" in features.columns

    def test_geographic_clustering_after_preprocess(self):
        from src.data.preprocessor import preprocess_data
        from src.models.geographic_clustering import run_kmeans_geo
        df = _minimal_raw(200)
        clean = preprocess_data(df, save_encoders=False)
        with _NO_SAVE:
            result = run_kmeans_geo(clean, k=3)
        assert len(result["labels"]) == len(clean)

    def test_temporal_clustering_after_feature_engineering(self):
        from src.data.preprocessor import preprocess_data
        from src.features.engineer import CrimeFeatureEngineer
        from src.models.temporal_clustering import run_kmeans_temporal
        from config import TEMPORAL_FEATURES

        df = _minimal_raw(200)
        clean = preprocess_data(df, save_encoders=False)
        eng = CrimeFeatureEngineer()
        eng.fit(clean)
        features = eng.transform(clean)

        available = [f for f in TEMPORAL_FEATURES if f in features.columns]
        assert len(available) >= 3, f"Too few temporal features: {available}"
        X = features[available]
        with _NO_SAVE_TEMP:
            result = run_kmeans_temporal(X, k=2)
        assert len(result["labels"]) == len(X)

    def test_labels_length_matches_input(self):
        """Core invariant: label count always equals input row count."""
        from src.data.preprocessor import preprocess_data
        from src.models.geographic_clustering import run_kmeans_geo
        for n in (100, 150, 200):
            df = _minimal_raw(n)
            clean = preprocess_data(df, save_encoders=False)
            with _NO_SAVE:
                result = run_kmeans_geo(clean, k=3)
            assert len(result["labels"]) == len(clean), \
                f"Label count {len(result['labels'])} != input count {len(clean)} for n={n}"
