# filename: tests/test_feature_engineer.py
# purpose:  Unit tests for CrimeFeatureEngineer — cyclical encoding + normalization

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.engineer import CrimeFeatureEngineer


def _sample_df(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "latitude":     np.random.uniform(41.65, 41.99, n),
        "longitude":    np.random.uniform(-87.89, -87.53, n),
        "Hour":         np.random.randint(0, 24, n),
        "Day_of_Week":  np.random.randint(0, 7, n),
        "Month":        np.random.randint(1, 13, n),
        "Season":       np.random.randint(0, 4, n),
        "Is_Weekend":   np.random.choice([True, False], n),
        "primary_type": np.random.choice(["THEFT", "BATTERY", "ASSAULT"], n),
        # Alias — engineer checks for 'primary_type' (lowercase)
    })


class TestCrimeFeatureEngineer:

    def test_fit_transform_returns_dataframe(self):
        eng = CrimeFeatureEngineer()
        df = _sample_df()
        eng.fit(df)
        result = eng.transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_cyclical_encoding_pythagorean_identity(self):
        """sin²+cos² must equal 1 for all cyclical pairs."""
        eng = CrimeFeatureEngineer()
        df = _sample_df(200)
        eng.fit(df)
        result = eng.transform(df)
        for prefix in ("hour", "day", "month"):
            sin_col, cos_col = f"{prefix}_sin", f"{prefix}_cos"
            if sin_col in result.columns and cos_col in result.columns:
                identity = result[sin_col] ** 2 + result[cos_col] ** 2
                np.testing.assert_allclose(identity.values, 1.0, atol=1e-5,
                    err_msg=f"{prefix} sin²+cos² != 1")

    def test_geo_normalization_within_0_1(self):
        """lat_norm and lon_norm must be in [0, 1] for training data."""
        eng = CrimeFeatureEngineer()
        df = _sample_df(200)
        eng.fit(df)
        result = eng.transform(df)
        if "lat_norm" in result.columns:
            assert result["lat_norm"].between(0.0, 1.0).all(), "lat_norm out of [0,1]"
        if "lon_norm" in result.columns:
            assert result["lon_norm"].between(0.0, 1.0).all(), "lon_norm out of [0,1]"

    def test_hour_0_and_23_are_close_in_cyclical_space(self):
        """Hour 0 and Hour 23 should be adjacent in cyclical encoding."""
        h0_sin = math.sin(2 * math.pi * 0 / 24)
        h0_cos = math.cos(2 * math.pi * 0 / 24)
        h23_sin = math.sin(2 * math.pi * 23 / 24)
        h23_cos = math.cos(2 * math.pi * 23 / 24)
        dist = math.sqrt((h0_sin - h23_sin) ** 2 + (h0_cos - h23_cos) ** 2)
        assert dist < 1.0, f"Hour 0 and 23 should be close in cyclical space, got dist={dist:.4f}"

    def test_new_features_added_to_output(self):
        eng = CrimeFeatureEngineer()
        df = _sample_df()
        eng.fit(df)
        result = eng.transform(df)
        expected = {"hour_sin", "hour_cos", "day_sin", "day_cos", "lat_norm", "lon_norm"}
        missing = expected - set(result.columns)
        assert not missing, f"Missing features: {missing}"

    def test_fit_then_transform_same_shape(self):
        eng = CrimeFeatureEngineer()
        df = _sample_df(150)
        eng.fit(df)
        result = eng.transform(df)
        assert len(result) == len(df)

    def test_out_of_sample_coords_clipped_to_0_1(self):
        """Coords slightly outside training bounds should be clipped, not raise."""
        eng = CrimeFeatureEngineer()
        df_train = _sample_df(100)
        eng.fit(df_train)
        df_oos = df_train.copy()
        df_oos.loc[0, "latitude"] = 43.0   # outside Chicago bounds
        df_oos.loc[1, "longitude"] = -86.0
        result = eng.transform(df_oos)
        if "lat_norm" in result.columns:
            assert result["lat_norm"].between(0.0, 1.0).all()
