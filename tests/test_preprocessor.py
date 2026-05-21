# filename: tests/test_preprocessor.py
# purpose:  Unit tests for data preprocessing pipeline

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.preprocessor import preprocess_data


def _raw_df(n: int = 50) -> pd.DataFrame:
    """
    Raw DataFrame matching the loader-normalised schema (snake_case, 22 columns).
    The loader normalises column names before passing to preprocess_data().
    """
    np.random.seed(42)
    return pd.DataFrame({
        "id":                   range(n),
        "case_number":          [f"JK{i:06d}" for i in range(n)],
        "date":                 ["05/15/2024 08:30:00 AM"] * n,
        "block":                ["001XX N STATE ST"] * n,
        "iucr":                 ["0560"] * n,
        "primary_type":         np.random.choice(["THEFT", "BATTERY", "ASSAULT"], n),
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


class TestCrimePreprocessor:

    def test_preprocess_returns_dataframe(self):
        df = _raw_df()
        result = preprocess_data(df, save_encoders=False)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_duplicate_removal(self):
        df = _raw_df(50)
        # Add 5 exact duplicates
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)
        assert len(df) == 55
        result = preprocess_data(df, save_encoders=False)
        assert len(result) <= 50, "Duplicates not removed"

    def test_temporal_columns_extracted(self):
        df = _raw_df()
        result = preprocess_data(df, save_encoders=False)
        for col in ("Hour", "Day_of_Week", "Month", "Year", "Is_Weekend"):
            assert col in result.columns, f"Missing temporal column: {col}"

    def test_hour_range(self):
        df = _raw_df()
        result = preprocess_data(df, save_encoders=False)
        if "Hour" in result.columns:
            assert result["Hour"].between(0, 23).all()

    def test_geographic_bounds_filtering(self):
        df = _raw_df(50)
        # Inject obviously out-of-bounds rows
        df.loc[0, "Latitude"] = 99.0
        df.loc[1, "Longitude"] = 0.0
        result = preprocess_data(df, save_encoders=False)
        if "latitude" in result.columns:
            assert result["latitude"].between(41.0, 43.0).all()
        if "longitude" in result.columns:
            assert result["longitude"].between(-89.0, -87.0).all()

    def test_output_columns_lowercase(self):
        df = _raw_df()
        result = preprocess_data(df, save_encoders=False)
        # Core columns should be lowercase
        for col in ("latitude", "longitude", "arrest", "domestic"):
            assert col in result.columns, f"Expected lowercase column: {col}"

    def test_no_null_lat_lon_after_preprocess(self):
        df = _raw_df(50)
        # Force some null coords
        df.loc[:4, "Latitude"] = np.nan
        df.loc[:4, "Longitude"] = np.nan
        result = preprocess_data(df, save_encoders=False)
        if "latitude" in result.columns:
            assert result["latitude"].notna().all(), "Null latitudes remain after preprocessing"
