# filename: tests/test_drift.py
# purpose:  Unit tests for drift detection module

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.monitoring.drift import _check_ks, _check_arrest_rate


class TestKSDrift:

    def test_identical_distributions_no_drift(self):
        np.random.seed(42)
        data = np.random.normal(41.85, 0.1, 500)
        result = _check_ks("latitude", pd.Series(data), pd.Series(data))
        assert not result["drift_detected"], "Identical data should not show drift"
        assert result["p_value"] == 1.0 or result["p_value"] > _P_VALUE_THRESHOLD()

    def test_shifted_distribution_detects_drift(self):
        np.random.seed(42)
        ref = np.random.normal(41.85, 0.1, 500)
        cur = np.random.normal(42.5, 0.1, 500)   # clearly shifted
        result = _check_ks("latitude", pd.Series(ref), pd.Series(cur))
        assert result["drift_detected"], "Clearly shifted distribution should be detected"

    def test_result_has_required_keys(self):
        np.random.seed(42)
        data = np.random.normal(0, 1, 100)
        result = _check_ks("test_feature", pd.Series(data), pd.Series(data))
        for key in ("feature", "statistic", "p_value", "drift_detected", "method"):
            assert key in result, f"Missing key: {key}"

    def test_feature_name_preserved(self):
        data = pd.Series(np.random.normal(0, 1, 100))
        result = _check_ks("my_feature", data, data)
        assert result["feature"] == "my_feature"

    def test_insufficient_data_returns_no_drift(self):
        result = _check_ks("tiny", pd.Series([1.0, 2.0]), pd.Series([1.0, 2.0]))
        assert not result["drift_detected"]


class TestArrestRateDrift:

    def test_same_rate_no_drift(self):
        arr = pd.Series([True, False, True, True, False] * 50)
        result = _check_arrest_rate(arr, arr)
        assert not result["drift_detected"]

    def test_large_shift_detects_drift(self):
        ref = pd.Series([True] * 100 + [False] * 100)   # 50% arrest rate
        cur = pd.Series([True] * 5 + [False] * 195)     # 2.5% arrest rate
        result = _check_arrest_rate(ref, cur)
        assert result["drift_detected"], "Large arrest rate shift should be detected"

    def test_handles_string_booleans(self):
        ref = pd.Series(["True", "False", "True"] * 50)
        cur = pd.Series(["True", "False", "True"] * 50)
        result = _check_arrest_rate(ref, cur)
        assert not result["drift_detected"]


def _P_VALUE_THRESHOLD():
    return 0.05
