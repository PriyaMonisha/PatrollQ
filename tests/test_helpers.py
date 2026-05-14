# filename: tests/test_helpers.py
# purpose:  Unit tests for src/utils/helpers.py — NumpyEncoder and save_json
# version:  1.0

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils.helpers import NumpyEncoder, save_json


class TestNumpyEncoder:
    def test_np_bool_true(self):
        assert json.dumps({"v": np.bool_(True)}, cls=NumpyEncoder) == '{"v": true}'

    def test_np_bool_false(self):
        assert json.dumps({"v": np.bool_(False)}, cls=NumpyEncoder) == '{"v": false}'

    def test_np_integer(self):
        result = json.loads(json.dumps({"v": np.int64(42)}, cls=NumpyEncoder))
        assert result["v"] == 42
        assert isinstance(result["v"], int)

    def test_np_floating(self):
        result = json.loads(json.dumps({"v": np.float64(3.14)}, cls=NumpyEncoder))
        assert abs(result["v"] - 3.14) < 1e-6
        assert isinstance(result["v"], float)

    def test_np_ndarray_1d(self):
        result = json.loads(json.dumps({"v": np.array([1, 2, 3])}, cls=NumpyEncoder))
        assert result["v"] == [1, 2, 3]

    def test_np_ndarray_2d(self):
        result = json.loads(json.dumps({"v": np.array([[1, 2], [3, 4]])}, cls=NumpyEncoder))
        assert result["v"] == [[1, 2], [3, 4]]

    def test_native_types_pass_through(self):
        data = {"int": 1, "float": 2.5, "str": "hello", "list": [1, 2]}
        assert json.loads(json.dumps(data, cls=NumpyEncoder)) == data

    def test_unserializable_raises(self):
        with pytest.raises(TypeError):
            json.dumps({"v": object()}, cls=NumpyEncoder)


class TestSaveJson:
    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.json"
            save_json({"key": "value"}, path)
            assert path.exists()

    def test_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "deep" / "out.json"
            save_json({"key": "value"}, path)
            assert path.exists()

    def test_roundtrip_numpy_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "metrics.json"
            data = {
                "silhouette": np.float64(0.412345),
                "n_clusters": np.int64(5),
                "labels": np.array([0, 1, 2]),
            }
            save_json(data, path)
            loaded = json.loads(path.read_text())
            assert abs(loaded["silhouette"] - 0.412345) < 1e-5
            assert loaded["n_clusters"] == 5
            assert loaded["labels"] == [0, 1, 2]

    def test_output_is_indented(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "out.json"
            save_json({"a": 1, "b": 2}, path)
            content = path.read_text()
            assert "\n" in content
