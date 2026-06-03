# filename: tests/test_api.py
# purpose:  Integration tests for FastAPI inference endpoints

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def _get_client():
    """Import TestClient lazily so tests are skipped if fastapi not installed."""
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)
    except ImportError:
        pytest.skip("fastapi not installed — install requirements-api.txt")


def _geo_mock():
    """Minimal KMeans mock — predict() returns cluster 3."""
    m = MagicMock()
    m.predict.return_value = np.array([3])
    return m


def _geo_profile():
    """Minimal cluster profile with only cluster 3."""
    return {
        3: {
            "risk_level": "HIGH",
            "dominant_crime": "ASSAULT",
            "avg_severity_score": 7.5,
            "arrest_rate": 0.21,
            "crime_count": 1234,
        }
    }


def _temporal_mock():
    """Minimal KMeans mock — predict() returns temporal cluster 2."""
    m = MagicMock()
    m.predict.return_value = np.array([2])
    return m


# ── Health ────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self):
        client = _get_client()
        response = client.get("/v1/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self):
        client = _get_client()
        data = client.get("/v1/health").json()
        for key in ("status", "geo_model_loaded", "temporal_model_loaded",
                    "api_version", "timestamp"):
            assert key in data, f"Missing key: {key}"

    def test_health_status_is_valid_literal(self):
        client = _get_client()
        data = client.get("/v1/health").json()
        assert data["status"] in ("ok", "degraded")


# ── Geographic Prediction ─────────────────────────────────────

class TestGeographicPrediction:

    def test_valid_chicago_coords_returns_200(self):
        client = _get_client()
        with patch("api.predictor.get_geo_model", return_value=_geo_mock()), \
             patch("api.predictor.get_cluster_profile", return_value=_geo_profile()):
            r = client.post("/v1/predict/geographic",
                            json={"lat": 41.85, "lon": -87.65})
        assert r.status_code == 200

    def test_valid_response_has_all_schema_fields(self):
        client = _get_client()
        with patch("api.predictor.get_geo_model", return_value=_geo_mock()), \
             patch("api.predictor.get_cluster_profile", return_value=_geo_profile()):
            r = client.post("/v1/predict/geographic",
                            json={"lat": 41.85, "lon": -87.65})
        assert r.status_code == 200
        data = r.json()
        for key in ("cluster_id", "cluster_label", "crime_risk_level",
                    "dominant_crime_type", "model_name", "model_version",
                    "prediction_timestamp"):
            assert key in data, f"Missing response key: {key}"
        assert data["crime_risk_level"] in ("HIGH", "MEDIUM", "LOW")

    def test_latitude_outside_chicago_returns_422(self):
        client = _get_client()
        r = client.post("/v1/predict/geographic",
                        json={"lat": 99.0, "lon": -87.65})
        assert r.status_code == 422

    def test_longitude_outside_chicago_returns_422(self):
        client = _get_client()
        r = client.post("/v1/predict/geographic",
                        json={"lat": 41.85, "lon": 0.0})
        assert r.status_code == 422

    def test_missing_field_returns_422(self):
        client = _get_client()
        r = client.post("/v1/predict/geographic", json={"lat": 41.85})
        assert r.status_code == 422

    def test_model_unavailable_returns_503(self):
        client = _get_client()
        with patch("api.predictor.get_geo_model",
                   side_effect=FileNotFoundError("model not found")):
            r = client.post("/v1/predict/geographic",
                            json={"lat": 41.85, "lon": -87.65})
        assert r.status_code == 503

    def test_cluster_not_in_profile_returns_500(self):
        """Model predicts cluster 99 which is absent from profile."""
        client = _get_client()
        mock = _geo_mock()
        mock.predict.return_value = np.array([99])
        with patch("api.predictor.get_geo_model", return_value=mock), \
             patch("api.predictor.get_cluster_profile", return_value=_geo_profile()):
            r = client.post("/v1/predict/geographic",
                            json={"lat": 41.85, "lon": -87.65})
        assert r.status_code == 500


# ── Temporal Prediction ───────────────────────────────────────

class TestTemporalPrediction:

    def test_valid_request_returns_200(self):
        client = _get_client()
        with patch("api.predictor.get_temporal_model", return_value=_temporal_mock()):
            r = client.post("/v1/predict/temporal",
                            json={"hour": 22, "day_of_week": 5, "month": 7})
        assert r.status_code == 200

    def test_hour_out_of_range_returns_422(self):
        client = _get_client()
        r = client.post("/v1/predict/temporal",
                        json={"hour": 25, "day_of_week": 1, "month": 6})
        assert r.status_code == 422

    def test_month_out_of_range_returns_422(self):
        client = _get_client()
        r = client.post("/v1/predict/temporal",
                        json={"hour": 10, "day_of_week": 1, "month": 13})
        assert r.status_code == 422

    def test_temporal_model_unavailable_returns_503(self):
        client = _get_client()
        with patch("api.predictor.get_temporal_model",
                   side_effect=FileNotFoundError("model not found")):
            r = client.post("/v1/predict/temporal",
                            json={"hour": 22, "day_of_week": 5, "month": 7})
        assert r.status_code == 503


# ── Admin Endpoint ────────────────────────────────────────────

class TestAdminReloadModels:

    def test_missing_api_key_returns_503(self):
        """Endpoint disabled if PATROLIQ_API_KEY not configured."""
        client = _get_client()
        with patch.dict("os.environ", {}, clear=False):
            # Ensure env var is absent
            import os
            os.environ.pop("PATROLIQ_API_KEY", None)
            r = client.post("/v1/admin/reload-models")
        assert r.status_code == 503

    def test_wrong_api_key_returns_403(self):
        client = _get_client()
        with patch.dict("os.environ", {"PATROLIQ_API_KEY": "correct-key"}):
            r = client.post("/v1/admin/reload-models",
                            headers={"X-API-Key": "wrong-key"})
        assert r.status_code == 403

    def test_correct_api_key_returns_200(self):
        client = _get_client()
        # Patch where main.py holds the reference (top-level import)
        with patch("api.main.clear_model_cache") as mock_clear, \
             patch.dict("os.environ", {"PATROLIQ_API_KEY": "test-key"}):
            r = client.post("/v1/admin/reload-models",
                            headers={"X-API-Key": "test-key"})
        assert r.status_code == 200
        mock_clear.assert_called_once()
        assert "message" in r.json()


# ── Metrics ───────────────────────────────────────────────────

class TestMetricsEndpoint:

    def test_metrics_endpoint_returns_200(self):
        client = _get_client()
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "prediction_requests_total" in response.text or \
               "model_version_info" in response.text
