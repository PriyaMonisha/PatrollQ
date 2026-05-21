# filename: tests/test_api.py
# purpose:  Integration tests for FastAPI inference endpoints

import sys
from pathlib import Path
from unittest.mock import patch

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


class TestHealthEndpoint:

    def test_health_returns_200(self):
        client = _get_client()
        response = client.get("/v1/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self):
        client = _get_client()
        data = client.get("/v1/health").json()
        for key in ("status", "geo_model_loaded", "temporal_model_loaded", "api_version"):
            assert key in data, f"Missing key: {key}"


class TestGeographicPrediction:

    def test_valid_chicago_coords_returns_200(self):
        client = _get_client()
        response = client.post("/v1/predict/geographic",
                               json={"lat": 41.85, "lon": -87.65})
        assert response.status_code in (200, 503)  # 503 if artifacts missing

    def test_latitude_outside_chicago_returns_422(self):
        client = _get_client()
        response = client.post("/v1/predict/geographic",
                               json={"lat": 99.0, "lon": -87.65})
        assert response.status_code == 422

    def test_longitude_outside_chicago_returns_422(self):
        client = _get_client()
        response = client.post("/v1/predict/geographic",
                               json={"lat": 41.85, "lon": 0.0})
        assert response.status_code == 422

    def test_missing_field_returns_422(self):
        client = _get_client()
        response = client.post("/v1/predict/geographic", json={"lat": 41.85})
        assert response.status_code == 422

    def test_valid_response_structure(self):
        client = _get_client()
        response = client.post("/v1/predict/geographic",
                               json={"lat": 41.85, "lon": -87.65})
        if response.status_code == 200:
            data = response.json()
            for key in ("cluster_id", "cluster_label", "crime_risk_level",
                        "model_name", "prediction_timestamp"):
                assert key in data, f"Missing response key: {key}"


class TestTemporalPrediction:

    def test_valid_request_returns_200_or_503(self):
        client = _get_client()
        response = client.post("/v1/predict/temporal",
                               json={"hour": 22, "day_of_week": 5, "month": 7})
        assert response.status_code in (200, 503)

    def test_hour_out_of_range_returns_422(self):
        client = _get_client()
        response = client.post("/v1/predict/temporal",
                               json={"hour": 25, "day_of_week": 1, "month": 6})
        assert response.status_code == 422

    def test_month_out_of_range_returns_422(self):
        client = _get_client()
        response = client.post("/v1/predict/temporal",
                               json={"hour": 10, "day_of_week": 1, "month": 13})
        assert response.status_code == 422


class TestMetricsEndpoint:

    def test_metrics_endpoint_returns_200(self):
        client = _get_client()
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "prediction_requests_total" in response.text or \
               "model_version_info" in response.text
