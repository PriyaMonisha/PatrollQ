# filename: api/main.py
# purpose:  PatrolIQ FastAPI inference server — geographic + temporal cluster prediction
#           Exposes /metrics endpoint for Prometheus scraping.

import time
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

from api.predictor import predict_geographic, predict_temporal
from api.schemas import (
    DriftFeatureResult,
    DriftReportResponse,
    GeoRequest,
    GeoResponse,
    HealthResponse,
    TemporalRequest,
    TemporalResponse,
)

# ── Prometheus metrics ────────────────────────────────────────

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests",
    ["endpoint", "cluster_id"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction request latency",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

DRIFT_DETECTED = Counter(
    "drift_detected_total",
    "Total drift detection events",
    ["feature"],
)

MODEL_VERSION = Gauge(
    "model_version_info",
    "Model version information",
    ["model_name", "version"],
)

# Set static model version gauges at startup
MODEL_VERSION.labels(model_name="kmeans_geo", version="v1").set(1)
MODEL_VERSION.labels(model_name="kmeans_temporal", version="v1").set(1)

# ── App setup ────────────────────────────────────────────────

app = FastAPI(
    title="PatrolIQ Crime Intelligence API",
    description="Geographic and temporal crime cluster prediction for Chicago PD",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────

@app.get("/metrics", include_in_schema=False)
def metrics():
    """Prometheus metrics endpoint — scraped by prometheus service."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/v1/health", response_model=HealthResponse, tags=["Health"])
def health():
    """Service health check — verifies model artifacts are loadable."""
    geo_ok = True
    temporal_ok = True
    try:
        from api.predictor import get_geo_labels
        get_geo_labels()
    except Exception:
        geo_ok = False
    try:
        from api.predictor import get_temporal_model
        get_temporal_model()
    except Exception:
        temporal_ok = False

    return HealthResponse(
        status="ok" if (geo_ok and temporal_ok) else "degraded",
        geo_model_loaded=geo_ok,
        temporal_model_loaded=temporal_ok,
        api_version="1.0.0",
    )


@app.post("/v1/predict/geographic", response_model=GeoResponse, tags=["Prediction"])
def predict_geo(request: GeoRequest):
    """
    Predict geographic crime cluster for a given coordinate.
    Uses nearest-neighbour lookup on pre-trained K-Means (k=8) labels.
    """
    start = time.perf_counter()
    try:
        result = predict_geographic(request.lat, request.lon)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency = time.perf_counter() - start
    PREDICTION_LATENCY.labels(endpoint="geographic").observe(latency)
    PREDICTION_REQUESTS.labels(
        endpoint="geographic", cluster_id=str(result["cluster_id"])
    ).inc()

    return GeoResponse(
        **result,
        prediction_timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.post("/v1/predict/temporal", response_model=TemporalResponse, tags=["Prediction"])
def predict_temp(request: TemporalRequest):
    """
    Predict temporal crime pattern cluster for a given time context.
    Uses pre-trained K-Means (k=4) on cyclical temporal features.
    """
    start = time.perf_counter()
    try:
        result = predict_temporal(
            request.hour, request.day_of_week, request.month, request.is_weekend
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    latency = time.perf_counter() - start
    PREDICTION_LATENCY.labels(endpoint="temporal").observe(latency)
    PREDICTION_REQUESTS.labels(
        endpoint="temporal", cluster_id=str(result["cluster_id"])
    ).inc()

    return TemporalResponse(
        **result,
        prediction_timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/v1/drift/report", response_model=DriftReportResponse, tags=["Monitoring"])
def drift_report():
    """
    Run KS-test and chi-squared drift detection on stored training distribution vs reference.
    Returns per-feature drift signals for all 4 monitored features.
    """
    try:
        from src.monitoring.drift import run_drift_report
        results_raw = run_drift_report()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {e}")

    results = [DriftFeatureResult(**r) for r in results_raw]

    # Increment Prometheus counter for each detected drift
    for r in results:
        if r.drift_detected:
            DRIFT_DETECTED.labels(feature=r.feature).inc()

    drifted = sum(1 for r in results if r.drift_detected)

    return DriftReportResponse(
        overall_drift_detected=drifted > 0,
        features_checked=len(results),
        features_drifted=drifted,
        results=results,
        report_timestamp=datetime.now(timezone.utc).isoformat(),
    )
